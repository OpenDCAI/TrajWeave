from __future__ import annotations

import asyncio
import json
import inspect
import logging
import os
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
import transfer_queue as tq
from omegaconf import OmegaConf, open_dict

from trajweave.backends.verl.extensions.drmas.agent_wise_grpo import TrajWeaveActorRolloutRefWorker
from trajweave.backends.verl.llm_routing import GroupedLLMServerClient
from trajweave.backends.verl.routing import safe_worker_role_key, split_tq_batch_by_field
from trajweave.backends.verl.schema import to_python
from trajweave.backends.verl.weight_sync import (
    CheckpointEnginePolicyEndpoint,
    GroupedPolicyWeightTransport,
    MultiActorWeightSyncContract,
)
from trajweave.backends.verl.async_buffer import PolicyBufferCoordinator
from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager, create_colocated_worker_cls
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.utils import Role, need_critic, need_teacher_policy
from verl.trainer.ppo.v1.trainer_base import (
    TrainingWorkerConfig,
    register_trainer,
    response_from_nested,
    value_loss,
)
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.debug.metrics import calculate_debug_metrics
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.workers.engine_workers import TrainingWorker

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@dataclass(frozen=True)
class WorkerGroupConfig:
    group_id: str
    role_key: str
    model_path: str | None
    tokenizer_path: str | None
    trainable: bool


class _NullLLMServerManager:
    def get_client(self):
        return None

    def get_replicas(self):
        return []


class _NullCheckpointEngineManager:
    def sleep_replicas(self):
        return None

    def update_weights(self, global_steps: int | None = None):
        return None


@register_trainer("trajweave_maporl_multi_actor_sync")
class TrajWeaveMAPoRLMultiActorSyncTrainer(PPOTrainerSync):
    """MAPoRL trainer that routes PPO actor work to multiple VERL actor worker groups."""

    def _setup(self):
        self._init_tokenizer()
        self._init_dataloader()
        self._init_dump_executor()
        self._init_multi_actor_specs()
        self._validate_multi_actor_specs()
        self._init_resource_pool_mgr()
        self.resource_pool_manager.create_resource_pool()
        self._create_worker_groups()
        self._init_runtime_managers()
        self._load_checkpoint()
        logger.info("TrajWeave MAPoRL multi-actor trainer initialized for groups: %s", sorted(self.actor_rollout_wgs))

    def _init_multi_actor_specs(self) -> None:
        groups = _worker_groups_from_config(self.config)
        if len(groups) < 2:
            raise ValueError("MAPoRL multi-actor trainer requires at least two worker groups.")
        role_keys = [group.role_key for group in groups]
        if len(role_keys) != len(set(role_keys)):
            raise ValueError(f"MAPoRL worker group ids produce duplicate safe role keys: {role_keys}")
        self.maporl_worker_group_specs = {group.group_id: group for group in groups}
        self.maporl_trainable_group_ids = [group.group_id for group in groups if group.trainable]
        if not self.maporl_trainable_group_ids:
            raise ValueError("MAPoRL multi-actor trainer requires at least one trainable worker group.")
        self.maporl_weight_sync = MultiActorWeightSyncContract(tuple(self.maporl_worker_group_specs))
        trajweave_cfg = self.config.get("trajweave", {}) or {}
        reward_range = trajweave_cfg.get("dynamic_filter_reward_range")
        if reward_range is not None:
            reward_range = (float(reward_range[0]), float(reward_range[1]))
        self.maporl_buffer_coordinator = PolicyBufferCoordinator(
            tuple(self.maporl_worker_group_specs),
            min_batch_size=int(trajweave_cfg.get("buffer_min_batch_size", 1)),
            reward_range=reward_range,
            max_policy_lag=int(trajweave_cfg.get("max_policy_lag", 1)),
        )

    def _validate_multi_actor_specs(self) -> None:
        tokenizers = {
            group.tokenizer_path or group.model_path or str(self.config.actor_rollout_ref.model.path)
            for group in self.maporl_worker_group_specs.values()
        }
        if self.use_critic and len(tokenizers) > 1:
            raise ValueError(
                "MAPoRL multi-actor with shared critic requires identical tokenizer_path across worker groups. "
                "Disable critic or use compatible tokenizer paths before enabling different-tokenizer training."
            )
        if self.use_reference_policy and not self.config.actor_rollout_ref.model.get("lora_adapter_path"):
            raise ValueError(
                "MAPoRL multi-actor currently supports reference policy only when ref is inside actor. "
                "Set algorithm.use_kl_in_reward=false for the current TrajWeave multi-actor path."
            )

    def _init_resource_pool_mgr(self):
        config = self.config
        self.role_worker_mapping = {}
        self.mapping = {}
        actor_count = len(self.maporl_worker_group_specs)
        colocate_count = max(3, actor_count + int(need_critic(config)) + int(need_teacher_policy(config)))
        global_pool_id = "global_pool"
        resource_pool_spec = {global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
        self.resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec,
            mapping=self.mapping,
            max_colocate_count=colocate_count,
        )

    def _create_worker_groups(self) -> None:
        resource_pool = self.resource_pool_manager.resource_pool_dict["global_pool"]
        class_dict = {}
        actor_role = _multi_actor_worker_role(self.config)
        for group in self.maporl_worker_group_specs.values():
            class_dict[group.role_key] = RayClassWithInitArgs(
                cls=__import__("ray").remote(TrajWeaveActorRolloutRefWorker),
                config=_actor_rollout_ref_config_for_group(self.config, group),
                distillation_config=self.config.get("distillation"),
                role=str(actor_role),
            )

        if self.use_critic:
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cfg.engine.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
            critic_cfg.engine.max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
            worker_cfg = TrainingWorkerConfig(
                model_type="value_model",
                model_config=getattr(critic_cfg, "model_config", None) or critic_cfg.model,
                engine_config=critic_cfg.engine,
                optimizer_config=critic_cfg.optim,
                checkpoint_config=critic_cfg.checkpoint,
            )
            class_dict[str(Role.Critic)] = RayClassWithInitArgs(cls=__import__("ray").remote(TrainingWorker), config=worker_cfg)
            self._trajweave_critic_cfg = critic_cfg

        wg_kwargs = {"device_name": self.config.trainer.device}
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )

        worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
        wg_dict = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, **wg_kwargs)
        spawned = wg_dict.spawn(prefix_set=class_dict.keys())

        self.actor_rollout_wgs = {}
        for group_id, group in self.maporl_worker_group_specs.items():
            wg = spawned[group.role_key]
            wg.init_model()
            self.actor_rollout_wgs[group_id] = wg
        self.actor_rollout_wg = self.actor_rollout_wgs[self.maporl_trainable_group_ids[0]]

        if self.use_critic:
            self.critic_wg = spawned[str(Role.Critic)]
            self.critic_wg.reset()
            value_loss_ = __import__("functools").partial(value_loss, config=self._trajweave_critic_cfg)
            self.critic_wg.set_loss_fn(value_loss_)

        self.ref_in_actor = True
        self.ref_policy_wg = None

    def _init_runtime_managers(self) -> None:
        from verl.experimental.reward_loop import RewardLoopManager

        resource_pool = None
        self.reward_loop_manager = RewardLoopManager(config=self.config, rm_resource_pool=resource_pool)

        if self.use_teacher_policy:
            raise ValueError("MAPoRL multi-actor trainer does not support teacher policy yet.")
        self.teacher_model_manager = None
        self.distillation_config = None
        self.maporl_llm_server_managers = {}
        self.maporl_checkpoint_managers = {}
        self.maporl_weight_transport = None
        if _multi_actor_vllm_enabled(self.config):
            self._init_multi_actor_vllm_endpoints()
        else:
            self.llm_server_manager = _NullLLMServerManager()
            self.checkpoint_manager = _NullCheckpointEngineManager()
            self.checkpoint_manager.sleep_replicas()

    def _init_multi_actor_vllm_endpoints(self) -> None:
        """Launch one native vLLM manager and checkpoint engine per policy group."""
        from verl.checkpoint_engine.base import CheckpointEngineManager
        from verl.workers.rollout.llm_server import LLMServerManager

        for group_index, (group_id, group) in enumerate(self.maporl_worker_group_specs.items()):
            group_config = OmegaConf.create(OmegaConf.to_container(self.config, resolve=True))
            with open_dict(group_config):
                group_config.actor_rollout_ref = _actor_rollout_ref_config_for_group(self.config, group)
                rollout_seed = group_config.actor_rollout_ref.rollout.get("seed")
                if rollout_seed is not None:
                    group_config.actor_rollout_ref.rollout.seed = int(rollout_seed) + group_index
            llm_manager = LLMServerManager(
                config=group_config,
                worker_group=self.actor_rollout_wgs[group_id],
                rollout_resource_pool=None,
            )
            _namespace_rollout_replica_class(llm_manager, group_id)
            _resolve_async(llm_manager._initialize_llm_servers(start_rank=0))
            _resolve_async(llm_manager._init_global_load_balancer())
            checkpoint_config = omega_conf_to_dataclass(group_config.actor_rollout_ref.rollout.checkpoint_engine)
            checkpoint_config.backend = "naive"
            checkpoint_manager = CheckpointEngineManager(
                config=checkpoint_config,
                actor_wg=self.actor_rollout_wgs[group_id],
                replicas=llm_manager.get_replicas(),
            )
            checkpoint_manager.sleep_replicas()
            self.maporl_llm_server_managers[group_id] = llm_manager
            self.maporl_checkpoint_managers[group_id] = checkpoint_manager

        self.llm_server_manager = GroupedLLMServerClient(
            {group_id: manager.get_client() for group_id, manager in self.maporl_llm_server_managers.items()}
        )
        self.checkpoint_manager = _NullCheckpointEngineManager()
        self.maporl_weight_transport = GroupedPolicyWeightTransport(
            {
                group_id: CheckpointEnginePolicyEndpoint(manager)
                for group_id, manager in self.maporl_checkpoint_managers.items()
            }
        )

    def _compute_old_log_prob(self, batch, metrics: dict):
        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
        bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
        if bypass_recomputing_logprobs:
            data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=["rollout_log_probs"])
            data["old_log_probs"] = data.pop("rollout_log_probs")
            tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=data)
            return batch

        batch.extra_info.update(
            {
                "calculate_entropy": True,
                "compute_loss": False,
                "temperature": self.config.actor_rollout_ref.rollout.temperature,
            }
        )
        for routed in self._route_batch(batch):
            output = self._actor_wg(routed.group_id).compute_log_prob(routed.batch)
            assert len(output) == len(routed.batch)

        fields = ["entropy", "log_probs", "response_mask"]
        if self.config.actor_rollout_ref.rollout.calculate_log_probs:
            fields.extend(["responses", "rollout_log_probs"])
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)
        data["old_log_probs"] = response_from_nested(data.pop("log_probs"), data["response_mask"])
        data["entropy"] = response_from_nested(data.pop("entropy"), data["response_mask"])
        tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=data.select("old_log_probs", "entropy"))

        data_proto = DataProto(batch=data.to_padded_tensor())
        actor_config = self.config.actor_rollout_ref.actor
        entropy_agg = agg_loss(
            loss_mat=data_proto.batch["entropy"],
            loss_mask=data_proto.batch["response_mask"],
            loss_agg_mode=actor_config.loss_agg_mode,
            loss_scale_factor=actor_config.loss_scale_factor,
        )
        metrics["actor/entropy"] = entropy_agg.detach().item()
        if self.config.actor_rollout_ref.rollout.calculate_log_probs:
            metrics.update(calculate_debug_metrics(data_proto))
        return batch

    def _compute_ref_log_prob(self, batch, metrics: dict):
        batch.extra_info.update(
            {
                "calculate_entropy": False,
                "compute_loss": False,
                "temperature": self.config.actor_rollout_ref.rollout.temperature,
                "no_lora_adapter": True,
            }
        )
        for routed in self._route_batch(batch):
            output = self._actor_wg(routed.group_id).compute_log_prob(routed.batch)
            assert len(output) == len(routed.batch)

        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=["log_probs", "response_mask"])
        data["ref_log_prob"] = response_from_nested(data.pop("log_probs"), data["response_mask"])
        tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=data.select("ref_log_prob"))
        return batch

    def _update_actor(self, batch, metrics: dict):
        actor_config = self.config.actor_rollout_ref.actor
        ppo_mini_batch_size = actor_config.ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        calculate_entropy = actor_config.calculate_entropy or (actor_config.entropy_coeff != 0.0)
        batch.extra_info.update(
            {
                "calculate_entropy": calculate_entropy,
                "distillation_use_topk": False,
                "global_batch_size": ppo_mini_batch_size,
                "mini_batch_size": ppo_mini_batch_size,
                "epochs": actor_config.ppo_epochs,
                "seed": actor_config.data_loader_seed,
                "dataloader_kwargs": {"shuffle": actor_config.shuffle},
                "temperature": self.config.actor_rollout_ref.rollout.temperature,
            }
        )

        updated_groups = []
        grouped_metrics = {}
        ready_keys = self._async_buffer_ready_keys(batch) if self._async_buffer_enabled() else None
        for routed in self._route_batch(batch):
            if routed.group_id not in self.maporl_trainable_group_ids:
                continue
            routed_batch = routed.batch
            if ready_keys is not None:
                selected_keys = [key for key in routed.batch.keys if key in ready_keys]
                if not selected_keys:
                    continue
                routed_batch = routed.batch.select_keys(selected_keys)
            routed_batch.extra_info.update(batch.extra_info)
            routed_batch.extra_info["global_batch_size"] = len(routed_batch)
            routed_batch.extra_info["mini_batch_size"] = None
            routed_batch.extra_info["num_mini_batch"] = 1
            output = self._actor_wg(routed.group_id).update_actor(routed_batch)
            output = rename_dict(output["metrics"], f"actor/{routed.group_id}/")
            if f"actor/{routed.group_id}/mfu" in output:
                output[f"perf/mfu/actor/{routed.group_id}"] = output.pop(f"actor/{routed.group_id}/mfu")
            reduced = reduce_metrics(output)
            grouped_metrics.update(reduced)
            self.maporl_weight_sync.record_actor_update(routed.group_id, self.global_steps)
            self.maporl_buffer_coordinator.update_actor_step(routed.group_id, self.global_steps)
            updated_groups.append(routed.group_id)

        metrics.update(grouped_metrics)
        metrics["trajweave/maporl/actor_groups/updated"] = len(updated_groups)
        metrics["trajweave/maporl/actor_groups/total"] = len(self.actor_rollout_wgs)
        metrics.update(self.maporl_weight_sync.metric_fields())
        metrics.update({f"trajweave/maporl/{key}": value for key, value in self.maporl_buffer_coordinator.metric_fields().items()})
        self._maporl_step_metrics = metrics
        return batch

    def _async_buffer_enabled(self) -> bool:
        return bool(OmegaConf.select(self.config, "trajweave.async_buffer.enabled", default=False))

    def _async_buffer_ready_keys(self, batch) -> set[str]:
        """Move completed trees into per-policy buffers and consume ready rows."""
        fields = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=[
                "worker_group",
                "tree_id",
                "raw_score",
                "rollout_policy_step",
                "rollout_global_step",
            ],
        )
        rows = _buffer_rows_from_fields(batch.keys, fields, global_step=self.global_steps)
        trees: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            trees.setdefault(str(row["tree_id"]), []).append(row)
        for tree_rows in trees.values():
            self.maporl_buffer_coordinator.add_tree(tree_rows, global_step=self.global_steps)

        ready_keys: set[str] = set()
        ready_groups = set(self.maporl_buffer_coordinator.ready_groups())
        for group_id in self.maporl_trainable_group_ids:
            if group_id not in ready_groups:
                continue
            for row in self.maporl_buffer_coordinator.pop_batch(group_id):
                ready_keys.add(str(row["_tq_key"]))
        return ready_keys

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )
        local_mkdir_safe(local_global_step_folder)
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None)
        for group_id, wg in self.actor_rollout_wgs.items():
            if group_id not in self.maporl_trainable_group_ids:
                continue
            actor_local_path = os.path.join(local_global_step_folder, "actors", _safe_path_name(group_id))
            wg.save_checkpoint(actor_local_path, None, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)
            with os.scandir(actor_local_path) as entries:
                model_files = sorted(
                    entry.path
                    for entry in entries
                    if entry.name.startswith("model_world_size_") and entry.name.endswith(".pt")
                )
            self.maporl_weight_sync.record_checkpoint(
                group_id,
                model_files[0] if model_files else os.path.join(actor_local_path, "model_world_size_unknown.pt"),
            )
        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            self.critic_wg.save_checkpoint(critic_local_path, None, self.global_steps)
        torch.save(self.train_dataloader.state_dict(), os.path.join(local_global_step_folder, "data.pt"))
        self._write_weight_sync_manifest()
        with open(os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"), "w") as f:
            f.write(str(self.global_steps))

    def _write_weight_sync_manifest(self) -> None:
        """Persist the latest actor/rollout versions after save or engine acknowledgement."""
        import json

        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )
        if not os.path.isdir(local_global_step_folder):
            return
        manifest_path = os.path.join(local_global_step_folder, "multi_actor_weight_sync.json")
        payload = self.maporl_weight_sync.as_dict()
        if hasattr(self, "maporl_buffer_coordinator"):
            payload["policy_buffers"] = self.maporl_buffer_coordinator.snapshot()
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")

    def _load_checkpoint(self):
        self.global_steps = 0
        resume_mode = str(self.config.trainer.resume_mode)
        if resume_mode == "disable":
            return
        if resume_mode == "auto":
            checkpoint_root = str(self.config.trainer.default_local_dir)
            if not os.path.isabs(checkpoint_root):
                checkpoint_root = os.path.join(os.getcwd(), checkpoint_root)
            global_step_folder = find_latest_ckpt_path(checkpoint_root)
            if global_step_folder is None:
                logger.info("No multi-actor checkpoint found; training from scratch")
                return
        elif resume_mode == "resume_path":
            global_step_folder = str(self.config.trainer.resume_from_path)
            if "global_step_" not in global_step_folder:
                raise ValueError("MAPoRL resume_path must specify a global_step_* directory.")
            if not os.path.isabs(global_step_folder):
                global_step_folder = os.path.join(os.getcwd(), global_step_folder)
        else:
            raise ValueError(f"Unknown MAPoRL resume mode: {resume_mode!r}")

        self.global_steps = int(global_step_folder.rsplit("global_step_", 1)[-1])
        logger.info("Resuming multi-actor training from %s at step %s", global_step_folder, self.global_steps)
        del_after_load = bool(getattr(self.config.trainer, "del_local_ckpt_after_load", False))
        for group_id, wg in self.actor_rollout_wgs.items():
            if group_id not in self.maporl_trainable_group_ids:
                continue
            actor_path = os.path.join(global_step_folder, "actors", _safe_path_name(group_id))
            if not os.path.isdir(actor_path):
                raise FileNotFoundError(f"Missing checkpoint for multi-actor group {group_id!r}: {actor_path}")
            wg.load_checkpoint(local_path=actor_path, del_local_after_load=del_after_load)
            self.maporl_weight_sync.record_actor_update(group_id, self.global_steps)
            model_files = sorted(
                os.path.join(actor_path, name)
                for name in os.listdir(actor_path)
                if name.startswith("model_world_size_") and name.endswith(".pt")
            )
            if model_files:
                self.maporl_weight_sync.record_checkpoint(group_id, model_files[0])

        manifest_path = os.path.join(global_step_folder, "multi_actor_weight_sync.json")
        if os.path.isfile(manifest_path) and hasattr(self, "maporl_buffer_coordinator"):
            try:
                with open(manifest_path, encoding="utf-8") as handle:
                    manifest = json.load(handle)
                buffer_snapshot = manifest.get("policy_buffers")
                if isinstance(buffer_snapshot, dict):
                    self.maporl_buffer_coordinator.restore_snapshot(buffer_snapshot)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                logger.warning("Could not restore policy buffer metadata from %s: %s", manifest_path, exc)

        dataloader_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_path):
            self.train_dataloader.load_state_dict(torch.load(dataloader_path, weights_only=False))
        else:
            logger.warning("No multi-actor dataloader state found at %s", dataloader_path)

    def _route_batch(self, batch):
        routed = split_tq_batch_by_field(batch, field="worker_group")
        for item in routed:
            if item.group_id not in self.actor_rollout_wgs:
                known = ", ".join(sorted(self.actor_rollout_wgs))
                raise KeyError(f"Unknown MAPoRL worker_group {item.group_id!r}. Known groups: {known}.")
        return routed

    def _actor_wg(self, group_id: str):
        return self.actor_rollout_wgs[group_id]

    def on_init_end(self):
        if self.maporl_weight_transport is None:
            self._sync_multi_actor_weights()
        else:
            self._sync_initial_multi_actor_weights()

    def on_step_end(self):
        with marked_timer("update_weights", self.timing_raw, color="red"):
            self._sync_multi_actor_weights()
        self._refresh_weight_sync_metrics()

    def _refresh_weight_sync_metrics(self) -> None:
        """Replace the pre-sync snapshot with the endpoint-acknowledged final state."""
        metrics = getattr(self, "_maporl_step_metrics", None)
        if metrics is not None:
            metrics.update(self.maporl_weight_sync.metric_fields())
            buffer_coordinator = getattr(self, "maporl_buffer_coordinator", None)
            if buffer_coordinator is not None:
                metrics.update(
                    {f"trajweave/maporl/{key}": value for key, value in buffer_coordinator.metric_fields().items()}
                )

    def on_sample_end(self):
        if self.maporl_checkpoint_managers:
            for manager in self.maporl_checkpoint_managers.values():
                manager.sleep_replicas()
        else:
            self.checkpoint_manager.sleep_replicas()

    def get_llm_client(self):
        if hasattr(self.llm_server_manager, "get_client"):
            return self.llm_server_manager.get_client()
        return self.llm_server_manager

    def _sync_multi_actor_weights(self) -> None:
        if self.maporl_weight_transport is None:
            self.checkpoint_manager.update_weights(self.global_steps)
            buffer_coordinator = getattr(self, "maporl_buffer_coordinator", None)
            for group_id in self.maporl_trainable_group_ids:
                if buffer_coordinator is not None:
                    state = buffer_coordinator.states[group_id]
                    if state.actor_step > state.rollout_synced_step:
                        buffer_coordinator.mark_rollout_sync(group_id, state.actor_step)
                weight_sync = getattr(self, "maporl_weight_sync", None)
                if weight_sync is not None:
                    version = weight_sync.as_dict().get("versions", {}).get(group_id)
                    if version is not None and int(version.get("global_step", -1)) == int(self.global_steps):
                        weight_sync.mark_rollout_sync(group_id, self.global_steps)
            if hasattr(self, "_write_weight_sync_manifest"):
                self._write_weight_sync_manifest()
            return
        results = self.maporl_weight_sync.sync_pending(self.maporl_weight_transport)
        buffer_coordinator = getattr(self, "maporl_buffer_coordinator", None)
        for result in results:
            if result.acknowledged and buffer_coordinator is not None:
                buffer_coordinator.mark_rollout_sync(result.group_id, result.global_step)
        self._write_weight_sync_manifest()
        logger.info(
            "Multi-actor vLLM weight sync step=%s results=%s",
            self.global_steps,
            [(result.group_id, result.acknowledged, result.error) for result in results],
        )

    def _sync_initial_multi_actor_weights(self) -> None:
        """Load live actor weights before the first rollout; hybrid vLLM starts with dummy weights."""
        results = []
        for group_id in self.maporl_worker_group_specs:
            version = self.maporl_weight_sync.record_actor_update(group_id, self.global_steps)
            acknowledged = bool(self.maporl_weight_transport.load_policy(version))
            if not acknowledged:
                raise RuntimeError(
                    f"Initial vLLM weight sync was not acknowledged for policy group {group_id!r}."
                )
            self.maporl_weight_sync.mark_rollout_sync(group_id, version.global_step)
            results.append((group_id, version.global_step))
        self._write_weight_sync_manifest()
        logger.info("Initial multi-actor vLLM weight sync completed: %s", results)


def _worker_groups_from_config(config: Any) -> list[WorkerGroupConfig]:
    agent_cfg = config.get("agent", {})
    raw_groups = to_python(agent_cfg.get("worker_groups", []))
    if isinstance(raw_groups, dict):
        raw_groups = [{"id": group_id, **dict(group or {})} for group_id, group in raw_groups.items()]
    if not isinstance(raw_groups, list):
        raise ValueError("agent.worker_groups must be a list of dicts or a mapping.")
    groups = []
    for raw in raw_groups:
        group = to_python(raw)
        if not isinstance(group, dict) or not group.get("id"):
            raise ValueError(f"Invalid MAPoRL worker group entry: {raw!r}")
        group_id = str(group["id"])
        groups.append(
            WorkerGroupConfig(
                group_id=group_id,
                role_key=safe_worker_role_key(group_id),
                model_path=str(group["model_path"]) if group.get("model_path") else None,
                tokenizer_path=str(group["tokenizer_path"]) if group.get("tokenizer_path") else None,
                trainable=bool(group.get("trainable", True)),
            )
        )
    return groups


def _buffer_rows_from_fields(keys: list[str], fields: Any, *, global_step: int) -> list[dict[str, Any]]:
    """Normalize TQ metadata into coordinator rows without assuming tensor types."""
    count = len(keys)

    def values(name: str, default: Any) -> list[Any]:
        raw = fields.get(name, default) if hasattr(fields, "get") else default
        raw = to_python(raw)
        if isinstance(raw, (str, bytes)) or not isinstance(raw, (list, tuple)):
            return [raw] * count
        if len(raw) != count:
            raise ValueError(f"TransferQueue field {name!r} has {len(raw)} values for {count} keys")
        return list(raw)

    groups = values("worker_group", "")
    trees = values("tree_id", None)
    scores = values("raw_score", 0.0)
    policy_steps = values("rollout_policy_step", global_step)
    global_steps = values("rollout_global_step", global_step)
    rows = []
    for index, key in enumerate(keys):
        tree_id = trees[index] if trees[index] is not None else key
        rows.append(
            {
                "_tq_key": key,
                "policy_group": str(groups[index]),
                "tree_id": str(tree_id),
                "raw_score": float(scores[index]),
                "rollout_policy_step": int(policy_steps[index]),
                "rollout_global_step": int(global_steps[index]),
            }
        )
    return rows


def _actor_rollout_ref_config_for_group(config: Any, group: WorkerGroupConfig) -> Any:
    actor_config = OmegaConf.create(OmegaConf.to_container(config.actor_rollout_ref, resolve=True))
    with open_dict(actor_config):
        actor_config.rollout.name_suffix = _safe_path_name(group.group_id)
        if group.model_path:
            actor_config.model.path = group.model_path
        if group.tokenizer_path:
            actor_config.model.tokenizer_path = group.tokenizer_path
    return actor_config


def _safe_path_name(value: str) -> str:
    return safe_worker_role_key(value).removeprefix("maporl_actor_")


def _multi_actor_vllm_enabled(config: Any) -> bool:
    return bool(OmegaConf.select(config, "trajweave.multi_actor.vllm.enabled", default=False))


def _multi_actor_worker_role(config: Any) -> Role:
    return Role.ActorRollout if _multi_actor_vllm_enabled(config) else Role.Actor


def _resolve_async(value: Any) -> Any:
    if not inspect.isawaitable(value):
        return value
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(value)
    raise RuntimeError("vLLM endpoint initialization returned an awaitable inside a running event loop.")


def _namespace_rollout_replica_class(manager: Any, group_id: str) -> None:
    """Keep each policy group's hybrid slice local while making Ray server names unique."""
    manager.rollout_replica_class = partial(
        manager.rollout_replica_class,
        name_suffix=_safe_path_name(group_id),
    )
