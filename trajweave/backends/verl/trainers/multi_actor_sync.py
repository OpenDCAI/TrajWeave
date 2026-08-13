from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import torch
import transfer_queue as tq
from omegaconf import OmegaConf, open_dict
from transfer_queue import KVBatchMeta

from trajweave.backends.verl.extensions.drmas.agent_wise_grpo import TrajWeaveActorRolloutRefWorker
from trajweave.backends.verl.routing import safe_actor_role_key, split_tq_batch_by_field
from trajweave.backends.verl.schema import to_python
from trajweave.backends.verl.tokenizer_compat import assert_compatible_tokenizers
from trajweave.backends.verl.weight_sync import sync_hf_local_rollout_weights
from verl import DataProto
from verl.single_controller.ray import (
    RayClassWithInitArgs,
    RayWorkerGroup,
    ResourcePoolManager,
    create_colocated_worker_cls,
)
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.padding_utils import upsample_batch_to_divisible_size
from verl.trainer.ppo.utils import Role
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
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.seqlen_balancing import (
    calculate_workload,
    get_seqlen_balanced_partitions,
    log_seqlen_unbalance,
)
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
    gpus: int


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


@register_trainer("trajweave_multi_actor_sync")
class TrajWeaveMultiActorSyncTrainer(PPOTrainerSync):
    """按 ``worker_group`` 将 Actor 计算和更新路由到多个 VERL Worker Group。"""

    def _setup(self):
        self._init_multi_actor_specs()
        self._align_global_actor_assets()
        self._validate_multi_actor_specs()
        self._init_tokenizer()
        self._init_dataloader()
        self._init_dump_executor()
        self._init_resource_pool_mgr()
        self.resource_pool_manager.create_resource_pool()
        self._create_worker_groups()
        self._init_runtime_managers()
        self._load_checkpoint()
        logger.info(
            "TrajWeave multi-actor trainer initialized for recipe=%s, groups=%s",
            self._recipe_name(),
            sorted(self.actor_rollout_wgs),
        )

    def _align_global_actor_assets(self) -> None:
        """Make VERL's single global model/tokenizer represent the primary actor group."""

        primary_group = self.multi_actor_worker_group_specs[self.multi_actor_trainable_group_ids[0]]
        if not primary_group.model_path or not primary_group.tokenizer_path:
            return
        model_config = self.config.actor_rollout_ref.model
        previous_model = str(model_config.get("path") or "")
        previous_tokenizer = str(model_config.get("tokenizer_path") or previous_model)
        with open_dict(model_config):
            model_config.path = primary_group.model_path
            model_config.tokenizer_path = primary_group.tokenizer_path
        if previous_model != primary_group.model_path or previous_tokenizer != primary_group.tokenizer_path:
            logger.info(
                "Aligned VERL global actor assets with primary worker group %s (model=%s, tokenizer=%s).",
                primary_group.group_id,
                primary_group.model_path,
                primary_group.tokenizer_path,
            )

    def _init_tokenizer(self):
        """Honor ``model.tokenizer_path`` instead of loading VERL's model path as a tokenizer."""

        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.fs import copy_to_local

        model_config = self.config.actor_rollout_ref.model
        tokenizer_source = model_config.get("tokenizer_path") or model_config.path
        local_tokenizer_path = copy_to_local(
            tokenizer_source,
            use_shm=model_config.get("use_shm", False),
        )
        trust_remote_code = self.config.data.get("trust_remote_code", False)
        self.tokenizer = hf_tokenizer(local_tokenizer_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(local_tokenizer_path, trust_remote_code=trust_remote_code, use_fast=True)

    def _init_multi_actor_specs(self) -> None:
        groups = _worker_groups_from_config(self.config)
        if len(groups) < 2:
            raise ValueError("TrajWeave multi-actor trainer requires at least two worker groups.")
        role_keys = [group.role_key for group in groups]
        if len(role_keys) != len(set(role_keys)):
            raise ValueError(f"Worker group ids produce duplicate safe role keys: {role_keys}")
        self.multi_actor_worker_group_specs = {group.group_id: group for group in groups}
        self.multi_actor_trainable_group_ids = [group.group_id for group in groups if group.trainable]
        if not self.multi_actor_trainable_group_ids:
            raise ValueError("TrajWeave multi-actor trainer requires at least one trainable worker group.")
        configured_model_ids = [str(item) for item in to_python(self.config.get("agent", {}).get("model_ids", []))]
        frozen_model_ids = sorted(set(configured_model_ids) - set(self.multi_actor_trainable_group_ids))
        if frozen_model_ids:
            raise ValueError(
                "TrajWeave multi-actor training requires every agent model_id to reference a trainable "
                f"worker group; frozen model_ids are not routed to PPO: {frozen_model_ids}."
            )

    def _validate_multi_actor_specs(self) -> None:
        tokenizer_mode = str(OmegaConf.select(self.config, "trajweave.multi_actor.tokenizer_mode") or "shared")
        if tokenizer_mode not in {"shared", "compatible"}:
            raise ValueError(
                "TrajWeave multi-actor trainer supports "
                "trajweave.multi_actor.tokenizer_mode in {'shared', 'compatible'}."
            )

        missing_fields: list[str] = []
        tokenizer_paths: list[str] = []
        for group_id in self.multi_actor_trainable_group_ids:
            group = self.multi_actor_worker_group_specs[group_id]
            if not group.model_path:
                missing_fields.append(f"{group_id}.model_path")
            if not group.tokenizer_path:
                missing_fields.append(f"{group_id}.tokenizer_path")
            else:
                tokenizer_paths.append(group.tokenizer_path)
        if missing_fields:
            raise ValueError(
                "TrajWeave multi-actor trainer requires explicit model_path and tokenizer_path for every "
                f"trainable worker group; missing: {missing_fields}."
            )
        unique_tokenizers = sorted(set(tokenizer_paths))
        if tokenizer_mode == "shared" and len(unique_tokenizers) != 1:
            raise ValueError(
                "TrajWeave shared-tokenizer multi-actor trainer requires identical tokenizer_path across "
                f"trainable worker groups. Got tokenizer paths: {unique_tokenizers}."
            )
        if tokenizer_mode == "compatible":
            assert_compatible_tokenizers(unique_tokenizers)
        if self.use_reference_policy and not self.config.actor_rollout_ref.model.get("lora_adapter_path"):
            raise ValueError(
                "TrajWeave multi-actor currently supports reference policy only when ref is inside actor. "
                "Set algorithm.use_kl_in_reward=false for the current TrajWeave multi-actor path."
            )
        if int(self.config.trainer.nnodes) != 1:
            raise ValueError("TrajWeave per-group GPU pools currently support trainer.nnodes=1 only.")
        requested_gpus = sum(
            self.multi_actor_worker_group_specs[group_id].gpus for group_id in self.multi_actor_trainable_group_ids
        )
        available_gpus = int(self.config.trainer.n_gpus_per_node)
        if requested_gpus > available_gpus:
            raise ValueError(
                "TrajWeave trainable worker groups request more GPUs than trainer.n_gpus_per_node: "
                f"requested={requested_gpus}, available={available_gpus}."
            )

    def _init_resource_pool_mgr(self):
        self.role_worker_mapping = {}
        self.mapping = {}
        resource_pool_spec = {
            self.multi_actor_worker_group_specs[group_id].role_key: [self.multi_actor_worker_group_specs[group_id].gpus]
            for group_id in self.multi_actor_trainable_group_ids
        }
        self.resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec,
            mapping=self.mapping,
            max_colocate_count=2 if self.use_critic else 1,
        )

    def _create_worker_groups(self) -> None:
        critic_class = None
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
            critic_class = RayClassWithInitArgs(cls=__import__("ray").remote(TrainingWorker), config=worker_cfg)
            self._trajweave_critic_cfg = critic_cfg

        wg_kwargs = {"device_name": self.config.trainer.device}
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )

        self.actor_rollout_wgs = {}
        critic_wg = None
        for index, group_id in enumerate(self.multi_actor_trainable_group_ids):
            group = self.multi_actor_worker_group_specs[group_id]
            class_dict = {
                group.role_key: RayClassWithInitArgs(
                    cls=__import__("ray").remote(TrajWeaveActorRolloutRefWorker),
                    config=_actor_rollout_ref_config_for_group(self.config, group),
                    distillation_config=self.config.get("distillation"),
                    role=str(Role.Actor),
                )
            }
            if index == 0 and critic_class is not None:
                class_dict[str(Role.Critic)] = critic_class
            resource_pool = self.resource_pool_manager.resource_pool_dict[group.role_key]
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, **wg_kwargs)
            spawned = wg_dict.spawn(prefix_set=class_dict.keys())
            wg = spawned[group.role_key]
            wg.init_model()
            self.actor_rollout_wgs[group_id] = wg
            if index == 0 and self.use_critic:
                critic_wg = spawned[str(Role.Critic)]
        self.actor_rollout_wg = self.actor_rollout_wgs[self.multi_actor_trainable_group_ids[0]]

        if self.use_critic:
            if critic_wg is None:
                raise RuntimeError("TrajWeave multi-actor critic worker group was not created.")
            self.critic_wg = critic_wg
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
            raise ValueError("TrajWeave multi-actor trainer does not support teacher policy yet.")
        self.teacher_model_manager = None
        self.distillation_config = None
        self.llm_server_manager = _NullLLMServerManager()
        self.checkpoint_manager = _NullCheckpointEngineManager()
        self.checkpoint_manager.sleep_replicas()

    def _compute_old_log_prob(self, batch, metrics: dict):
        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
        bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
        if bypass_recomputing_logprobs:
            data = tq.kv_batch_get(
                keys=batch.keys, partition_id=batch.partition_id, select_fields=["rollout_log_probs"]
            )
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
        tq.kv_batch_put(
            keys=batch.keys, partition_id=batch.partition_id, fields=data.select("old_log_probs", "entropy")
        )

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

    def _balance_batch(self, batch, metrics: dict, logging_prefix="global_seqlen", keep_minibatch=False):
        """Pad and balance each actor route independently across its own data-parallel ranks."""

        del keep_minibatch
        balanced_routes = []
        for routed in split_tq_batch_by_field(batch, field="worker_group"):
            if routed.group_id not in self.actor_rollout_wgs:
                raise KeyError(f"Unknown worker_group {routed.group_id!r} during multi-actor balancing.")
            worker_group = self.actor_rollout_wgs[routed.group_id]
            dp_size = _actor_data_parallel_size(worker_group)
            balanced = upsample_batch_to_divisible_size(
                routed.batch,
                dp_size,
                self.tokenizer.eos_token_id,
            )
            sequence_lengths = torch.tensor([tag["seq_len"] for tag in balanced.tags], dtype=torch.int64)
            workloads = calculate_workload(sequence_lengths)
            partitions = get_seqlen_balanced_partitions(workloads, k_partitions=dp_size, equal_size=True)
            balanced.reorder([row for partition in partitions for row in partition])
            metrics.update(
                log_seqlen_unbalance(
                    seqlen_list=sequence_lengths.tolist(),
                    partitions=partitions,
                    prefix=f"{logging_prefix}/{_safe_metric_name(routed.group_id)}",
                )
            )
            balanced_routes.append(balanced)
        return KVBatchMeta(
            keys=[key for routed in balanced_routes for key in routed.keys],
            tags=[tag for routed in balanced_routes for tag in routed.tags],
            partition_id=batch.partition_id,
            fields=batch.fields,
            extra_info=batch.extra_info,
        )

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

        data = tq.kv_batch_get(
            keys=batch.keys, partition_id=batch.partition_id, select_fields=["log_probs", "response_mask"]
        )
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
        routed_batches = self._route_batch(batch)
        routed_counts = {routed.group_id: _batch_len(routed.batch) for routed in routed_batches}
        for group_id, sample_count in routed_counts.items():
            metric_name = f"trajweave/{self._metric_namespace()}/actor_groups/{_safe_metric_name(group_id)}/samples"
            metrics[metric_name] = sample_count

        missing_trainable_groups = [
            group_id for group_id in self.multi_actor_trainable_group_ids if routed_counts.get(group_id, 0) == 0
        ]
        if missing_trainable_groups:
            raise RuntimeError(
                "TrajWeave synchronous multi-actor step is missing samples for trainable worker groups: "
                f"{missing_trainable_groups}. No actor was updated."
            )
        metrics[f"trajweave/{self._metric_namespace()}/actor_groups/missing_trainable"] = len(missing_trainable_groups)

        for routed in routed_batches:
            if routed.group_id not in self.multi_actor_trainable_group_ids:
                continue
            routed.batch.extra_info.update(batch.extra_info)
            routed.batch.extra_info["global_batch_size"] = _batch_len(routed.batch)
            routed.batch.extra_info["mini_batch_size"] = None
            routed.batch.extra_info["num_mini_batch"] = 1
            output = self._actor_wg(routed.group_id).update_actor(routed.batch)
            output = rename_dict(output["metrics"], f"actor/{routed.group_id}/")
            if f"actor/{routed.group_id}/mfu" in output:
                output[f"perf/mfu/actor/{routed.group_id}"] = output.pop(f"actor/{routed.group_id}/mfu")
            reduced = reduce_metrics(output)
            grouped_metrics.update(reduced)
            updated_groups.append(routed.group_id)
            grouped_metrics[
                f"trajweave/{self._metric_namespace()}/actor_groups/{_safe_metric_name(routed.group_id)}/updated"
            ] = 1

        metrics.update(grouped_metrics)
        metrics[f"trajweave/{self._metric_namespace()}/actor_groups/updated"] = len(updated_groups)
        metrics[f"trajweave/{self._metric_namespace()}/actor_groups/total"] = len(self.actor_rollout_wgs)
        if not updated_groups:
            raise RuntimeError("TrajWeave multi-actor update did not update any trainable worker group.")
        return batch

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )
        local_mkdir_safe(local_global_step_folder)
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None)
        for group_id, wg in self.actor_rollout_wgs.items():
            if group_id not in self.multi_actor_trainable_group_ids:
                continue
            actor_local_path = os.path.join(local_global_step_folder, "actors", _safe_path_name(group_id))
            wg.save_checkpoint(actor_local_path, None, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)
        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            self.critic_wg.save_checkpoint(critic_local_path, None, self.global_steps)
        torch.save(self.train_dataloader.state_dict(), os.path.join(local_global_step_folder, "data.pt"))
        with open(os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"), "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        self.global_steps = 0
        mode = str(self.config.trainer.resume_mode)
        if mode == "disable":
            return
        resume_from_path = self.config.trainer.get("resume_from_path", None)
        marker = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        if mode == "auto" and not resume_from_path and not os.path.exists(marker):
            logger.info("No multi-actor checkpoint marker found; training from scratch.")
            return
        marker_exists = os.path.exists(marker)
        raise ValueError(
            "TrajWeave multi-actor checkpoint resume is not implemented. "
            f"Refusing resume_mode={mode!r}, resume_from_path={resume_from_path!r}, "
            f"marker_exists={marker_exists}."
        )

    def _route_batch(self, batch):
        routed = split_tq_batch_by_field(batch, field="worker_group")
        routed_key_count = sum(_batch_len(item.batch) for item in routed)
        total_key_count = _batch_len(batch)
        if routed_key_count != total_key_count:
            raise ValueError(
                f"worker_group routing lost batch keys: routed {routed_key_count}, expected {total_key_count}."
            )
        for item in routed:
            if item.group_id not in self.actor_rollout_wgs:
                known = ", ".join(sorted(self.actor_rollout_wgs))
                raise KeyError(f"Unknown worker_group {item.group_id!r}. Known groups: {known}.")
            dp_size = _actor_data_parallel_size(self.actor_rollout_wgs[item.group_id])
            if _batch_len(item.batch) % dp_size != 0:
                raise ValueError(
                    "Routed batch must be divisible by its actor data-parallel size; "
                    f"group={item.group_id!r}, samples={_batch_len(item.batch)}, dp_size={dp_size}. "
                    "Increase rollout.n/train_batch_size or reduce worker_groups.<id>.gpus."
                )
        return routed

    def _actor_wg(self, group_id: str):
        return self.actor_rollout_wgs[group_id]

    def _recipe_name(self) -> str:
        value = OmegaConf.select(self.config, "trajweave.recipe")
        return str(value or "unknown")

    def _metric_namespace(self) -> str:
        value = OmegaConf.select(self.config, "trajweave.multi_actor.metric_namespace")
        if value:
            return _safe_metric_name(str(value))
        recipe = self._recipe_name()
        return _safe_metric_name(recipe.split("_", 1)[0])

    def on_init_end(self):
        if getattr(self, "agent_loop_manager", None) is not None:
            self.checkpoint_manager.update_weights(self.global_steps)

    def on_step_end(self):
        with marked_timer("update_weights", self.timing_raw, color="red"):
            sync_hf_local_rollout_weights(self)


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
            raise ValueError(f"Invalid worker group entry: {raw!r}")
        group_id = str(group["id"])
        groups.append(
            WorkerGroupConfig(
                group_id=group_id,
                role_key=safe_actor_role_key(group_id),
                model_path=str(group["model_path"]) if group.get("model_path") else None,
                tokenizer_path=str(group["tokenizer_path"]) if group.get("tokenizer_path") else None,
                trainable=bool(group.get("trainable", True)),
                gpus=_positive_gpu_count(group.get("gpus", 1), group_id=group_id),
            )
        )
    return groups


def _actor_rollout_ref_config_for_group(config: Any, group: WorkerGroupConfig) -> Any:
    actor_config = OmegaConf.create(OmegaConf.to_container(config.actor_rollout_ref, resolve=True))
    with open_dict(actor_config):
        if group.model_path:
            actor_config.model.path = group.model_path
        if group.tokenizer_path:
            actor_config.model.tokenizer_path = group.tokenizer_path
    return actor_config


def _safe_path_name(value: str) -> str:
    return safe_actor_role_key(value).removeprefix("trajweave_actor_")


def _safe_metric_name(value: str) -> str:
    return _safe_path_name(value)


def _batch_len(batch: Any) -> int:
    if hasattr(batch, "keys"):
        return len(batch.keys)
    return len(batch)


def _positive_gpu_count(value: Any, *, group_id: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Worker group {group_id!r} gpus must be a positive integer.")
    try:
        gpus = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Worker group {group_id!r} gpus must be a positive integer.") from exc
    if gpus <= 0:
        raise ValueError(f"Worker group {group_id!r} gpus must be a positive integer.")
    return gpus


def _actor_data_parallel_size(worker_group: Any) -> int:
    dispatch_info = getattr(worker_group, "_dispatch_info", None)
    query = getattr(worker_group, "_query_dispatch_info", None)
    if isinstance(dispatch_info, dict) and callable(query):
        role = "actor"
        mapping = dispatch_info.get(role)
        if mapping is None:
            mapping = query(role)
            dispatch_info[role] = mapping
        if mapping:
            return max(int(rank) for rank in mapping) + 1
    world_size = int(getattr(worker_group, "world_size", 1))
    if world_size < 1:
        raise ValueError("Actor worker-group world_size must be positive.")
    return world_size
