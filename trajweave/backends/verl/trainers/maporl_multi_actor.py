from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import torch
import transfer_queue as tq
from omegaconf import OmegaConf, open_dict

from trajweave.backends.verl.extensions.drmas.agent_wise_grpo import TrajWeaveActorRolloutRefWorker
from trajweave.backends.verl.routing import safe_worker_role_key, split_tq_batch_by_field
from trajweave.backends.verl.schema import to_python
from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager, create_colocated_worker_cls
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.utils import Role, need_critic, need_reference_policy, need_teacher_policy
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
        for group in self.maporl_worker_group_specs.values():
            class_dict[group.role_key] = RayClassWithInitArgs(
                cls=__import__("ray").remote(TrajWeaveActorRolloutRefWorker),
                config=_actor_rollout_ref_config_for_group(self.config, group),
                distillation_config=self.config.get("distillation"),
                role=str(Role.Actor),
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
        from verl.experimental.teacher_loop import MultiTeacherModelManager

        resource_pool = None
        self.reward_loop_manager = RewardLoopManager(config=self.config, rm_resource_pool=resource_pool)

        if self.use_teacher_policy:
            raise ValueError("MAPoRL multi-actor trainer does not support teacher policy yet.")
        self.teacher_model_manager = None
        self.distillation_config = None
        self.llm_server_manager = _NullLLMServerManager()
        self.checkpoint_manager = _NullCheckpointEngineManager()
        self.checkpoint_manager.sleep_replicas()

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
        for routed in self._route_batch(batch):
            if routed.group_id not in self.maporl_trainable_group_ids:
                continue
            routed.batch.extra_info.update(batch.extra_info)
            routed.batch.extra_info["global_batch_size"] = len(routed.batch)
            routed.batch.extra_info["mini_batch_size"] = None
            routed.batch.extra_info["num_mini_batch"] = 1
            output = self._actor_wg(routed.group_id).update_actor(routed.batch)
            output = rename_dict(output["metrics"], f"actor/{routed.group_id}/")
            if f"actor/{routed.group_id}/mfu" in output:
                output[f"perf/mfu/actor/{routed.group_id}"] = output.pop(f"actor/{routed.group_id}/mfu")
            reduced = reduce_metrics(output)
            grouped_metrics.update(reduced)
            updated_groups.append(routed.group_id)

        metrics.update(grouped_metrics)
        metrics["trajweave/maporl/actor_groups/updated"] = len(updated_groups)
        metrics["trajweave/maporl/actor_groups/total"] = len(self.actor_rollout_wgs)
        return batch

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
        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            self.critic_wg.save_checkpoint(critic_local_path, None, self.global_steps)
        torch.save(self.train_dataloader.state_dict(), os.path.join(local_global_step_folder, "data.pt"))
        with open(os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"), "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        self.global_steps = 0
        if self.config.trainer.resume_mode == "disable":
            return
        raise ValueError("MAPoRL multi-actor checkpoint resume is not enabled yet; use trainer.resume_mode=disable.")

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
        self.checkpoint_manager.update_weights(self.global_steps)

    def on_step_end(self):
        with marked_timer("update_weights", self.timing_raw, color="red"):
            self.checkpoint_manager.update_weights(self.global_steps)


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


def _actor_rollout_ref_config_for_group(config: Any, group: WorkerGroupConfig) -> Any:
    actor_config = OmegaConf.create(OmegaConf.to_container(config.actor_rollout_ref, resolve=True))
    with open_dict(actor_config):
        if group.model_path:
            actor_config.model.path = group.model_path
        if group.tokenizer_path:
            actor_config.model.tokenizer_path = group.tokenizer_path
    return actor_config


def _safe_path_name(value: str) -> str:
    return safe_worker_role_key(value).removeprefix("maporl_actor_")
