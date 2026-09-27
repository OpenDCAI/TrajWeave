from __future__ import annotations

import os
from typing import Any

import transfer_queue as tq
from omegaconf import OmegaConf

from trajweave.backends.verl.trainers.multi_actor_sync import (
    TrajWeaveMultiActorSyncTrainer,
    _batch_len,
    _safe_metric_name,
)
from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict


@register_trainer("trajweave_mrlx_async")
class TrajWeaveMrlXAsyncTrainer(TrajWeaveMultiActorSyncTrainer):
    """MrlX trainer with an on-policy explorer and one-step-lag adapter."""

    def _load_checkpoint(self) -> None:
        """模型恢复不足以恢复延迟样本；完整 replay 持久化前拒绝续训。"""
        self.global_steps = 0
        mode = str(self.config.trainer.resume_mode)
        if mode == "disable":
            return
        resume_path = self.config.trainer.get("resume_from_path", None)
        marker = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        if mode == "auto" and not resume_path and not os.path.exists(marker):
            return
        raise ValueError("MrlX checkpoint resume requires persisted Adapter replay and is not implemented.")

    def _init_multi_actor_specs(self) -> None:
        super()._init_multi_actor_specs()
        self.mrlx_explorer_group = str(OmegaConf.select(self.config, "trajweave.mrlx.explorer_group") or "")
        self.mrlx_adapter_group = str(OmegaConf.select(self.config, "trajweave.mrlx.adapter_group") or "")
        self._mrlx_adapter_replay = None
        self._mrlx_replay_partition = f"mrlx_adapter_replay_{id(self)}"
        self._mrlx_adapter_collected_samples = 0
        self._mrlx_adapter_updated_samples = 0

    def _validate_multi_actor_specs(self) -> None:
        super()._validate_multi_actor_specs()
        if self._recipe_name() != "mrlx_research_qa":
            raise ValueError("trajweave_mrlx_async only supports the mrlx_research_qa recipe.")
        if not self.mrlx_explorer_group or not self.mrlx_adapter_group:
            raise ValueError("MrlX trainer requires explorer_group and adapter_group.")
        if self.mrlx_explorer_group == self.mrlx_adapter_group:
            raise ValueError("MrlX explorer_group and adapter_group must be distinct.")
        configured = set(self.multi_actor_trainable_group_ids)
        expected = {self.mrlx_explorer_group, self.mrlx_adapter_group}
        if configured != expected:
            raise ValueError(
                "MrlX trainer requires exactly the configured explorer and adapter trainable groups; "
                f"expected={sorted(expected)}, configured={sorted(configured)}."
            )
        delay_steps = int(OmegaConf.select(self.config, "trajweave.mrlx.adapter_delay_steps") or 1)
        if delay_steps != 1:
            raise ValueError("MrlX trainer currently supports adapter_delay_steps=1 only.")
        if not bool(OmegaConf.select(self.config, "trajweave.mrlx.drain_adapter_replay")):
            raise ValueError("MrlX trainer requires drain_adapter_replay=true.")
        critic_warmup = int(OmegaConf.select(self.config, "trainer.critic_warmup") or 0)
        if critic_warmup != 0:
            raise ValueError(
                "MrlX trainer requires trainer.critic_warmup=0 so every training step reaches both Actor schedules."
            )

    def _init_dataloader(self) -> None:
        super()._init_dataloader()
        reachable_steps = len(self.train_dataloader) * int(self.config.trainer.total_epochs)
        if int(self.total_training_steps) < 2:
            raise ValueError("MrlX one-step Adapter lag requires at least two training steps.")
        if int(self.total_training_steps) > reachable_steps:
            raise ValueError(
                "MrlX total_training_steps exceeds the steps reachable under total_epochs; "
                f"configured={self.total_training_steps}, reachable={reachable_steps}."
            )

    def _update_actor(self, batch: Any, metrics: dict) -> Any:
        self._prepare_actor_update(batch)
        routed = {item.group_id: item.batch for item in self._route_batch(batch)}
        if self.mrlx_explorer_group not in routed:
            raise RuntimeError("MrlX step is missing current Explorer samples.")

        namespace = f"trajweave/{self._metric_namespace()}"
        explorer_batch = routed[self.mrlx_explorer_group]
        adapter_batch = routed.get(self.mrlx_adapter_group)
        metrics[f"{namespace}/explorer/collected_samples"] = _batch_len(explorer_batch)
        metrics[f"{namespace}/adapter/collected_samples"] = (
            _batch_len(adapter_batch) if adapter_batch is not None else 0
        )
        self._mrlx_adapter_collected_samples = getattr(self, "_mrlx_adapter_collected_samples", 0) + (
            _batch_len(adapter_batch) if adapter_batch is not None else 0
        )
        is_last_step = self.global_steps >= self._mrlx_final_training_step()
        replay_consumed = self._mrlx_adapter_replay is not None
        if is_last_step and adapter_batch is not None and not replay_consumed:
            raise RuntimeError(
                "MrlX cannot drain the final Adapter batch without consuming an older replay first; "
                "updating it now would violate adapter_delay_steps=1."
            )
        if (
            is_last_step
            and not replay_consumed
            and getattr(self, "_mrlx_adapter_updated_samples", 0) == 0
        ):
            raise RuntimeError(
                "MrlX training reached its final step without an Adapter update; "
                "the run did not exercise two-policy co-training."
            )

        updated_groups: list[str] = []
        metrics.update(self._update_group(self.mrlx_explorer_group, explorer_batch, phase="current"))
        updated_groups.append(self.mrlx_explorer_group)

        if replay_consumed:
            replay = self._mrlx_adapter_replay
            collected_step = int((replay.extra_info or {}).get("mrlx_collected_step", self.global_steps - 1))
            metrics.update(self._update_group(self.mrlx_adapter_group, replay, phase="delayed"))
            metrics[f"{namespace}/adapter/update_lag_steps"] = self.global_steps - collected_step
            metrics[f"{namespace}/adapter/replayed_samples"] = _batch_len(replay)
            self._mrlx_adapter_updated_samples = getattr(self, "_mrlx_adapter_updated_samples", 0) + _batch_len(
                replay
            )
            tq.kv_clear(keys=replay.keys, partition_id=replay.partition_id)
            self._mrlx_adapter_replay = None
            updated_groups.append(self.mrlx_adapter_group)
        else:
            metrics[f"{namespace}/adapter/replayed_samples"] = 0
            metrics[f"{namespace}/adapter/warmup_deferred"] = 1

        drain_replay = bool(OmegaConf.select(self.config, "trajweave.mrlx.drain_adapter_replay"))
        if is_last_step and drain_replay and adapter_batch is not None:
            metrics.update(self._update_group(self.mrlx_adapter_group, adapter_batch, phase="drain"))
            metrics[f"{namespace}/adapter/drained_samples"] = _batch_len(adapter_batch)
            self._mrlx_adapter_updated_samples = getattr(self, "_mrlx_adapter_updated_samples", 0) + _batch_len(
                adapter_batch
            )
            metrics[f"{namespace}/adapter/replay_pending"] = 0
            if self.mrlx_adapter_group not in updated_groups:
                updated_groups.append(self.mrlx_adapter_group)
        elif adapter_batch is not None:
            self._mrlx_adapter_replay = self._stage_adapter_replay(adapter_batch)
            metrics[f"{namespace}/adapter/drained_samples"] = 0
            metrics[f"{namespace}/adapter/replay_pending"] = _batch_len(adapter_batch)
        else:
            metrics[f"{namespace}/adapter/drained_samples"] = 0
            metrics[f"{namespace}/adapter/replay_pending"] = 0

        metrics[f"{namespace}/adapter/collected_samples_total"] = self._mrlx_adapter_collected_samples
        metrics[f"{namespace}/adapter/updated_samples_total"] = getattr(self, "_mrlx_adapter_updated_samples", 0)
        if is_last_step and getattr(self, "_mrlx_adapter_updated_samples", 0) == 0:
            raise RuntimeError(
                "MrlX training reached its final step without an Adapter update; "
                "the run did not exercise two-policy co-training."
            )

        metrics[f"{namespace}/actor_groups/updated"] = len(updated_groups)
        metrics[f"{namespace}/actor_groups/total"] = len(self.actor_rollout_wgs)
        metrics[f"{namespace}/actor_groups/missing_trainable"] = 0
        return batch

    def _mrlx_final_training_step(self) -> int:
        """Return the last step reachable under both trainer stop conditions."""

        configured_steps = int(getattr(self, "total_training_steps", self.global_steps))
        train_dataloader = getattr(self, "train_dataloader", None)
        total_epochs = OmegaConf.select(self.config, "trainer.total_epochs")
        if train_dataloader is None or total_epochs is None:
            return configured_steps
        epoch_limited_steps = len(train_dataloader) * int(total_epochs)
        return min(configured_steps, epoch_limited_steps)

    def _prepare_actor_update(self, batch: Any) -> None:
        actor_config = self.config.actor_rollout_ref.actor
        batch.extra_info.update(
            {
                "calculate_entropy": actor_config.calculate_entropy or actor_config.entropy_coeff != 0.0,
                "distillation_use_topk": False,
                "global_batch_size": actor_config.ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n,
                "mini_batch_size": actor_config.ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n,
                "epochs": actor_config.ppo_epochs,
                "seed": actor_config.data_loader_seed,
                "dataloader_kwargs": {"shuffle": actor_config.shuffle},
                "temperature": self.config.actor_rollout_ref.rollout.temperature,
            }
        )

    def _update_group(self, group_id: str, route: Any, *, phase: str) -> dict[str, Any]:
        route.extra_info.update(
            {
                "calculate_entropy": self.config.actor_rollout_ref.actor.calculate_entropy
                or self.config.actor_rollout_ref.actor.entropy_coeff != 0.0,
                "distillation_use_topk": False,
                "global_batch_size": _batch_len(route),
                "mini_batch_size": None,
                "num_mini_batch": 1,
                "epochs": self.config.actor_rollout_ref.actor.ppo_epochs,
                "seed": self.config.actor_rollout_ref.actor.data_loader_seed,
                "dataloader_kwargs": {"shuffle": self.config.actor_rollout_ref.actor.shuffle},
                "temperature": self.config.actor_rollout_ref.rollout.temperature,
                "mrlx_update_phase": phase,
            }
        )
        output = self._actor_wg(group_id).update_actor(route)
        self._record_actor_update(group_id)
        prefixed = rename_dict(output["metrics"], f"actor/{group_id}/{phase}/")
        mfu_key = f"actor/{group_id}/{phase}/mfu"
        if mfu_key in prefixed:
            prefixed[f"perf/mfu/actor/{group_id}/{phase}"] = prefixed.pop(mfu_key)
        reduced = reduce_metrics(prefixed)
        reduced[
            f"trajweave/{self._metric_namespace()}/actor_groups/{_safe_metric_name(group_id)}/updated"
        ] = 1
        return reduced

    def _stage_adapter_replay(self, batch: Any) -> Any:
        if self._mrlx_adapter_replay is not None:
            raise RuntimeError("MrlX adapter replay already contains an unconsumed batch.")
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id)
        replay_keys = [f"step-{self.global_steps}:{index}:{key}" for index, key in enumerate(batch.keys)]
        replay_tags = [
            {
                **tag,
                "mrlx_replay": True,
                "mrlx_collected_step": self.global_steps,
                "mrlx_policy_lag": 1,
            }
            for tag in batch.tags
        ]
        replay = tq.kv_batch_put(
            keys=replay_keys,
            partition_id=self._mrlx_replay_partition,
            fields=data,
            tags=replay_tags,
        )
        replay.extra_info = {
            **(batch.extra_info or {}),
            "mrlx_collected_step": self.global_steps,
            "mrlx_policy_lag": 1,
        }
        return replay

    def on_train_end(self) -> None:
        replay = self._mrlx_adapter_replay
        if replay is not None:
            tq.kv_clear(keys=replay.keys, partition_id=replay.partition_id)
            self._mrlx_adapter_replay = None
            raise RuntimeError("MrlX training ended with an unconsumed Adapter replay batch.")
        if getattr(self, "_mrlx_adapter_updated_samples", 0) == 0:
            raise RuntimeError(
                "MrlX training ended without an Adapter update; the run did not exercise two-policy co-training."
            )
