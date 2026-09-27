from __future__ import annotations

from typing import Any

import torch
import transfer_queue as tq
from tensordict import TensorDict

from trajweave.backends.verl.extensions.comlrl.actor_critic import (
    ACTOR_CRITIC_TQ_FIELDS,
    prepare_actor_critic_batch,
)
from trajweave.backends.verl.routing import safe_critic_role_key
from trajweave.backends.verl.trainers.multi_actor_critic_sync import (
    TrajWeaveMultiActorCriticSyncTrainer,
    _actor_critic_settings,
    _unpack_advantage_tq_field,
    _validate_single_gpu_critic_routes,
)
from trajweave.backends.verl.trainers.multi_actor_sync import TrajWeaveMultiActorSyncTrainer
from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync


@register_trainer("trajweave_marft_shared_critic_sync")
class MARFTSharedCriticSyncTrainer(PPOTrainerSync):
    """Standard shared-actor MARFT trainer with value-model-aware critic LoRA."""

    def _critic_training_worker_cls(self) -> type[Any]:
        from trajweave.backends.verl.workers.value_lora import MARFTValueCriticTrainingWorker

        return MARFTValueCriticTrainingWorker


@register_trainer("trajweave_marft_multi_actor_sync")
class MARFTMultiActorSyncTrainer(TrajWeaveMultiActorSyncTrainer):
    """Role-routed MARFT actors with one shared value-model-aware critic."""

    def _critic_training_worker_cls(self) -> type[Any]:
        from trajweave.backends.verl.workers.value_lora import MARFTValueCriticTrainingWorker

        return MARFTValueCriticTrainingWorker


@register_trainer("trajweave_marft_independent_critic_sync")
class MARFTIndependentCriticSyncTrainer(TrajWeaveMultiActorCriticSyncTrainer):
    """Role-routed MARFT critics trained against projected workflow returns."""

    def _critic_training_worker_cls(self) -> type[Any]:
        from trajweave.backends.verl.workers.value_lora import MARFTValueCriticTrainingWorker

        return MARFTValueCriticTrainingWorker

    def _validate_multi_actor_specs(self) -> None:
        TrajWeaveMultiActorSyncTrainer._validate_multi_actor_specs(self)
        if not self.use_critic:
            raise ValueError("MARFT independent-critic trainer requires critic.enable=true.")
        if self.critic_topology != "independent":
            raise ValueError("MARFT independent-critic trainer only supports topology=independent.")
        loss_mode = str(self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla"))
        if loss_mode != "vanilla":
            raise ValueError("MARFT uses clipped PPO and requires actor.policy_loss.loss_mode=vanilla.")
        _validate_single_gpu_critic_routes(self.critic_route_specs.values())
        role_keys = [safe_critic_role_key(group_id) for group_id in self.critic_route_specs]
        if len(role_keys) != len(set(role_keys)):
            raise ValueError(f"MARFT critic group ids produce colliding safe role keys: {role_keys}.")
        requested = sum(route.gpus for route in self.critic_route_specs.values())
        requested += sum(
            self.multi_actor_worker_group_specs[group_id].gpus for group_id in self.multi_actor_trainable_group_ids
        )
        available = int(self.config.trainer.n_gpus_per_node)
        if requested > available:
            raise ValueError(
                "MARFT actor and independent critic groups request more GPUs than "
                f"trainer.n_gpus_per_node: requested={requested}, available={available}."
            )

    def _compute_advantage(self, batch: Any, metrics: dict[str, Any]) -> Any:
        del metrics
        fields = (*ACTOR_CRITIC_TQ_FIELDS, "response_mask", "old_values", "marft_projected_return")
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)
        mapping = {key: _unpack_advantage_tq_field(data[key]) for key in fields if key in data}
        projected = _projected_return_tensor(
            mapping["marft_projected_return"],
            row_count=len(batch.keys),
            like=mapping["old_values"],
        )
        mapping["joint_reward"] = projected
        mapping["joint_done"] = torch.ones_like(projected, dtype=torch.bool)
        mapping["joint_truncated"] = torch.zeros_like(projected, dtype=torch.bool)
        settings = _actor_critic_settings(self.config)
        prepared = prepare_actor_critic_batch(
            mapping,
            topology="independent",
            gamma=float(settings.get("gamma", self.config.algorithm.gamma)),
            normalize_by_population_std=bool(settings.get("normalize_advantages", True)),
        )
        from verl.workers.utils.padding import response_to_nested

        output = TensorDict(
            {
                "advantages": response_to_nested(prepared.actor_advantages, data["response_mask"]),
                "returns": response_to_nested(prepared.actor_returns, data["response_mask"]),
                "critic_returns": prepared.scalar_targets,
            },
            batch_size=len(batch.keys),
        )
        tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=output)
        return batch

    def _update_critic(self, batch: Any, metrics: dict[str, Any]) -> Any:
        result = super()._update_critic(batch, metrics)
        if "trajweave/comlrl/critic_groups/updated" in metrics:
            metrics["trajweave/marft/critic_groups/updated"] = metrics.pop("trajweave/comlrl/critic_groups/updated")
        return result


def _projected_return_tensor(value: Any, *, row_count: int, like: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=like.device, dtype=like.dtype).reshape(-1)
    else:
        if hasattr(value, "tolist"):
            value = value.tolist()
        values = list(value) if isinstance(value, list | tuple) else [value]
        tensor = torch.tensor(values, device=like.device, dtype=like.dtype).reshape(-1)
    if tensor.numel() != row_count:
        raise ValueError(f"MARFT projected returns contain {tensor.numel()} rows, expected {row_count}.")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("MARFT projected returns must be finite.")
    return tensor


__all__ = [
    "MARFTIndependentCriticSyncTrainer",
    "MARFTMultiActorSyncTrainer",
    "MARFTSharedCriticSyncTrainer",
]
