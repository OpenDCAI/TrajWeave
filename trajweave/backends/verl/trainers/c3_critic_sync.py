from __future__ import annotations

from functools import partial
from math import isfinite, log
from typing import Any

import torch
import torch.nn.functional as F
import transfer_queue as tq
from tensordict import TensorDict

from trajweave.backends.verl.extensions.c3 import C3_TQ_FIELDS
from trajweave.backends.verl.routing import split_tq_batch_by_field
from trajweave.backends.verl.schema import to_python
from trajweave.backends.verl.trainers.multi_actor_critic_sync import (
    CriticRoutedBatch,
    TrajWeaveMultiActorCriticSyncTrainer,
    _actor_critic_settings,
    _batch_len,
    _materialize_critic_tq_batch,
    _read_critic_scalar_values,
    _scalar_values_to_response,
    _select_real_critic_rows,
    _tq_string_values,
    _validate_single_gpu_critic_routes,
    _validated_or_tokenized_critic_fields,
)
from trajweave.backends.verl.trainers.multi_actor_sync import TrajWeaveMultiActorSyncTrainer
from trajweave.credit.c3 import compute_c3_scalar_credit
from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.workers.utils.padding import response_to_nested


@register_trainer("trajweave_c3_critic_sync")
class TrajWeaveC3CriticSyncTrainer(TrajWeaveMultiActorCriticSyncTrainer):
    """Multi-actor C3 trainer with one centralized prefix-Q critic."""

    def _validate_multi_actor_specs(self) -> None:
        TrajWeaveMultiActorSyncTrainer._validate_multi_actor_specs(self)
        if not self.use_critic:
            raise ValueError("C3 value-assisted/value-only training requires critic.enable=true.")
        loss_mode = str(self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla"))
        if loss_mode != "vanilla_no_dual_clip":
            raise ValueError("C3 requires actor.policy_loss.loss_mode=vanilla_no_dual_clip.")
        if self.critic_topology != "centralized" or self.critic_type != "q":
            raise ValueError("C3 requires one centralized Q critic route.")
        if len(self.critic_route_specs) != 1:
            raise ValueError("C3 requires exactly one centralized Q critic group.")
        [route] = self.critic_route_specs.values()
        if set(route.actor_groups) != set(self.multi_actor_trainable_group_ids):
            raise ValueError("C3 Q critic must cover all trainable role policy groups.")
        _validate_single_gpu_critic_routes(self.critic_route_specs.values())
        requested = route.gpus + sum(
            self.multi_actor_worker_group_specs[group_id].gpus for group_id in self.multi_actor_trainable_group_ids
        )
        available = int(self.config.trainer.n_gpus_per_node)
        if requested > available:
            raise ValueError(
                "C3 Actor and prefix-Q critic groups request more GPUs than trainer.n_gpus_per_node: "
                f"requested={requested}, available={available}."
            )

    def _create_worker_groups(self) -> None:
        super()._create_worker_groups()
        coefficient = _validated_loss_coefficient(_actor_critic_settings(self.config).get("value_loss_coef", 1.0))
        for critic_group, worker_group in self.critic_wgs.items():
            worker_group.set_loss_fn(
                partial(
                    verl_c3_weighted_bce_critic_loss,
                    self._critic_cfgs[critic_group],
                    value_loss_coef=coefficient,
                )
            )

    def _ensure_critic_fields(self, batch: Any) -> Any:
        fields = (
            "worker_group",
            "agent_id",
            "traj_uid",
            "turn_id",
            "response_mask",
            "critic_group",
            "critic_type",
            "critic_input_ids",
            "critic_attention_mask",
            "critic_position_ids",
            "critic_loss_mask",
            *C3_TQ_FIELDS,
        )
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)
        row_count = len(batch.keys)
        rows = [
            {field: _row_value(data[field], row, row_count=row_count) for field in fields} for row in range(row_count)
        ]
        active = [bool(torch.as_tensor(row["response_mask"]).bool().any()) for row in rows]
        [route] = self.critic_route_specs.values()
        output_rows = []
        seen_nodes: set[tuple[str, str]] = set()
        for row_index, (row, is_active) in enumerate(zip(rows, active, strict=True)):
            if not is_active:
                output_rows.append(
                    {
                        "critic_group": row["critic_group"],
                        "critic_type": row["critic_type"],
                        "critic_input_ids": [],
                        "critic_attention_mask": [],
                        "critic_position_ids": [],
                        "critic_loss_mask": 0.0,
                    }
                )
                continue
            node_key = (str(row["traj_uid"]), str(row["node_id"]))
            if node_key in seen_nodes:
                raise ValueError(f"Duplicate C3 prefix critic row: {node_key!r}.")
            seen_nodes.add(node_key)
            prefix_text = str(row["c3_prefix_text"])
            if not prefix_text.strip():
                raise ValueError(f"Active C3 row {batch.keys[row_index]!r} is missing c3_prefix_text.")
            group_id = str(row["c3_group_id"])
            if not group_id or group_id.startswith(("__missing__", "__padding__")):
                raise ValueError(f"Active C3 row {batch.keys[row_index]!r} is missing a real c3_group_id.")
            supplied_type = str(row["critic_type"])
            if not supplied_type.startswith(("__missing__", "__padding__")) and supplied_type != "q":
                raise ValueError("C3 critic rows must use critic_type='q'.")
            token_fields = _validated_or_tokenized_critic_fields(
                row,
                text=prefix_text,
                tokenizer=self._critic_tokenizer(route),
                route=route,
                batch_key=batch.keys[row_index],
            )
            output_rows.append(
                {
                    "critic_group": route.critic_group,
                    "critic_type": "q",
                    "critic_loss_mask": 1.0,
                    **token_fields,
                }
            )
        from verl.utils.tensordict_utils import list_of_dict_to_tensordict

        tq.kv_batch_put(
            keys=batch.keys,
            partition_id=batch.partition_id,
            fields=list_of_dict_to_tensordict(output_rows),
        )
        return batch

    def _route_critic_batch(self, batch: Any) -> list[CriticRoutedBatch]:
        self._ensure_critic_fields(batch)
        real_batch = _select_real_critic_rows(batch)
        if real_batch is None:
            return []
        routed = split_tq_batch_by_field(real_batch, field="critic_group")
        if len(routed) != 1:
            raise ValueError("C3 requires all prefix rows to route to one Q critic group.")
        item = routed[0]
        if item.group_id not in self.critic_wgs:
            raise KeyError(f"Unknown C3 critic group {item.group_id!r}.")
        if set(_tq_string_values(item.batch, "critic_type")) != {"q"}:
            raise ValueError("C3 routed critic rows must all use critic_type='q'.")
        world_size = int(self.critic_wgs[item.group_id].world_size)
        if _batch_len(item.batch) % world_size != 0:
            raise ValueError("C3 critic rows must be divisible by critic worker world size.")
        return [CriticRoutedBatch(item.group_id, item.batch)]

    def _compute_values(self, batch: Any, metrics: dict) -> Any:
        del metrics
        values_by_key: dict[str, float] = {}
        for routed in self._route_critic_batch(batch):
            critic_batch = _materialize_critic_tq_batch(routed.batch, include_targets=False)
            critic_batch.extra_info["compute_loss"] = False
            try:
                result = self.critic_wgs[routed.critic_group].infer_batch(critic_batch)
                if hasattr(result, "futures"):
                    __import__("ray").get(result.futures)
                values = _c3_q_probabilities(_read_critic_scalar_values(critic_batch))
                values_by_key.update(dict(zip(routed.batch.keys, values, strict=True)))
            finally:
                tq.kv_clear(keys=critic_batch.keys, partition_id=critic_batch.partition_id)
        response_data = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=["response_mask"],
        )
        response_mask = response_data["response_mask"]
        scalar_values = torch.tensor(
            [values_by_key.get(key, 0.0) for key in batch.keys],
            dtype=torch.float32,
        )
        token_values = _scalar_values_to_response(scalar_values, response_mask)
        tq.kv_batch_put(
            keys=batch.keys,
            partition_id=batch.partition_id,
            fields=TensorDict(
                {"old_values": scalar_values, "values": token_values},
                batch_size=len(batch.keys),
            ),
        )
        return batch

    def _compute_advantage(self, batch: Any, metrics: dict) -> Any:
        fields = (*C3_TQ_FIELDS, "response_mask", "old_values")
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)
        from trajweave.backends.verl.trainers.multi_actor_critic_sync import _unpack_advantage_tq_field

        nested_response_mask = data["response_mask"]
        response_mask = _unpack_advantage_tq_field(nested_response_mask)
        row_count = len(batch.keys)
        active = response_mask.bool().any(dim=-1).detach().cpu().tolist()
        real_rows = [row for row, value in enumerate(active) if value]
        if not real_rows:
            raise ValueError("C3 Q-critic batch has no non-padding prefix rows.")
        groups = [_row_value(data["c3_group_id"], row, row_count=row_count) for row in range(row_count)]
        rewards = [float(_row_value(data["c3_subtree_return"], row, row_count=row_count)) for row in range(row_count)]
        leaf_counts = [float(_row_value(data["c3_leaf_count"], row, row_count=row_count)) for row in range(row_count)]
        _validate_targets_and_weights(
            torch.tensor([rewards[row] for row in real_rows], dtype=torch.float64),
            torch.tensor([leaf_counts[row] for row in real_rows], dtype=torch.float64),
        )
        old_values = torch.as_tensor(_unpack_advantage_tq_field(data["old_values"]), dtype=torch.float32).view(-1)
        settings = _c3_settings(self.config)
        result = compute_c3_scalar_credit(
            subtree_returns=[rewards[row] for row in real_rows],
            group_ids=[groups[row] for row in real_rows],
            q_values=[float(old_values[row]) for row in real_rows],
            variant=str(settings.get("credit_variant", "value_assisted")),
            baseline_mode=str(settings.get("baseline_mode", "loo")),
            value_assisted_alpha=float(settings.get("value_assisted_alpha", 1.0)),
            normalize=bool(settings.get("normalize_advantages", True)),
        )
        advantages = torch.zeros(row_count, dtype=torch.float32)
        returns = torch.zeros(row_count, dtype=torch.float32)
        targets = torch.zeros(row_count, dtype=torch.float32)
        for offset, row in enumerate(real_rows):
            advantages[row] = result.advantages[offset]
            returns[row] = rewards[row]
            targets[row] = rewards[row]
        mask = response_mask.to(dtype=torch.float32)
        tq.kv_batch_put(
            keys=batch.keys,
            partition_id=batch.partition_id,
            fields=TensorDict(
                {
                    "advantages": response_to_nested(advantages.unsqueeze(-1) * mask, nested_response_mask),
                    "returns": response_to_nested(returns.unsqueeze(-1) * mask, nested_response_mask),
                    "critic_returns": targets,
                },
                batch_size=row_count,
            ),
        )
        metrics["trajweave/c3/prefix_rows"] = len(real_rows)
        metrics["trajweave/c3/sibling_groups"] = len({str(groups[row]) for row in real_rows})
        return batch

    def _update_critic(self, batch: Any, metrics: dict) -> Any:
        updated = 0
        for routed in self._route_critic_batch(batch):
            critic_batch = _materialize_c3_critic_tq_batch(routed.batch)
            critic_cfg = self._critic_cfgs[routed.critic_group]
            critic_batch.extra_info.update(
                {
                    "global_batch_size": len(critic_batch.keys),
                    "mini_batch_size": None,
                    "num_mini_batch": 1,
                    "epochs": critic_cfg.ppo_epochs,
                    "seed": critic_cfg.data_loader_seed,
                    "dataloader_kwargs": {"shuffle": critic_cfg.shuffle},
                }
            )
            try:
                output = self.critic_wgs[routed.critic_group].train_mini_batch(critic_batch).get()
                renamed = rename_dict(output["metrics"], f"critic/{routed.critic_group}/")
                metrics.update(reduce_metrics(renamed))
                updated += 1
            finally:
                tq.kv_clear(keys=critic_batch.keys, partition_id=critic_batch.partition_id)
        if updated == 0:
            raise RuntimeError("C3 critic update did not update its prefix-Q critic worker group.")
        metrics["trajweave/c3/critic_groups/updated"] = updated
        return batch


def verl_c3_weighted_bce_critic_loss(
    config: Any,
    model_output: dict[str, Any],
    data: Any,
    dp_group=None,
    *,
    value_loss_coef: float = 1.0,
):
    """Leaf-count-weighted BCE on raw prefix-Q logits."""

    del config, dp_group
    logits_all = _aligned_loss_tensor(model_output["values"], field="model_output['values']")
    targets_all = _aligned_loss_tensor(data["returns"], field="data['returns']").to(
        dtype=torch.float32, device=logits_all.device
    )
    mask = _aligned_loss_tensor(data["loss_mask"], field="data['loss_mask']").to(
        dtype=torch.bool, device=logits_all.device
    )
    weights_all = _aligned_loss_tensor(data["c3_leaf_weights"], field="data['c3_leaf_weights']").to(
        dtype=torch.float32, device=logits_all.device
    )
    if logits_all.shape != targets_all.shape or logits_all.shape != mask.shape or logits_all.shape != weights_all.shape:
        raise ValueError("C3 critic logits, targets, loss_mask, and leaf weights must have identical token positions.")
    if not bool(mask.any()):
        raise ValueError("C3 critic loss requires at least one active prefix target.")
    if not bool(torch.isfinite(logits_all[mask]).all()):
        raise FloatingPointError("C3 critic logits must be finite at every active prefix target.")
    if not bool(torch.isfinite(targets_all[mask]).all()):
        raise FloatingPointError("C3 critic targets must be finite at every active prefix target.")
    if not bool(((targets_all[mask] >= 0.0) & (targets_all[mask] <= 1.0)).all()):
        raise ValueError("C3 critic targets must be in [0, 1].")
    if not bool(torch.isfinite(weights_all).all()):
        raise FloatingPointError("C3 critic leaf weights must be finite.")
    if not bool((weights_all[mask] > 0.0).all()):
        raise ValueError("C3 critic leaf weights must be positive at every active prefix target.")
    if not bool((weights_all[~mask] == 0.0).all()):
        raise ValueError("C3 critic leaf weights must be zero outside the active loss positions.")

    from verl.utils import tensordict_utils as tu

    total_weight = _validated_positive_finite(
        tu.get_non_tensor_data(data, "c3_total_leaf_weight", None),
        field="c3_total_leaf_weight",
    )
    positive_weight = _validated_finite(
        tu.get_non_tensor_data(data, "c3_positive_leaf_weight", None),
        field="c3_positive_leaf_weight",
    )
    if not 0.0 <= positive_weight <= total_weight:
        raise ValueError("c3_positive_leaf_weight must be in [0, c3_total_leaf_weight].")
    coefficient = _validated_loss_coefficient(value_loss_coef)
    prior, logit_bias = _laplace_smoothed_logit_bias(positive_weight, total_weight)
    dp_size = _validated_positive_finite(tu.get_non_tensor_data(data, "dp_size", 1), field="dp_size")

    logits = logits_all[mask].float()
    targets = targets_all[mask].float()
    weights = weights_all[mask].float()
    elementwise = F.binary_cross_entropy_with_logits(logits + logit_bias, targets, reduction="none")
    bce_loss = (elementwise * weights).sum() * (dp_size / total_weight)
    loss = bce_loss * coefficient
    probabilities = torch.sigmoid(logits + logit_bias).detach()
    return loss, {
        "critic/bce_loss": float(bce_loss.detach()),
        "critic/value_loss": float(loss.detach()),
        "critic/value_loss_coef": coefficient,
        "critic/q_prob_mean": float(probabilities.mean()),
        "critic/pos_prior": prior,
        "critic/logit_bias": logit_bias,
    }


def _materialize_c3_critic_tq_batch(batch: Any):
    source = tq.kv_batch_get(
        keys=batch.keys,
        partition_id=batch.partition_id,
        select_fields=["critic_returns", "c3_leaf_count"],
    )
    row_count = len(batch.keys)
    targets = torch.tensor(
        [float(_row_value(source["critic_returns"], row, row_count=row_count)) for row in range(row_count)],
        dtype=torch.float64,
    )
    weights = torch.tensor(
        [float(_row_value(source["c3_leaf_count"], row, row_count=row_count)) for row in range(row_count)],
        dtype=torch.float64,
    )
    _validate_targets_and_weights(targets, weights)
    critic_batch = _materialize_critic_tq_batch(batch, include_targets=True)
    try:
        materialized = tq.kv_batch_get(
            keys=critic_batch.keys,
            partition_id=critic_batch.partition_id,
            select_fields=["loss_mask"],
        )
        rows = []
        for row, weight in enumerate(weights.tolist()):
            loss_mask = torch.as_tensor(to_python(materialized["loss_mask"][row]), dtype=torch.bool)
            rows.append({"c3_leaf_weights": loss_mask.to(dtype=torch.float32) * float(weight)})
        from verl.utils.tensordict_utils import list_of_dict_to_tensordict

        tq.kv_batch_put(
            keys=critic_batch.keys,
            partition_id=critic_batch.partition_id,
            fields=list_of_dict_to_tensordict(rows),
        )
        total_weight = float(weights.sum().item())
        positive_weight = float((targets * weights).sum().item())
        prior, logit_bias = _laplace_smoothed_logit_bias(positive_weight, total_weight)
        critic_batch.extra_info.update(
            {
                "c3_total_leaf_weight": total_weight,
                "c3_positive_leaf_weight": positive_weight,
                "c3_pos_prior": prior,
                "c3_logit_bias": logit_bias,
            }
        )
        return critic_batch
    except Exception:
        tq.kv_clear(keys=critic_batch.keys, partition_id=critic_batch.partition_id)
        raise


def _c3_q_probabilities(raw_logits: list[float]) -> list[float]:
    logits = torch.as_tensor(raw_logits, dtype=torch.float32)
    if logits.ndim != 1 or not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("C3 Q-critic inference must return one finite raw logit per prefix.")
    return torch.sigmoid(logits).tolist()


def _validate_targets_and_weights(targets: torch.Tensor, weights: torch.Tensor) -> None:
    if targets.ndim != 1 or weights.shape != targets.shape or targets.numel() == 0:
        raise ValueError("C3 critic targets and leaf weights must be aligned non-empty vectors.")
    if not bool(torch.isfinite(targets).all()):
        raise FloatingPointError("C3 critic targets must be finite.")
    if not bool(((targets >= 0.0) & (targets <= 1.0)).all()):
        raise ValueError("C3 critic targets must be in [0, 1].")
    if not bool(torch.isfinite(weights).all()):
        raise FloatingPointError("C3 critic leaf weights must be finite.")
    if not bool((weights > 0.0).all()):
        raise ValueError("C3 critic leaf weights must be positive.")


def _laplace_smoothed_logit_bias(positive_weight: float, total_weight: float) -> tuple[float, float]:
    positive = _validated_finite(positive_weight, field="positive_weight")
    total = _validated_positive_finite(total_weight, field="total_weight")
    if not 0.0 <= positive <= total:
        raise ValueError("positive_weight must be in [0, total_weight].")
    prior = (positive + 0.5) / (total + 1.0)
    prior = min(max(prior, 1.0e-6), 1.0 - 1.0e-6)
    return prior, log(prior / (1.0 - prior))


def _aligned_loss_tensor(value: Any, *, field: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.values() if value.is_nested else value
    values_method = getattr(value, "values", None)
    if callable(values_method):
        flattened = values_method()
        if isinstance(flattened, torch.Tensor):
            return flattened
    try:
        return torch.as_tensor(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be a tensor or nested tensor with .values().") from exc


def _validated_finite(value: Any, *, field: str) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be finite.") from exc
    if not isfinite(normalized):
        raise ValueError(f"{field} must be finite.")
    return normalized


def _validated_positive_finite(value: Any, *, field: str) -> float:
    normalized = _validated_finite(value, field=field)
    if normalized <= 0.0:
        raise ValueError(f"{field} must be positive.")
    return normalized


def _validated_loss_coefficient(value: Any) -> float:
    coefficient = _validated_finite(value, field="C3 value_loss_coef")
    if coefficient <= 0.0:
        raise ValueError("C3 value_loss_coef must be positive.")
    return coefficient


def _row_value(value: Any, row: int, *, row_count: int) -> Any:
    if isinstance(value, torch.Tensor):
        return to_python(value[row])
    unpacked = to_python(value)
    if isinstance(unpacked, list | tuple) and len(unpacked) == row_count:
        return unpacked[row]
    return unpacked


def _c3_settings(config: Any) -> dict[str, Any]:
    value = to_python(config)
    if not isinstance(value, dict):
        raise TypeError("C3 trainer config must be a mapping.")
    trajweave = value.get("trajweave", {}) or {}
    c3 = trajweave.get("c3", {}) if isinstance(trajweave, dict) else {}
    if not isinstance(c3, dict):
        raise TypeError("trajweave.c3 must be a mapping.")
    return c3


__all__ = ["TrajWeaveC3CriticSyncTrainer", "verl_c3_weighted_bce_critic_loss"]
