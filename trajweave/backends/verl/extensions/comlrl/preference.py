from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any

import torch

from trajweave.backends.verl.schema import batch_item, to_python
from trajweave.credit.comlrl.madpo import (
    madpo_actor_loss as reference_free_madpo_actor_loss,
)
from trajweave.credit.comlrl.madpo import sequence_logprob_sum

PREFERENCE_TQ_FIELDS = (
    "preference_pair_id",
    "preference_side",
    "chosen_reward",
    "rejected_reward",
    "preference_loss_mask",
    "worker_group",
    "response_mask",
)
PREFERENCE_LOGPROB_TQ_FIELDS = (*PREFERENCE_TQ_FIELDS, "log_probs")


@dataclass(frozen=True)
class PreferencePairRows:
    preference_pair_id: str
    worker_group: str
    chosen_row: int
    rejected_row: int
    chosen_reward: float
    rejected_reward: float


@dataclass(frozen=True)
class MaterializedPreferenceBatch:
    response_log_probs: torch.Tensor
    response_mask: torch.Tensor
    sequence_logps: torch.Tensor
    pairs: tuple[PreferencePairRows, ...]
    deltas: dict[tuple[str, str], torch.Tensor]


def sequence_logprob_sums(log_probs: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    return sequence_logprob_sum(log_probs, response_mask)


def validate_preference_pairs(
    fields: Mapping[str, Any],
    *,
    expected_worker_groups: Sequence[str] | None = None,
) -> tuple[PreferencePairRows, ...]:
    """Validate active TQ rows and pair them by ``(pair id, worker group)``."""

    missing = [field for field in PREFERENCE_TQ_FIELDS if field not in fields]
    if missing:
        raise KeyError(f"MADPO TransferQueue rows are missing fields: {missing}")
    row_count = _row_count(fields["preference_pair_id"])
    for field in PREFERENCE_TQ_FIELDS:
        if _row_count(fields[field]) != row_count:
            raise ValueError(f"MADPO field {field!r} does not align with preference_pair_id")

    expected = tuple(str(group) for group in expected_worker_groups or ())
    if len(expected) != len(set(expected)) or any(not group for group in expected):
        raise ValueError("expected MADPO worker groups must be unique and non-empty")
    expected_set = set(expected)
    grouped: dict[tuple[str, str], dict[str, int]] = {}
    rewards: dict[tuple[str, str], tuple[float, float]] = {}
    groups_by_pair: dict[str, set[str]] = {}

    for row in range(row_count):
        active_value = float(to_python(batch_item(fields["preference_loss_mask"], row)))
        if not isfinite(active_value):
            raise ValueError("preference_loss_mask must be finite")
        if active_value == 0:
            continue
        if active_value != 1:
            raise ValueError("active preference_loss_mask values must equal 1")

        pair_id = str(to_python(batch_item(fields["preference_pair_id"], row)))
        worker_group = str(to_python(batch_item(fields["worker_group"], row)))
        side = str(to_python(batch_item(fields["preference_side"], row))).strip().lower()
        if not pair_id or not worker_group:
            raise ValueError("active MADPO rows require non-empty pair and worker-group identifiers")
        if side not in {"chosen", "rejected"}:
            raise ValueError(f"MADPO row {row} has invalid preference_side={side!r}")
        if expected_set and worker_group not in expected_set:
            raise ValueError(f"MADPO row references non-trainable worker group {worker_group!r}")

        chosen_reward = float(to_python(batch_item(fields["chosen_reward"], row)))
        rejected_reward = float(to_python(batch_item(fields["rejected_reward"], row)))
        if not isfinite(chosen_reward) or not isfinite(rejected_reward):
            raise ValueError("MADPO preference rewards must be finite")
        if chosen_reward <= rejected_reward:
            raise ValueError("MADPO chosen_reward must be strictly greater than rejected_reward")

        key = (pair_id, worker_group)
        by_side = grouped.setdefault(key, {})
        if side in by_side:
            raise ValueError(f"MADPO pair/group {key!r} has duplicate {side!r} rows")
        by_side[side] = row
        previous_rewards = rewards.setdefault(key, (chosen_reward, rejected_reward))
        if previous_rewards != (chosen_reward, rejected_reward):
            raise ValueError(f"MADPO pair/group {key!r} has inconsistent reward metadata")
        groups_by_pair.setdefault(pair_id, set()).add(worker_group)

    if not grouped:
        raise ValueError("MADPO batch contains no active preference rows")

    output: list[PreferencePairRows] = []
    for key in sorted(grouped):
        by_side = grouped[key]
        if set(by_side) != {"chosen", "rejected"}:
            raise ValueError(f"MADPO pair/group {key!r} must contain one chosen and one rejected row")
        chosen_reward, rejected_reward = rewards[key]
        output.append(
            PreferencePairRows(
                preference_pair_id=key[0],
                worker_group=key[1],
                chosen_row=by_side["chosen"],
                rejected_row=by_side["rejected"],
                chosen_reward=chosen_reward,
                rejected_reward=rejected_reward,
            )
        )

    if expected_set:
        for pair_id, actual_groups in groups_by_pair.items():
            if actual_groups != expected_set:
                missing_groups = sorted(expected_set - actual_groups)
                extra_groups = sorted(actual_groups - expected_set)
                raise ValueError(
                    f"MADPO pair {pair_id!r} must cover every trainable actor; "
                    f"missing={missing_groups}, extra={extra_groups}"
                )
    return tuple(output)


def materialize_pair_response_tensors(fields: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize TQ response masks and model log probabilities as aligned padded tensors."""

    if "log_probs" not in fields or "response_mask" not in fields:
        raise KeyError("MADPO materialization requires log_probs and response_mask")
    log_probs = fields["log_probs"]
    response_mask = fields["response_mask"]
    if not isinstance(log_probs, torch.Tensor) or not isinstance(response_mask, torch.Tensor):
        raise TypeError("MADPO log_probs and response_mask must be tensors")

    if getattr(log_probs, "is_nested", False):
        from trajweave.backends.verl.extensions.common.nested_compat import _nested_to_padded_tensor_compat

        if not getattr(response_mask, "is_nested", False):
            raise ValueError("nested MADPO log_probs require a nested response_mask")
        if torch.equal(log_probs.offsets().diff(), response_mask.offsets().diff()):
            log_probs = _nested_to_padded_tensor_compat(log_probs, padding=0)
        else:
            from verl.workers.utils.padding import response_from_nested

            response_log_probs = response_from_nested(log_probs, response_mask)
            log_probs = _nested_to_padded_tensor_compat(response_log_probs, padding=0)
    elif log_probs.ndim != 2:
        raise ValueError("MADPO response log_probs must be a nested tensor or a padded 2D tensor")

    if getattr(response_mask, "is_nested", False):
        from trajweave.backends.verl.extensions.common.nested_compat import _nested_to_padded_tensor_compat

        response_mask = _nested_to_padded_tensor_compat(response_mask, padding=0)
    if response_mask.ndim != 2 or response_mask.shape != log_probs.shape:
        raise ValueError("MADPO response masks and log probabilities must be aligned [batch, response]")
    return log_probs, response_mask.to(device=log_probs.device)


def materialize_pair_response_logprobs(
    fields: Mapping[str, Any],
    *,
    expected_worker_groups: Sequence[str] | None = None,
) -> MaterializedPreferenceBatch:
    pairs = validate_preference_pairs(fields, expected_worker_groups=expected_worker_groups)
    log_probs, response_mask = materialize_pair_response_tensors(fields)
    active_rows = sorted({row for pair in pairs for row in (pair.chosen_row, pair.rejected_row)})
    sequence_logps = log_probs.new_zeros(log_probs.shape[0])
    sequence_logps[active_rows] = sequence_logprob_sum(log_probs[active_rows], response_mask[active_rows])
    deltas = {
        (pair.preference_pair_id, pair.worker_group): (
            sequence_logps[pair.chosen_row] - sequence_logps[pair.rejected_row]
        )
        for pair in pairs
    }
    if any(not bool(torch.isfinite(delta)) for delta in deltas.values()):
        raise FloatingPointError("MADPO pair deltas must be finite")
    return MaterializedPreferenceBatch(log_probs, response_mask, sequence_logps, pairs, deltas)


def materialize_tq_preference_pairs(
    batch: Any,
    *,
    expected_worker_groups: Sequence[str] | None = None,
) -> MaterializedPreferenceBatch:
    import transfer_queue as tq

    fields = tq.kv_batch_get(
        keys=batch.keys,
        partition_id=batch.partition_id,
        select_fields=PREFERENCE_LOGPROB_TQ_FIELDS,
    )
    return materialize_pair_response_logprobs(fields, expected_worker_groups=expected_worker_groups)


def snapshot_pair_deltas(
    sequence_logps: torch.Tensor,
    *,
    pair_ids: Sequence[Any],
    sides: Sequence[str],
    worker_groups: Sequence[str],
    active_mask: Sequence[bool] | torch.Tensor | None = None,
) -> dict[tuple[str, str], torch.Tensor]:
    if sequence_logps.ndim != 1:
        raise ValueError("MADPO sequence_logps must contain one scalar per row")
    if not bool(torch.isfinite(sequence_logps).all()):
        raise FloatingPointError("MADPO sequence_logps must be finite")
    row_count = sequence_logps.numel()
    if any(len(values) != row_count for values in (pair_ids, sides, worker_groups)):
        raise ValueError("MADPO pair IDs, sides, and worker groups must align with sequence_logps")
    active = (
        torch.ones(row_count, dtype=torch.bool, device=sequence_logps.device)
        if active_mask is None
        else torch.as_tensor(active_mask, dtype=torch.bool, device=sequence_logps.device)
    )
    if active.shape != sequence_logps.shape:
        raise ValueError("MADPO active_mask must align with sequence_logps")
    grouped: dict[tuple[str, str], dict[str, int]] = {}
    for row, (pair_id, side, worker_group) in enumerate(zip(pair_ids, sides, worker_groups, strict=True)):
        if not bool(active[row]):
            continue
        normalized_side = str(side).strip().lower()
        if normalized_side not in {"chosen", "rejected"}:
            raise ValueError(f"MADPO row {row} has invalid preference_side={side!r}")
        key = (str(pair_id), str(worker_group))
        by_side = grouped.setdefault(key, {})
        if normalized_side in by_side:
            raise ValueError(f"MADPO pair/group {key!r} has duplicate {normalized_side!r} rows")
        by_side[normalized_side] = row
    output: dict[tuple[str, str], torch.Tensor] = {}
    for key, by_side in grouped.items():
        if set(by_side) != {"chosen", "rejected"}:
            raise ValueError(f"MADPO pair/group {key!r} must contain one chosen and one rejected row")
        output[key] = sequence_logps[by_side["chosen"]] - sequence_logps[by_side["rejected"]]
    return output


def detached_other_agent_deltas(
    deltas: Mapping[tuple[str, str], torch.Tensor],
) -> dict[tuple[str, str], torch.Tensor]:
    groups_by_pair: dict[str, list[str]] = {}
    for pair_id, worker_group in deltas:
        groups_by_pair.setdefault(pair_id, []).append(worker_group)
    output: dict[tuple[str, str], torch.Tensor] = {}
    for pair_id, worker_groups in groups_by_pair.items():
        for worker_group in worker_groups:
            others = [deltas[(pair_id, other)].detach() for other in worker_groups if other != worker_group]
            output[(pair_id, worker_group)] = sum(others, deltas[(pair_id, worker_group)].new_zeros(()))
    return output


def madpo_joint_pair_loss(
    current_deltas: torch.Tensor,
    other_agent_deltas: torch.Tensor,
    *,
    beta: float = 0.1,
    reduction: str = "mean",
) -> torch.Tensor:
    if current_deltas.ndim != 1 or other_agent_deltas.shape != current_deltas.shape:
        raise ValueError("MADPO current and detached-other deltas must be aligned 1D tensors")
    if not bool(torch.isfinite(current_deltas).all()) or not bool(torch.isfinite(other_agent_deltas).all()):
        raise FloatingPointError("MADPO current and other-agent deltas must be finite")
    snapshot = {
        "own": current_deltas.detach(),
        "others": other_agent_deltas.detach(),
    }
    return reference_free_madpo_actor_loss(
        current_deltas,
        snapshot,
        actor_id="own",
        beta=beta,
        reduction=reduction,
    )


def madpo_microbatch_loss(
    log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    pair_indices: torch.Tensor,
    preference_side: torch.Tensor,
    other_agent_delta: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    beta: float,
    global_pair_count: int,
    dp_size: int = 1,
) -> tuple[torch.Tensor, int]:
    """Return a globally normalized MADPO loss contribution for one engine microbatch."""

    if log_probs.ndim != 2 or response_mask.shape != log_probs.shape:
        raise ValueError("MADPO microbatch log_probs and response_mask must be aligned 2D tensors")
    row_count = log_probs.shape[0]
    for name, value in {
        "pair_indices": pair_indices,
        "preference_side": preference_side,
        "other_agent_delta": other_agent_delta,
        "active_mask": active_mask,
    }.items():
        if value.reshape(-1).numel() != row_count:
            raise ValueError(f"MADPO {name} must contain one value per response row")
    if isinstance(global_pair_count, bool) or int(global_pair_count) < 1:
        raise ValueError("MADPO global_pair_count must be a positive integer")
    if isinstance(dp_size, bool) or int(dp_size) < 1:
        raise ValueError("MADPO dp_size must be a positive integer")

    if not log_probs.is_floating_point():
        raise TypeError("MADPO microbatch log_probs must be floating point")
    response_mask = response_mask.to(dtype=torch.bool, device=log_probs.device)
    pair_indices = pair_indices.to(dtype=torch.long, device=log_probs.device).reshape(-1)
    preference_side = preference_side.to(dtype=log_probs.dtype, device=log_probs.device).reshape(-1)
    other_agent_delta = other_agent_delta.to(dtype=log_probs.dtype, device=log_probs.device).reshape(-1)
    active = active_mask.to(dtype=torch.bool, device=log_probs.device).reshape(-1)
    if not bool(active.any()):
        zero_loss = torch.where(response_mask, log_probs, torch.zeros_like(log_probs)).sum() * 0.0
        return zero_loss, 0

    active_token_log_probs = log_probs[active][response_mask[active]]
    if not bool(torch.isfinite(active_token_log_probs).all()):
        fallback = torch.tensor(0.1, dtype=log_probs.dtype, device=log_probs.device, requires_grad=True)
        return fallback, 0
    sequence_logps = log_probs.new_zeros(row_count)
    sequence_logps[active] = sequence_logprob_sum(log_probs[active], response_mask[active])
    current: list[torch.Tensor] = []
    detached_other: list[torch.Tensor] = []
    for pair_index in torch.unique(pair_indices[active], sorted=True):
        rows = torch.nonzero(active & pair_indices.eq(pair_index), as_tuple=False).flatten()
        if rows.numel() != 2 or set(preference_side[rows].tolist()) != {-1.0, 1.0}:
            raise ValueError("Each MADPO actor microbatch pair must contain one chosen and one rejected row")
        current.append((sequence_logps[rows] * preference_side[rows]).sum())
        pair_other = other_agent_delta[rows]
        if not bool(torch.isfinite(pair_other).all()):
            raise FloatingPointError("MADPO other-agent deltas must be finite")
        if not torch.allclose(pair_other, pair_other[0].expand_as(pair_other)):
            raise ValueError("MADPO other-agent delta must be identical on chosen/rejected rows")
        detached_other.append(pair_other[0])
    pair_count = len(current)
    if pair_count > int(global_pair_count):
        raise ValueError("MADPO microbatch pair count cannot exceed global_pair_count")
    loss_sum = madpo_joint_pair_loss(
        torch.stack(current),
        torch.stack(detached_other),
        beta=float(beta),
        reduction="sum",
    )
    return loss_sum * (int(dp_size) / int(global_pair_count)), pair_count


def madpo_actor_loss(config: Any, model_output: Mapping[str, Any], data: Any, dp_group=None):
    """VERL ``TrainingWorker.set_loss_fn`` callback for reference-free MADPO."""

    del dp_group
    from verl.utils import tensordict_utils as tu
    from verl.workers.utils.padding import no_padding_2_padding
    from trajweave.backends.verl.extensions.common.nested_compat import _tensordict_to_padded_tensor_compat

    raw_log_probs = model_output["log_probs"]
    if (not raw_log_probs.is_nested and raw_log_probs.ndim == 2
            and raw_log_probs.shape[0] == 1 and data["response_mask"].is_nested):
        # LZ 的无 remove-padding 兼容层已抽取响应并拼成一行，按原响应长度还原偏好对。
        lengths = data["response_mask"].offsets().diff().tolist()
        if sum(lengths) != raw_log_probs.numel():
            raise ValueError("MADPO flat response log probabilities do not match response lengths")
        pieces = raw_log_probs.reshape(-1).split(lengths)
        max_length = max(lengths)
        log_probs = torch.stack([torch.nn.functional.pad(piece, (0, max_length - len(piece)))
                                 for piece in pieces])
    else:
        log_probs = no_padding_2_padding(raw_log_probs, data)
    selected = _tensordict_to_padded_tensor_compat(data.select(
        "response_mask",
        "madpo_pair_index",
        "madpo_preference_side",
        "madpo_other_agent_delta",
        "preference_loss_mask",
    ))
    global_pair_count = tu.get_non_tensor_data(data, "madpo_global_pair_count", None)
    if global_pair_count is None:
        raise ValueError("MADPO loss requires madpo_global_pair_count from the trainer batch contract")
    dp_size = tu.get_non_tensor_data(data, "dp_size", 1)
    beta = _config_value(config, "madpo_beta", _config_value(config, "beta", 0.1))
    # TQ 的逐行标量经 TensorDict 取出后可能是 LinkedList，先还原为损失需要的张量。
    def scalar_rows(key: str) -> torch.Tensor:
        value = selected[key]
        if isinstance(value, torch.Tensor):
            return value
        return torch.as_tensor(to_python(value), device=log_probs.device)

    loss, pair_count = madpo_microbatch_loss(
        log_probs,
        selected["response_mask"],
        scalar_rows("madpo_pair_index"),
        scalar_rows("madpo_preference_side"),
        scalar_rows("madpo_other_agent_delta"),
        scalar_rows("preference_loss_mask"),
        beta=float(beta),
        global_pair_count=int(global_pair_count),
        dp_size=int(dp_size),
    )
    return loss, {
        "actor/madpo_loss": float(loss.detach()),
        "actor/madpo_pair_count": float(pair_count),
    }


def _row_count(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.shape[0])
    normalized = to_python(value)
    if isinstance(normalized, str) or not hasattr(normalized, "__len__"):
        raise TypeError("MADPO TQ fields must contain one value per row")
    return len(normalized)


def _config_value(config: Any, key: str, default: Any) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    try:
        return config.get(key, default)
    except (AttributeError, TypeError):
        return getattr(config, key, default)


__all__ = [
    "MaterializedPreferenceBatch",
    "PREFERENCE_LOGPROB_TQ_FIELDS",
    "PREFERENCE_TQ_FIELDS",
    "PreferencePairRows",
    "detached_other_agent_deltas",
    "madpo_actor_loss",
    "madpo_joint_pair_loss",
    "madpo_microbatch_loss",
    "materialize_pair_response_logprobs",
    "materialize_pair_response_tensors",
    "materialize_tq_preference_pairs",
    "sequence_logprob_sums",
    "snapshot_pair_deltas",
    "validate_preference_pairs",
]
