from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from trajweave.backends.verl.extensions.common.hooks import PPOExtensionHooks
from trajweave.credit.comlrl.actor_critic import (
    deduplicate_maac_critic_indices,
    deduplicate_maac_critic_samples,
    normalize_agent_turn_advantages,
    one_step_td_targets,
    unclipped_mse_critic_loss,
)

CRITIC_INPUT_FIELDS = (
    "critic_input_ids",
    "critic_attention_mask",
    "critic_position_ids",
    "critic_loss_mask",
)
ACTOR_CRITIC_TQ_FIELDS = (
    "uid",
    "traj_uid",
    "agent_id",
    "worker_group",
    "turn_id",
    "joint_action_ids",
    "joint_transition_ids",
    "joint_reward",
    "joint_done",
    "joint_truncated",
    "critic_group",
    "critic_type",
    *CRITIC_INPUT_FIELDS,
)


@dataclass(frozen=True)
class PreparedActorCriticBatch:
    actor_advantages: torch.Tensor
    actor_returns: torch.Tensor
    scalar_advantages: torch.Tensor
    scalar_targets: torch.Tensor
    next_old_values: torch.Tensor
    critic_row_indices: tuple[int, ...]
    critic_fields: dict[str, Any]


def prepare_actor_critic_batch(
    fields: Mapping[str, Any],
    *,
    topology: str,
    gamma: float,
    normalize_by_population_std: bool = True,
) -> PreparedActorCriticBatch:
    """Prepare actor TD advantages and critic-only fields from joint TQ rows.

    This is a one-step TD path. It intentionally does not call VERL token GAE.
    """

    topology_name = str(topology).strip().lower()
    if topology_name not in {"independent", "centralized"}:
        raise ValueError("topology must be 'independent' or 'centralized'.")
    response_mask = _tensor_field(fields, "response_mask").to(dtype=torch.bool)
    if response_mask.ndim != 2:
        raise ValueError("response_mask must be a [batch, response] tensor.")
    row_count = response_mask.shape[0]
    rewards = _scalar_tensor(fields, "joint_reward", row_count=row_count, like=response_mask)
    old_values = _old_value_scalars(fields, row_count=row_count, like=rewards)
    done = _scalar_tensor(fields, "joint_done", row_count=row_count, like=rewards, dtype=torch.bool)
    truncated = _scalar_tensor(fields, "joint_truncated", row_count=row_count, like=rewards, dtype=torch.bool)
    agent_ids = _row_values(fields, "agent_id", row_count, fallback_key="worker_group")
    turn_ids = _row_values(fields, "turn_id", row_count)
    trajectory_ids = _row_values(fields, "traj_uid", row_count, fallback_key="uid")
    critic_masks = _scalar_tensor(fields, "critic_loss_mask", row_count=row_count, like=rewards).bool()
    active = response_mask.any(dim=-1) & critic_masks

    rows = [_row_mapping(fields, row, row_count) for row in range(row_count)]
    _validate_linear_actor_critic_rows(
        rows,
        topology=topology_name,
        active=active,
        trajectory_ids=trajectory_ids,
        agent_ids=agent_ids,
        turn_ids=turn_ids,
    )
    if topology_name == "centralized":
        deduplicate_maac_critic_samples(rows)
        critic_indices = deduplicate_maac_critic_indices(rows)
        next_values = _centralized_next_values(
            old_values,
            rows=rows,
            unique_indices=critic_indices,
            trajectory_ids=trajectory_ids,
            turn_ids=turn_ids,
            terminal=done | truncated,
        )
    else:
        critic_indices = [row for row in range(row_count) if bool(active[row])]
        next_values = _independent_next_values(
            old_values,
            trajectory_ids=trajectory_ids,
            agent_ids=agent_ids,
            turn_ids=turn_ids,
            active=active,
            terminal=done | truncated,
        )

    td = one_step_td_targets(
        rewards,
        old_values,
        next_values,
        done=done,
        truncated=truncated,
        gamma=gamma,
    )
    scalar_advantages = normalize_agent_turn_advantages(
        td.advantages,
        agent_ids=agent_ids,
        turn_ids=turn_ids,
        enabled=normalize_by_population_std,
        active_mask=active,
    )
    scalar_advantages = scalar_advantages * active
    scalar_targets = td.targets * active
    actor_advantages = scalar_advantages.unsqueeze(-1) * response_mask
    actor_returns = scalar_targets.unsqueeze(-1) * response_mask
    critic_fields = build_critic_worker_fields(fields, row_indices=critic_indices, targets=td.targets)
    return PreparedActorCriticBatch(
        actor_advantages=actor_advantages,
        actor_returns=actor_returns,
        scalar_advantages=scalar_advantages,
        scalar_targets=scalar_targets,
        next_old_values=next_values,
        critic_row_indices=tuple(critic_indices),
        critic_fields=critic_fields,
    )


def build_critic_worker_fields(
    fields: Mapping[str, Any],
    *,
    row_indices: Sequence[int],
    targets: torch.Tensor,
) -> dict[str, Any]:
    """Map critic-prefixed TQ inputs to the standard TrainingWorker contract."""

    row_count = targets.shape[0]
    output: dict[str, Any] = {}
    for source, target in (
        ("critic_input_ids", "input_ids"),
        ("critic_attention_mask", "attention_mask"),
        ("critic_position_ids", "position_ids"),
    ):
        if source not in fields:
            raise KeyError(f"Actor-critic batch is missing {source!r}; actor input fallback is forbidden.")
        output[target] = _select_rows(fields[source], row_indices, row_count=row_count)

    selected_masks = _select_rows(fields["critic_loss_mask"], row_indices, row_count=row_count)
    loss_masks = _critic_token_loss_masks(output["attention_mask"], selected_masks)
    selected_targets = targets[torch.as_tensor(row_indices, dtype=torch.long, device=targets.device)]
    output["loss_mask"] = loss_masks
    output["response_mask"] = loss_masks
    output["returns"] = _place_targets_at_loss_token(selected_targets, loss_masks)
    return output


def verl_unclipped_mse_critic_loss(
    config: Any,
    model_output: Mapping[str, Any],
    data: Any,
    dp_group=None,
    *,
    value_loss_coef: float = 0.6,
):
    """TrainingWorker ``set_loss_fn`` callback for separate IAC/MAAC critics."""

    del config, dp_group
    predictions = _aligned_value_tensor(model_output["values"], field="model_output['values']")
    targets = _aligned_value_tensor(data["returns"], field="data['returns']").to(predictions)
    mask = _aligned_value_tensor(data["loss_mask"], field="data['loss_mask']").to(
        dtype=torch.bool, device=predictions.device
    )
    if predictions.shape != targets.shape or predictions.shape != mask.shape:
        raise ValueError(
            "Critic values, returns, and loss_mask must have identical token positions after nested flattening."
        )
    coefficient = float(value_loss_coef)
    if coefficient < 0.0:
        raise ValueError("CoMLRL value_loss_coef must be non-negative.")
    mse_loss = unclipped_mse_critic_loss(predictions, targets, mask)
    loss = mse_loss * coefficient
    metrics = {
        "critic/mse_loss": float(mse_loss.detach()),
        "critic/value_loss": float(loss.detach()),
        "critic/value_loss_coef": coefficient,
        "critic/vpred_mean": float(predictions[mask].detach().mean()),
    }
    return loss, metrics


def _aligned_value_tensor(value: Any, *, field: str) -> torch.Tensor:
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


@dataclass(frozen=True)
class CoMLRLActorCriticHooks(PPOExtensionHooks):
    topology: str = "independent"
    name: str = "comlrl_actor_critic"

    def batch_schema_fields(self, stage: str, config: Any = None) -> tuple[str, ...]:
        del config
        return ACTOR_CRITIC_TQ_FIELDS if stage == "advantage" else ()

    def tq_select_fields(
        self,
        stage: str,
        default_fields: tuple[str, ...],
        config: Any = None,
    ) -> tuple[str, ...]:
        del config
        fields = list(default_fields)
        if stage == "advantage":
            fields.extend(ACTOR_CRITIC_TQ_FIELDS)
        return tuple(dict.fromkeys(fields))

    def compute_advantage(
        self,
        data: Any,
        *,
        batch_keys: list[str],
        adv_estimator: Any,
        gamma: float,
        lam: float,
        num_repeat: int,
        norm_adv_by_std_in_grpo: bool,
        config: Any = None,
        fallback: Any = None,
    ) -> Any:
        del batch_keys, adv_estimator, lam, num_repeat, fallback
        fields: dict[str, Any] = dict(data.batch)
        fields.update(data.non_tensor_batch)
        normalize = bool(_config_get(config, "normalize_advantages", norm_adv_by_std_in_grpo))
        prepared = prepare_actor_critic_batch(
            fields,
            topology=self.topology,
            gamma=float(gamma),
            normalize_by_population_std=normalize,
        )
        data.batch["advantages"] = prepared.actor_advantages
        data.batch["returns"] = prepared.actor_returns
        data.meta_info["comlrl_critic_row_indices"] = prepared.critic_row_indices
        data.meta_info["comlrl_critic_fields"] = prepared.critic_fields
        data.meta_info["comlrl_uses_token_gae"] = False
        return data


@dataclass(frozen=True)
class IACActorCriticHooks(CoMLRLActorCriticHooks):
    topology: str = "independent"
    name: str = "comlrl_iac_actor_critic"


@dataclass(frozen=True)
class MAACActorCriticHooks(CoMLRLActorCriticHooks):
    topology: str = "centralized"
    name: str = "comlrl_maac_actor_critic"


def _validate_linear_actor_critic_rows(
    rows: list[dict[str, Any]],
    *,
    topology: str,
    active: torch.Tensor,
    trajectory_ids: list[Any],
    agent_ids: list[Any],
    turn_ids: list[Any],
) -> None:
    iac_keys: set[tuple[str, str, int]] = set()
    maac_turns: dict[tuple[str, int], str] = {}
    maac_transition_turns: dict[tuple[str, str], int] = {}
    for row_index, row in enumerate(rows):
        if not bool(active[row_index]):
            continue
        actions = row.get("joint_action_ids")
        transitions = row.get("joint_transition_ids")
        if isinstance(actions, str):
            actions = [actions]
        if isinstance(transitions, str):
            transitions = [transitions]
        if not isinstance(actions, list | tuple) or len(actions) != 1:
            raise ValueError(
                f"Active actor-critic row {row_index} must reference exactly one joint action; "
                "cross/full-tree candidates are unsupported."
            )
        if not isinstance(transitions, list | tuple) or len(transitions) != 1:
            raise ValueError(
                f"Active actor-critic row {row_index} must reference exactly one joint transition; "
                "cross/full-tree candidates are unsupported."
            )
        transition_id = str(transitions[0])
        if not transition_id or transition_id.startswith(("__padding__", "__missing__")):
            raise ValueError(f"Active actor-critic row {row_index} has no real joint transition id.")
        trajectory_id = str(trajectory_ids[row_index])
        turn_id = int(turn_ids[row_index])
        if topology == "independent":
            key = (trajectory_id, str(agent_ids[row_index]), turn_id)
            if key in iac_keys:
                raise ValueError(
                    "IAC supports at most one transition per (traj_uid, agent, turn); "
                    f"duplicate={key!r}. Cross/full-tree candidates are unsupported."
                )
            iac_keys.add(key)
        else:
            key = (trajectory_id, turn_id)
            previous = maac_turns.setdefault(key, transition_id)
            if previous != transition_id:
                raise ValueError(
                    "MAAC supports at most one unique transition per (traj_uid, turn); "
                    f"key={key!r}, transitions={sorted({previous, transition_id})}. "
                    "Cross/full-tree candidates are unsupported."
                )
            transition_key = (trajectory_id, transition_id)
            previous_turn = maac_transition_turns.setdefault(transition_key, turn_id)
            if previous_turn != turn_id:
                raise ValueError(
                    f"MAAC transition {transition_key!r} appears at multiple turns; linear chains require one turn."
                )


def _independent_next_values(
    values: torch.Tensor,
    *,
    trajectory_ids: list[Any],
    agent_ids: list[Any],
    turn_ids: list[Any],
    active: torch.Tensor,
    terminal: torch.Tensor,
) -> torch.Tensor:
    output = torch.zeros_like(values)
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in range(values.shape[0]):
        if bool(active[row]):
            grouped[(str(trajectory_ids[row]), str(agent_ids[row]))].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: (int(turn_ids[row]), row))
        for current, following in zip(rows, rows[1:], strict=False):
            if int(turn_ids[following]) != int(turn_ids[current]) + 1:
                raise ValueError("IAC TD requires contiguous per-agent turns; a bootstrap transition is missing.")
            output[current] = values[following]
        if rows and not bool(terminal[rows[-1]]):
            raise ValueError("IAC TD requires the final sampled transition to be done or truncated.")
    return output


def _centralized_next_values(
    values: torch.Tensor,
    *,
    rows: list[dict[str, Any]],
    unique_indices: list[int],
    trajectory_ids: list[Any],
    turn_ids: list[Any],
    terminal: torch.Tensor,
) -> torch.Tensor:
    output = torch.zeros_like(values)
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in unique_indices:
        grouped[str(trajectory_ids[row])].append(row)
    transition_value: dict[tuple[str, str], torch.Tensor] = {}
    transition_next: dict[tuple[str, str], torch.Tensor] = {}
    for trajectory_id, indexes in grouped.items():
        source_turns = [int(turn_ids[row]) for row in indexes]
        if len(source_turns) != len(set(source_turns)):
            raise ValueError(
                "MAAC TD cannot infer a unique next value for multiple joint transitions at the same trajectory turn."
            )
        indexes.sort(key=lambda row: (int(turn_ids[row]), row))
        for position, row in enumerate(indexes):
            transition_id = str(rows[row]["joint_transition_ids"][0])
            composite = (trajectory_id, transition_id)
            transition_value[composite] = values[row]
            if position + 1 < len(indexes):
                following = indexes[position + 1]
                if int(turn_ids[following]) != int(turn_ids[row]) + 1:
                    raise ValueError("MAAC TD requires contiguous joint turns; a bootstrap transition is missing.")
                transition_next[composite] = values[following]
            else:
                if not bool(terminal[row]):
                    raise ValueError("MAAC TD requires the final sampled transition to be done or truncated.")
                transition_next[composite] = values.new_zeros(())
    for row, sample in enumerate(rows):
        values_for_row = sample.get("joint_transition_ids")
        if isinstance(values_for_row, list | tuple) and len(values_for_row) == 1:
            transition_id = str(values_for_row[0])
            composite = (str(trajectory_ids[row]), transition_id)
            if composite in transition_value:
                if not torch.allclose(values[row], transition_value[composite]):
                    raise ValueError(f"MAAC duplicate transition {composite!r} has inconsistent old critic values.")
                output[row] = transition_next[composite]
    return output


def _old_value_scalars(fields: Mapping[str, Any], *, row_count: int, like: torch.Tensor) -> torch.Tensor:
    key = "old_values" if "old_values" in fields else "values"
    if key not in fields:
        raise KeyError("Actor-critic TD preparation requires old_values (or values) from critic inference.")
    values = torch.as_tensor(fields[key], dtype=torch.float32, device=like.device)
    if values.ndim == 1:
        if values.shape[0] != row_count:
            raise ValueError("old critic values must have one scalar per TQ row.")
        return values
    if values.ndim != 2 or values.shape[0] != row_count:
        raise ValueError("old critic values must be [batch] or [batch, critic_sequence].")
    if "critic_attention_mask" not in fields:
        raise KeyError("Token critic values require critic_attention_mask.")
    attention = torch.as_tensor(fields["critic_attention_mask"], device=values.device)
    if attention.shape != values.shape:
        raise ValueError("critic_attention_mask must align with token critic values.")
    last = _last_valid_token_indices(attention)
    return values.gather(1, last.unsqueeze(-1)).squeeze(-1)


def _critic_token_loss_masks(attention_masks: Any, row_masks: Any) -> Any:
    if torch.is_tensor(attention_masks):
        attention = attention_masks.to(dtype=torch.bool)
        if attention.ndim != 2:
            raise ValueError("critic_attention_mask must be a 2D tensor after batching.")
        rows = torch.as_tensor(row_masks, dtype=torch.bool, device=attention.device).reshape(-1)
        mask = torch.zeros_like(attention, dtype=torch.float32)
        valid = rows & attention.any(dim=-1)
        indexes = torch.nonzero(valid, as_tuple=False).flatten()
        if indexes.numel():
            last = _last_valid_token_indices(attention)
            mask[indexes, last[indexes]] = 1.0
        return mask
    attention_rows = list(attention_masks)
    masks = _as_list(row_masks)
    output = []
    for attention, row_mask in zip(attention_rows, masks, strict=True):
        tensor = torch.as_tensor(attention, dtype=torch.bool)
        mask = torch.zeros_like(tensor, dtype=torch.float32)
        valid = torch.nonzero(tensor, as_tuple=False).flatten()
        if bool(float(row_mask)) and valid.numel():
            mask[valid[-1]] = 1.0
        output.append(mask)
    return output


def _last_valid_token_indices(attention: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(attention.shape[-1], device=attention.device).expand_as(attention)
    return positions.masked_fill(~attention.to(dtype=torch.bool), -1).amax(dim=-1).clamp_min(0)


def _place_targets_at_loss_token(targets: torch.Tensor, masks: Any) -> Any:
    if torch.is_tensor(masks):
        return masks.to(targets) * targets.unsqueeze(-1)
    return [mask.to(targets) * target for mask, target in zip(masks, targets, strict=True)]


def _tensor_field(fields: Mapping[str, Any], key: str) -> torch.Tensor:
    if key not in fields:
        raise KeyError(f"Actor-critic batch is missing {key!r}.")
    return torch.as_tensor(fields[key])


def _scalar_tensor(
    fields: Mapping[str, Any],
    key: str,
    *,
    row_count: int,
    like: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if key not in fields:
        raise KeyError(f"Actor-critic batch is missing {key!r}.")
    tensor = torch.as_tensor(_as_list(fields[key]), dtype=dtype, device=like.device).reshape(-1)
    if tensor.numel() != row_count:
        raise ValueError(f"{key} must have one scalar per row.")
    return tensor


def _row_values(fields: Mapping[str, Any], key: str, row_count: int, fallback_key: str | None = None) -> list[Any]:
    source = key if key in fields else fallback_key
    if source is None or source not in fields:
        raise KeyError(f"Actor-critic batch is missing {key!r}.")
    values = _as_list(fields[source])
    if len(values) != row_count:
        raise ValueError(f"{source} must have one value per row.")
    return values


def _row_mapping(fields: Mapping[str, Any], row: int, row_count: int) -> dict[str, Any]:
    output = {}
    for key in ACTOR_CRITIC_TQ_FIELDS:
        if key not in fields:
            continue
        value = fields[key]
        try:
            output[key] = value[row]
        except (IndexError, KeyError, TypeError):
            values = _as_list(value)
            output[key] = values[row] if len(values) == row_count else value
        if torch.is_tensor(output[key]) and output[key].ndim == 0:
            output[key] = output[key].item()
    return output


def _select_rows(value: Any, rows: Sequence[int], *, row_count: int) -> Any:
    if torch.is_tensor(value):
        index = torch.as_tensor(rows, dtype=torch.long, device=value.device)
        return value.index_select(0, index)
    values = _as_list(value)
    if len(values) != row_count:
        raise ValueError("Critic input field row count does not match targets.")
    return [values[row] for row in rows]


def _as_list(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list | tuple):
        return list(value)
    return [value]


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    try:
        return config.get(key, default)
    except (AttributeError, TypeError):
        return getattr(config, key, default)


__all__ = [
    "ACTOR_CRITIC_TQ_FIELDS",
    "CRITIC_INPUT_FIELDS",
    "CoMLRLActorCriticHooks",
    "IACActorCriticHooks",
    "MAACActorCriticHooks",
    "PreparedActorCriticBatch",
    "build_critic_worker_fields",
    "prepare_actor_critic_batch",
    "verl_unclipped_mse_critic_loss",
]
