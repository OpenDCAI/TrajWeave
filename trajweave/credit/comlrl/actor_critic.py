from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from trajweave.core.specs import TeamSpec


@dataclass(frozen=True)
class OneStepTD:
    targets: torch.Tensor
    advantages: torch.Tensor


def build_iac_critic_text(
    prompt: str,
    *,
    action: str | None = None,
    critic_type: str = "v",
    agent_name: str | None = None,
) -> str:
    """Build deterministic local-history V(h) or Q(h,a) critic text."""

    kind = _critic_type(critic_type)
    del agent_name
    text = str(prompt)
    if kind == "q":
        if action is None:
            raise ValueError("IAC Q critic text requires the selected action.")
        text += str(action)
    elif action is not None:
        raise ValueError("IAC V critic text must not include an action.")
    return text


def build_iac_v_critic_text(prompt: str, *, agent_name: str | None = None) -> str:
    return build_iac_critic_text(prompt, critic_type="v", agent_name=agent_name)


def build_iac_q_critic_text(prompt: str, action: str, *, agent_name: str | None = None) -> str:
    return build_iac_critic_text(prompt, action=action, critic_type="q", agent_name=agent_name)


def build_maac_critic_text(
    team: TeamSpec | Sequence[str],
    prompts_by_agent: Mapping[str, str] | Sequence[str],
    *,
    actions_by_agent: Mapping[str, str] | Sequence[str] | None = None,
    critic_type: str = "v",
) -> str:
    """Build joint critic text in immutable ``TeamSpec.agents`` order."""

    kind = _critic_type(critic_type)
    names = (
        tuple(agent.name for agent in team.agents)
        if isinstance(team, TeamSpec)
        else tuple(str(agent_name) for agent_name in team)
    )
    if not names or any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("MAAC critic agent order must contain unique non-empty agent names.")
    prompts = _ordered_agent_values(names, prompts_by_agent, field="prompts")
    sections = [f"[Agent {index}] {prompt}" for index, prompt in enumerate(prompts)]
    if kind == "q":
        if actions_by_agent is None:
            raise ValueError("MAAC Q critic text requires one joint action component per team agent.")
        actions = _ordered_agent_values(names, actions_by_agent, field="actions")
        sections.append("[Joint Action]")
        sections.extend(f"[Agent {index} action]\n{action}" for index, action in enumerate(actions))
    elif actions_by_agent is not None:
        raise ValueError("MAAC V critic text must not include joint actions.")
    return "\n\n".join(sections)


def build_maac_v_critic_text(team: TeamSpec, prompts_by_agent: Mapping[str, str] | Sequence[str]) -> str:
    return build_maac_critic_text(team, prompts_by_agent, critic_type="v")


def build_maac_q_critic_text(
    team: TeamSpec,
    prompts_by_agent: Mapping[str, str] | Sequence[str],
    actions_by_agent: Mapping[str, str] | Sequence[str],
) -> str:
    return build_maac_critic_text(
        team,
        prompts_by_agent,
        actions_by_agent=actions_by_agent,
        critic_type="q",
    )


def one_step_td_targets(
    rewards: torch.Tensor | Sequence[float],
    old_values: torch.Tensor | Sequence[float],
    next_old_values: torch.Tensor | Sequence[float],
    *,
    done: torch.Tensor | Sequence[bool],
    truncated: torch.Tensor | Sequence[bool],
    gamma: float,
) -> OneStepTD:
    """Terminal-aware one-step TD; truncation is terminal by contract."""

    values = torch.as_tensor(old_values)
    if not values.is_floating_point():
        values = values.to(torch.float32)
    rewards_tensor = torch.as_tensor(rewards, dtype=values.dtype, device=values.device)
    next_values = torch.as_tensor(next_old_values, dtype=values.dtype, device=values.device)
    done_tensor = torch.as_tensor(done, dtype=torch.bool, device=values.device)
    truncated_tensor = torch.as_tensor(truncated, dtype=torch.bool, device=values.device)
    try:
        rewards_tensor, values, next_values, done_tensor, truncated_tensor = torch.broadcast_tensors(
            rewards_tensor, values, next_values, done_tensor, truncated_tensor
        )
    except RuntimeError as exc:
        raise ValueError("TD rewards, values, next values, and terminal flags must be broadcast-compatible.") from exc
    _require_finite("rewards", rewards_tensor)
    _require_finite("old_values", values)
    _require_finite("next_old_values", next_values)
    gamma_value = float(gamma)
    if not torch.isfinite(torch.tensor(gamma_value)):
        raise FloatingPointError("gamma must be finite.")
    terminal = done_tensor | truncated_tensor
    targets = torch.where(terminal, rewards_tensor, rewards_tensor + gamma_value * next_values)
    advantages = targets - values
    _require_finite("TD targets", targets)
    _require_finite("TD advantages", advantages)
    return OneStepTD(targets=targets, advantages=advantages)


def normalize_agent_turn_advantages(
    advantages: torch.Tensor | Sequence[float],
    *,
    agent_ids: Sequence[Any],
    turn_ids: Sequence[Any],
    enabled: bool = True,
    epsilon: float = 1e-6,
    active_mask: torch.Tensor | Sequence[bool] | None = None,
) -> torch.Tensor:
    """Population z-score within each agent-and-turn candidate group."""

    tensor = torch.as_tensor(advantages)
    if not tensor.is_floating_point():
        tensor = tensor.to(torch.float32)
    _require_finite("advantages", tensor)
    if tensor.ndim != 1:
        raise ValueError("Agent-turn normalization expects one scalar advantage per row.")
    if len(agent_ids) != tensor.numel() or len(turn_ids) != tensor.numel():
        raise ValueError("agent_ids and turn_ids must have one value per advantage row.")
    active = (
        torch.ones(tensor.numel(), dtype=torch.bool, device=tensor.device)
        if active_mask is None
        else torch.as_tensor(active_mask, dtype=torch.bool, device=tensor.device)
    )
    if active.shape != tensor.shape:
        raise ValueError("active_mask must align with scalar advantages.")
    output = torch.zeros_like(tensor)
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row, (agent_id, turn_id) in enumerate(zip(agent_ids, turn_ids, strict=True)):
        if bool(active[row]):
            groups[(str(agent_id), str(turn_id))].append(row)
    for rows in groups.values():
        index = torch.tensor(rows, dtype=torch.long, device=tensor.device)
        group = tensor[index]
        if enabled and len(rows) > 1:
            mean = group.mean()
            std = group.std(unbiased=False)
            output[index] = (group - mean) / std.clamp_min(float(epsilon))
        else:
            output[index] = group
    _require_finite("normalized advantages", output)
    return output


def ratio_free_sequence_actor_loss(
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """GPG-style ``seq-mean-token-sum`` loss with no policy ratio."""

    if log_probs.ndim != 2 or loss_mask.shape != log_probs.shape:
        raise ValueError("log_probs and loss_mask must be aligned [batch, sequence] tensors.")
    mask = loss_mask.to(dtype=torch.bool, device=log_probs.device)
    advantage = advantages.to(dtype=log_probs.dtype, device=log_probs.device)
    if advantage.ndim == 1:
        if advantage.shape[0] != log_probs.shape[0]:
            raise ValueError("Scalar advantages must have one value per sequence.")
        advantage = advantage.unsqueeze(-1)
    elif advantage.shape != log_probs.shape:
        raise ValueError("Token advantages must align with log_probs.")
    active = mask.any(dim=-1)
    if not bool(active.any()):
        raise ValueError("Actor loss requires at least one non-padding sequence.")
    _require_finite("log_probs", log_probs[mask])
    expanded_advantage = advantage.expand_as(log_probs) if advantage.shape[-1] == 1 else advantage
    _require_finite("advantages", expanded_advantage[mask])
    masked_log_probs = log_probs.masked_fill(~mask, 0.0)
    masked_advantages = expanded_advantage.masked_fill(~mask, 0.0)
    per_sequence = -(masked_log_probs * masked_advantages).sum(dim=-1)
    loss = per_sequence[active].mean()
    _require_finite("actor loss", loss)
    return loss


def unclipped_mse_critic_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    prediction, target = torch.broadcast_tensors(predictions, targets.to(predictions))
    mask = _broadcast_loss_mask(loss_mask, prediction)
    if not bool(mask.any()):
        raise ValueError("Critic loss requires at least one non-padding value target.")
    _require_finite("critic predictions", prediction[mask])
    _require_finite("critic targets", target[mask])
    difference = prediction.masked_fill(~mask, 0.0) - target.masked_fill(~mask, 0.0)
    loss = difference.square()[mask].mean()
    _require_finite("critic loss", loss)
    return loss


def iac_shared_head_clipped_value_loss(
    predictions: torch.Tensor,
    old_values: torch.Tensor,
    targets: torch.Tensor,
    *,
    clip_range: float,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Upstream-compatible max(unclipped, clipped) value loss for a true shared head."""

    prediction, old, target = torch.broadcast_tensors(predictions, old_values.to(predictions), targets.to(predictions))
    mask = _broadcast_loss_mask(loss_mask, prediction)
    if not bool(mask.any()):
        raise ValueError("Shared-head value loss requires at least one non-padding target.")
    _require_finite("shared-head predictions", prediction[mask])
    _require_finite("shared-head old values", old[mask])
    _require_finite("shared-head targets", target[mask])
    clip = float(clip_range)
    if not torch.isfinite(torch.tensor(clip)) or clip < 0:
        raise ValueError("clip_range must be a finite non-negative value.")
    safe_prediction = prediction.masked_fill(~mask, 0.0)
    safe_old = old.masked_fill(~mask, 0.0)
    safe_target = target.masked_fill(~mask, 0.0)
    clipped = safe_old + (safe_prediction - safe_old).clamp(-clip, clip)
    error = torch.maximum((safe_prediction - safe_target).square(), (clipped - safe_target).square())
    loss = error[mask].mean()
    _require_finite("shared-head value loss", loss)
    return loss


def deduplicate_maac_critic_samples(samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return one centralized critic row per real joint transition, preserving order."""

    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for row, sample in enumerate(samples):
        if not _sample_is_active(sample):
            continue
        trajectory_id, transition_id = _single_transition_key(sample, row=row)
        candidate = dict(sample)
        candidate["joint_transition_id"] = transition_id
        existing = unique.get((trajectory_id, transition_id))
        if existing is None:
            unique[(trajectory_id, transition_id)] = candidate
            continue
        for field in (
            "critic_group",
            "critic_type",
            "critic_input_ids",
            "critic_attention_mask",
            "critic_position_ids",
            "joint_reward",
            "joint_done",
            "joint_truncated",
        ):
            if field in existing and field in candidate and not _equal_value(existing[field], candidate[field]):
                raise ValueError(
                    "Duplicate MAAC transition "
                    f"{(trajectory_id, transition_id)!r} has inconsistent centralized field {field!r}."
                )
    return list(unique.values())


def deduplicate_maac_critic_indices(samples: Sequence[Mapping[str, Any]]) -> list[int]:
    seen: set[tuple[str, str]] = set()
    output: list[int] = []
    for row, sample in enumerate(samples):
        if not _sample_is_active(sample):
            continue
        transition_key = _single_transition_key(sample, row=row)
        if transition_key not in seen:
            seen.add(transition_key)
            output.append(row)
    return output


def _ordered_agent_values(
    names: tuple[str, ...], values: Mapping[str, str] | Sequence[str], *, field: str
) -> tuple[str, ...]:
    if isinstance(values, Mapping):
        missing = [name for name in names if name not in values]
        extra = sorted(set(str(key) for key in values) - set(names))
        if missing or extra:
            raise ValueError(f"Joint {field} must exactly match TeamSpec agents; missing={missing}, extra={extra}.")
        return tuple(str(values[name]) for name in names)
    if isinstance(values, str) or len(values) != len(names):
        raise ValueError(f"Joint {field} must contain exactly {len(names)} values in TeamSpec order.")
    return tuple(str(value) for value in values)


def _critic_type(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in {"v", "q"}:
        raise ValueError(f"critic_type must be 'v' or 'q', got {value!r}.")
    return normalized


def _sample_is_active(sample: Mapping[str, Any]) -> bool:
    mask = sample.get("critic_loss_mask", 1.0)
    if torch.is_tensor(mask):
        return bool(mask.to(dtype=torch.bool).any())
    return bool(float(mask))


def _single_transition_key(sample: Mapping[str, Any], *, row: int) -> tuple[str, str]:
    trajectory_id = sample.get("traj_uid", sample.get("uid"))
    normalized_trajectory = str(trajectory_id or "")
    if not normalized_trajectory or normalized_trajectory.startswith(("__padding__", "__missing__")):
        raise ValueError(f"MAAC critic row {row} has no real trajectory id.")
    value = sample.get("joint_transition_id")
    if value is None:
        values = sample.get("joint_transition_ids")
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list | tuple) or len(values) != 1:
            raise ValueError(f"MAAC critic row {row} must reference exactly one joint transition.")
        value = values[0]
    normalized = str(value)
    if not normalized or normalized.startswith(("__padding__", "__missing__")):
        raise ValueError(f"MAAC critic row {row} has no real joint transition id.")
    return normalized_trajectory, normalized


def _equal_value(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return bool(torch.equal(torch.as_tensor(left), torch.as_tensor(right)))
    return left == right


def _broadcast_loss_mask(loss_mask: torch.Tensor | None, target: torch.Tensor) -> torch.Tensor:
    if loss_mask is None:
        return torch.ones_like(target, dtype=torch.bool)
    mask = loss_mask.to(dtype=torch.bool, device=target.device)
    if mask.ndim == target.ndim - 1 and mask.shape == target.shape[:-1]:
        mask = mask.unsqueeze(-1)
    try:
        return torch.broadcast_to(mask, target.shape)
    except RuntimeError as exc:
        raise ValueError("loss_mask must be broadcast-compatible with predictions.") from exc


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} contains non-finite values.")


# Compatibility aliases with concise algorithm names.
terminal_aware_one_step_td = one_step_td_targets
population_normalize_by_agent_turn = normalize_agent_turn_advantages
sequence_actor_loss = ratio_free_sequence_actor_loss
critic_mse_loss = unclipped_mse_critic_loss
shared_head_clipped_value_loss = iac_shared_head_clipped_value_loss


__all__ = [
    "OneStepTD",
    "build_iac_critic_text",
    "build_iac_q_critic_text",
    "build_iac_v_critic_text",
    "build_maac_critic_text",
    "build_maac_q_critic_text",
    "build_maac_v_critic_text",
    "critic_mse_loss",
    "deduplicate_maac_critic_indices",
    "deduplicate_maac_critic_samples",
    "iac_shared_head_clipped_value_loss",
    "normalize_agent_turn_advantages",
    "one_step_td_targets",
    "population_normalize_by_agent_turn",
    "ratio_free_sequence_actor_loss",
    "sequence_actor_loss",
    "shared_head_clipped_value_loss",
    "terminal_aware_one_step_td",
    "unclipped_mse_critic_loss",
]
