from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from trajweave.backends.verl.extensions.common.hooks import PPOExtensionHooks
from trajweave.backends.verl.schema import to_python

COMLRL_REINFORCE_TQ_FIELDS = (
    "joint_advantage",
    "effective_projected_joint_return",
    "joint_action_ids",
    "joint_transition_ids",
    "joint_return_components",
)
_DISPATCH_MARKERS = {"grpo", "reinforce_plus_plus", "remax", "rloo"}


@dataclass(frozen=True)
class CoMLRLReinforceHooks(PPOExtensionHooks):
    """Use rollout-computed CoMLRL v1.4.1 returns and advantages verbatim."""

    name: str = "comlrl_reinforce"

    def batch_schema_fields(self, stage: str, config: Any = None) -> tuple[str, ...]:
        del config
        return COMLRL_REINFORCE_TQ_FIELDS if stage == "advantage" else ()

    def tq_select_fields(
        self,
        stage: str,
        default_fields: tuple[str, ...],
        config: Any = None,
    ) -> tuple[str, ...]:
        del config
        fields = list(default_fields)
        if stage == "advantage":
            fields.extend(COMLRL_REINFORCE_TQ_FIELDS)
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
        del batch_keys, gamma, lam, num_repeat, norm_adv_by_std_in_grpo, fallback
        marker = _dispatch_marker(adv_estimator)
        if marker not in _DISPATCH_MARKERS:
            raise ValueError(
                "CoMLRL REINFORCE-family hooks require adv_estimator to be only a dispatch marker in "
                f"{sorted(_DISPATCH_MARKERS)}; got {adv_estimator!r}."
            )
        if "response_mask" not in data.batch:
            from verl.trainer.ppo.ray_trainer import compute_response_mask

            data.batch["response_mask"] = compute_response_mask(data)
        if "token_level_rewards" not in data.batch:
            raise KeyError("CoMLRL REINFORCE advantage requires token_level_rewards.")
        missing = [field for field in COMLRL_REINFORCE_TQ_FIELDS if field not in data.non_tensor_batch]
        if missing:
            raise KeyError(f"CoMLRL REINFORCE advantage requires non_tensor_batch fields: {missing}.")

        response_mask = data.batch["response_mask"].to(dtype=torch.bool)
        if response_mask.ndim != 2:
            raise ValueError("CoMLRL response_mask must be a [batch, response] tensor.")
        rewards = data.batch["token_level_rewards"]
        if rewards.shape != response_mask.shape:
            raise ValueError("CoMLRL token_level_rewards must align with response_mask.")
        row_count = response_mask.shape[0]
        real_rows = response_mask.any(dim=-1)
        non_tensors = data.non_tensor_batch
        advantages = _finite_scalar_tensor(
            non_tensors["joint_advantage"],
            field="joint_advantage",
            row_count=row_count,
            active_rows=real_rows,
            device=rewards.device,
            dtype=rewards.dtype,
        )
        effective_returns = _finite_scalar_tensor(
            non_tensors["effective_projected_joint_return"],
            field="effective_projected_joint_return",
            row_count=row_count,
            active_rows=real_rows,
            device=rewards.device,
            dtype=rewards.dtype,
        )
        action_ids = _batch_values(non_tensors["joint_action_ids"], row_count, "joint_action_ids")
        transition_ids = _batch_values(non_tensors["joint_transition_ids"], row_count, "joint_transition_ids")
        components = _batch_values(non_tensors["joint_return_components"], row_count, "joint_return_components")
        for row in torch.nonzero(real_rows, as_tuple=False).flatten().tolist():
            actions = _sequence_value(action_ids[row], "joint_action_ids", row)
            transitions = _sequence_value(transition_ids[row], "joint_transition_ids", row)
            returns = _sequence_value(components[row], "joint_return_components", row)
            if not actions:
                raise ValueError(f"CoMLRL real row {row} must reference at least one joint action.")
            if len(actions) != len(transitions) or len(actions) != len(returns):
                raise ValueError(
                    f"CoMLRL real row {row} requires equal action, transition, and return-component lengths; "
                    f"got {len(actions)}, {len(transitions)}, and {len(returns)}."
                )
            component_tensor = torch.as_tensor(returns, dtype=rewards.dtype, device=rewards.device)
            if not bool(torch.isfinite(component_tensor).all()):
                raise ValueError(f"CoMLRL joint_return_components must be finite on real row {row}.")

        masked_rewards = torch.where(response_mask, rewards, torch.zeros_like(rewards))
        if bool(response_mask.any()) and not bool(torch.isfinite(rewards[response_mask]).all()):
            raise ValueError("CoMLRL token_level_rewards must be finite on real response tokens.")
        observed_returns = masked_rewards.sum(dim=-1)
        validation_mode = _token_reward_validation_mode(config)
        mismatched = real_rows & ~torch.isclose(observed_returns, effective_returns, atol=1e-6, rtol=0.0)
        if bool(mismatched.any()) and validation_mode == "exact":
            rows = torch.nonzero(mismatched, as_tuple=False).flatten().tolist()
            raise ValueError(
                "CoMLRL token_level_rewards must sum to effective_projected_joint_return on every real row; "
                f"mismatched rows={rows}. Configure algorithm.comlrl.token_reward_validation explicitly "
                "for a KL or external-reward adjustment path."
            )

        data.batch["advantages"] = advantages.unsqueeze(-1) * response_mask
        data.batch["returns"] = effective_returns.unsqueeze(-1) * response_mask
        data.meta_info["comlrl_adv_estimator_dispatch_marker"] = marker
        data.meta_info["comlrl_token_reward_validation"] = validation_mode
        data.meta_info["comlrl_real_row_count"] = int(real_rows.sum().item())
        return data


def apply_comlrl_reinforce_patch(config: Any = None) -> None:
    """Enable hook-aware TQ execution without invoking VERL's native estimators."""

    from trajweave.backends.verl.extensions.common.runtime import apply_hook_aware_tq_runtime

    apply_hook_aware_tq_runtime(config)


def _dispatch_marker(value: Any) -> str:
    raw = getattr(value, "value", value)
    marker = str(raw).strip().lower()
    if "." in marker:
        marker = marker.rsplit(".", maxsplit=1)[-1]
    return marker


def _finite_scalar_tensor(
    values: Any,
    *,
    field: str,
    row_count: int,
    active_rows: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    normalized = _batch_values(values, row_count, field)
    try:
        tensor = torch.tensor([float(value) for value in normalized], dtype=dtype, device=device)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"CoMLRL {field} must contain one finite scalar per row.") from exc
    if bool(active_rows.any()) and not bool(torch.isfinite(tensor[active_rows]).all()):
        raise ValueError(f"CoMLRL {field} must contain only finite values on real rows.")
    return torch.where(active_rows, tensor, torch.zeros_like(tensor))


def _batch_values(values: Any, row_count: int, field: str) -> list[Any]:
    normalized = to_python(values)
    if not isinstance(normalized, list | tuple):
        normalized = [normalized]
    output = list(normalized)
    if len(output) != row_count:
        raise ValueError(f"CoMLRL field {field!r} has {len(output)} rows, expected {row_count}.")
    return output


def _sequence_value(value: Any, field: str, row: int) -> list[Any]:
    normalized = to_python(value)
    if not isinstance(normalized, list | tuple):
        raise TypeError(f"CoMLRL {field} on row {row} must be a sequence.")
    return list(normalized)


def _token_reward_validation_mode(config: Any) -> str:
    comlrl = _config_get(config, "comlrl", {}) or {}
    mode = str(_config_get(comlrl, "token_reward_validation", "exact")).strip().lower()
    if bool(_config_get(comlrl, "allow_token_reward_mismatch", False)):
        mode = "external"
    allowed = {"exact", "kl", "external", "skip"}
    if mode not in allowed:
        raise ValueError(f"algorithm.comlrl.token_reward_validation must be one of {sorted(allowed)}.")
    if mode == "kl" and not bool(_config_get(config, "use_kl_in_reward", False)):
        raise ValueError("token_reward_validation='kl' requires algorithm.use_kl_in_reward=true.")
    return mode


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
    "COMLRL_REINFORCE_TQ_FIELDS",
    "CoMLRLReinforceHooks",
    "apply_comlrl_reinforce_patch",
]
