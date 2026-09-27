from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from trajweave.backends.verl.extensions.common.hooks import _DEFAULT_ADVANTAGE_TQ_FIELDS, PPOExtensionHooks
from trajweave.credit.c3 import compute_c3_scalar_credit

C3_TQ_FIELDS = (
    "node_id",
    "c3_group_id",
    "c3_depth",
    "c3_role_index",
    "c3_parent_id",
    "c3_is_leaf",
    "c3_prefix_text",
    "c3_subtree_return",
    "c3_leaf_count",
)


@dataclass(frozen=True)
class C3ContextualCounterfactualHooks(PPOExtensionHooks):
    name: str = "c3_contextual_counterfactual"

    def batch_schema_fields(self, stage: str, config: Any = None) -> tuple[str, ...]:
        return C3_TQ_FIELDS if stage == "advantage" else ()

    def tq_select_fields(
        self,
        stage: str,
        default_fields: tuple[str, ...] | None = None,
        config: Any = None,
    ) -> tuple[str, ...]:
        if default_fields is None:
            default_fields = _DEFAULT_ADVANTAGE_TQ_FIELDS if stage == "advantage" else ()
        fields = list(super().tq_select_fields(stage, default_fields, config=config))
        if stage == "advantage":
            fields.extend(C3_TQ_FIELDS)
        return tuple(dict.fromkeys(fields))

    def compute_advantage(
        self,
        data: Any,
        *,
        batch_keys: list[str] | None = None,
        adv_estimator: Any = None,
        gamma: float = 1.0,
        lam: float = 1.0,
        num_repeat: int = 1,
        norm_adv_by_std_in_grpo: bool = True,
        config: Any = None,
        fallback: Any = None,
    ) -> Any:
        del batch_keys, adv_estimator, gamma, lam, num_repeat, fallback
        import torch

        c3 = _config_get(config, "c3", {}) or {}
        variant = str(_config_get(c3, "credit_variant", "reward_only")).strip().lower()
        if variant != "reward_only":
            raise RuntimeError(f"C3 {variant} requires the trajweave_c3_critic_sync trainer.")
        response_mask = data.batch["response_mask"]
        row_count = response_mask.shape[0]
        groups = _row_values(data.non_tensor_batch, "c3_group_id", row_count)
        subtree_returns = [float(value) for value in _row_values(data.non_tensor_batch, "c3_subtree_return", row_count)]
        active = response_mask.bool().any(dim=-1).detach().cpu().tolist()
        real_rows = [row for row, value in enumerate(active) if value]
        if not real_rows:
            raise ValueError("C3 advantage computation requires non-padding rows.")
        result = compute_c3_scalar_credit(
            subtree_returns=[subtree_returns[row] for row in real_rows],
            group_ids=[groups[row] for row in real_rows],
            variant="reward_only",
            baseline_mode=str(_config_get(c3, "baseline_mode", "loo")),
            normalize=bool(_config_get(c3, "normalize_advantages", norm_adv_by_std_in_grpo)),
        )
        scalar = torch.zeros(row_count, dtype=torch.float32, device=response_mask.device)
        returns = torch.zeros_like(scalar)
        for offset, row in enumerate(real_rows):
            scalar[row] = result.advantages[offset]
            returns[row] = subtree_returns[row]
        mask = response_mask.to(dtype=torch.float32)
        data.batch["advantages"] = scalar.unsqueeze(-1) * mask
        data.batch["returns"] = returns.unsqueeze(-1) * mask
        return data


def apply_c3_contextual_counterfactual_patch(config: Any = None) -> None:
    from trajweave.backends.verl.extensions.common.runtime import apply_hook_aware_tq_runtime

    apply_hook_aware_tq_runtime(config)


def _row_values(fields: Any, name: str, row_count: int) -> list[Any]:
    if name not in fields:
        raise KeyError(f"C3 advantage computation requires {name!r}.")
    values = fields[name]
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, list | tuple):
        values = [values]
    output = list(values)
    if len(output) != row_count:
        raise ValueError(f"C3 field {name!r} has {len(output)} rows, expected {row_count}.")
    return output


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    try:
        return config.get(key, default)
    except (AttributeError, TypeError):
        return getattr(config, key, default)
