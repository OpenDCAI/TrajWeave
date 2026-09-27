from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from trajweave.backends.verl.extensions.common.hooks import _DEFAULT_ADVANTAGE_TQ_FIELDS, PPOExtensionHooks

MARFT_PPO_TQ_FIELDS = (
    "agent_id",
    "policy_group",
    "worker_group",
    "traj_uid",
    "turn_id",
    "marft_node_id",
    "marft_layer",
    "marft_role_index",
    "marft_credit_strategy",
    "marft_credit_discount",
    "marft_return_gamma",
    "marft_step_reward",
    "marft_projected_return",
)


@dataclass(frozen=True)
class MARFTPPOHooks(PPOExtensionHooks):
    """Validate MARFT row credit before delegating token credit to VERL GAE."""

    name: str = "marft_ctde"

    def batch_schema_fields(self, stage: str, config: Any = None) -> tuple[str, ...]:
        del config
        return MARFT_PPO_TQ_FIELDS if stage == "advantage" else ()

    def tq_select_fields(
        self,
        stage: str,
        default_fields: tuple[str, ...] | None = None,
        config: Any = None,
    ) -> tuple[str, ...]:
        del config
        if default_fields is None:
            default_fields = _DEFAULT_ADVANTAGE_TQ_FIELDS if stage == "advantage" else ()
        fields = list(default_fields)
        if stage == "advantage":
            fields.extend(MARFT_PPO_TQ_FIELDS)
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
        import torch

        if _estimator_name(adv_estimator) != "gae":
            raise ValueError(f"MARFT PPO requires algorithm.adv_estimator=gae, got {adv_estimator!r}.")
        if fallback is None:
            raise RuntimeError("MARFT PPO requires VERL's fallback GAE implementation.")
        missing = [field for field in MARFT_PPO_TQ_FIELDS if field not in data.non_tensor_batch]
        if missing:
            raise KeyError(f"MARFT PPO rows are missing metadata fields: {missing}.")
        if "token_level_rewards" not in data.batch or "response_mask" not in data.batch:
            raise KeyError("MARFT PPO requires token_level_rewards and response_mask.")

        response_mask = data.batch["response_mask"].bool()
        real_rows = response_mask.any(dim=-1)
        projected = torch.tensor(
            [float(value) for value in _batch_values(data.non_tensor_batch["marft_projected_return"])],
            dtype=data.batch["token_level_rewards"].dtype,
            device=data.batch["token_level_rewards"].device,
        )
        if projected.shape[0] != response_mask.shape[0]:
            raise ValueError("MARFT projected-return metadata must have one value per TQ row.")
        score_tensor = data.batch.get("token_level_scores", data.batch["token_level_rewards"])
        observed = (score_tensor * response_mask).sum(dim=-1)
        if bool(real_rows.any()) and not torch.allclose(observed[real_rows], projected[real_rows], atol=1e-6, rtol=0):
            raise ValueError(
                "MARFT token rewards differ from marft_projected_return. "
                "Disable additional reward shaping or fix the rollout credit projection."
            )
        _project_future_row_reward_adjustments(data, response_mask=response_mask, real_rows=real_rows)
        return fallback(
            data,
            batch_keys=batch_keys,
            adv_estimator=adv_estimator,
            gamma=gamma,
            lam=lam,
            num_repeat=num_repeat,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config,
        )

    def compute_extra_metrics(self, data: Any, metrics: dict[str, Any], stage: str) -> dict[str, Any]:
        del metrics
        if stage != "advantage":
            return {}
        real_rows = data.batch["response_mask"].bool().any(dim=-1).tolist()
        values = [
            float(value)
            for value, is_real in zip(_batch_values(data.non_tensor_batch["marft_step_reward"]), real_rows, strict=True)
            if is_real
        ]
        projected = [
            float(value)
            for value, is_real in zip(
                _batch_values(data.non_tensor_batch["marft_projected_return"]), real_rows, strict=True
            )
            if is_real
        ]
        if not values:
            return {}
        return {
            "trajweave/marft/step_reward_mean": sum(values) / len(values),
            "trajweave/marft/projected_return_mean": sum(projected) / len(projected),
        }


def apply_marft_ctde_patch(config: Any = None) -> None:
    from trajweave.backends.verl.extensions.common.runtime import apply_hook_aware_tq_runtime

    apply_hook_aware_tq_runtime(config)


def _estimator_name(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw).strip().lower().rsplit(".", 1)[-1]


def _batch_values(values: Any) -> list[Any]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    return list(values) if isinstance(values, list | tuple) else [values]


def _project_future_row_reward_adjustments(data: Any, *, response_mask: Any, real_rows: Any) -> None:
    """Carry post-score token rewards across MARFT's role-row boundaries."""

    import torch

    if "token_level_scores" not in data.batch:
        return
    rewards = data.batch["token_level_rewards"]
    scores = data.batch["token_level_scores"]
    row_adjustments = ((rewards - scores) * response_mask).sum(dim=-1)
    has_no_adjustment = torch.allclose(
        row_adjustments[real_rows],
        torch.zeros_like(row_adjustments[real_rows]),
    )
    if not bool(real_rows.any()) or bool(has_no_adjustment):
        return

    trajectory_ids = _batch_values(data.non_tensor_batch["traj_uid"])
    turn_ids = _batch_values(data.non_tensor_batch["turn_id"])
    if len(trajectory_ids) != response_mask.shape[0] or len(turn_ids) != response_mask.shape[0]:
        raise ValueError("MARFT trajectory and turn metadata must have one value per TQ row.")

    grouped_rows: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row, is_real in enumerate(real_rows.tolist()):
        if is_real:
            grouped_rows[str(trajectory_ids[row])].append((int(turn_ids[row]), row))

    projected_rewards = rewards.clone()
    for trajectory_id, rows in grouped_rows.items():
        rows.sort()
        ordered_turns = [turn_id for turn_id, _ in rows]
        if len(ordered_turns) != len(set(ordered_turns)):
            raise ValueError(f"MARFT trajectory {trajectory_id!r} contains duplicate turn_id values.")
        future_adjustment = rewards.new_zeros(())
        for _, row in reversed(rows):
            valid_tokens = torch.nonzero(response_mask[row], as_tuple=False).flatten()
            projected_rewards[row, valid_tokens[-1]] += future_adjustment
            future_adjustment += row_adjustments[row]

    data.batch["token_level_rewards"] = projected_rewards
