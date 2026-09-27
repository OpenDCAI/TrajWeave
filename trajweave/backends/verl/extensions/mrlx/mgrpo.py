from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from trajweave.backends.verl.extensions.common.hooks import AgentWiseGRPOHooks


@dataclass(frozen=True)
class MrlXMGRPOHooks(AgentWiseGRPOHooks):
    """Role-local GRPO grouping plus MrlX scheduling metadata."""

    name: str = "mrlx_mgrpo"

    def compute_grpo_outcome_advantage(
        self,
        *,
        token_level_rewards: Any,
        response_mask: Any,
        index: Any,
        traj_index: Any | None = None,
        epsilon: float = 1e-6,
        norm_adv_by_std_in_grpo: bool = True,
        group_by_agent_id: bool = False,
    ) -> tuple[Any, Any]:
        # One MrlX agent trajectory can contain several assistant turns. Collapse
        # those turns before computing the role-local baseline, matching the
        # upstream loss-masked multi-turn sample at the baseline level. The row-wise
        # loss layout remains an explicit approximation of that upstream sample.
        del group_by_agent_id
        return super().compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            traj_index=traj_index,
            epsilon=epsilon,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            group_by_agent_id=False,
        )

    def batch_schema_fields(self, stage: str, config: Any = None) -> tuple[str, ...]:
        if stage != "advantage":
            return ()
        return (
            "agent_id",
            "traj_uid",
            "worker_group",
            "mrlx_turn_role",
            "mrlx_training_mode",
            "mrlx_policy_lag",
            "mrlx_format_valid",
        )

    def tq_select_fields(
        self,
        stage: str,
        default_fields: tuple[str, ...] | None = None,
        config: Any = None,
    ) -> tuple[str, ...]:
        fields = list(super().tq_select_fields(stage, default_fields=default_fields, config=config))
        fields.extend(self.batch_schema_fields(stage, config=config))
        return tuple(dict.fromkeys(fields))


def apply_mrlx_mgrpo_patch(config: Any = None) -> None:
    from trajweave.backends.verl.extensions.common.runtime import apply_hook_aware_tq_runtime

    apply_hook_aware_tq_runtime(config)
