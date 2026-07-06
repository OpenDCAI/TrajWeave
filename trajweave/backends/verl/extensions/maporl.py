from __future__ import annotations

from typing import Any


def apply_maporl_full_ppo_patch(config: Any = None) -> None:
    """Install MAPoRL PPO compatibility around VERL.

    The algorithm-specific logic stays in TrajWeave. VERL receives only generic
    hook metadata through ``algorithm.extension_hooks_class`` plus the existing
    TransferQueue compatibility patch used by TrajWeave-managed agent loops.
    """

    from trajweave.backends.verl.extensions.drmas import apply_drmas_agent_wise_grpo_patch

    apply_drmas_agent_wise_grpo_patch(config)


def apply_maporl_single_model_patch(config: Any = None) -> None:
    """Backward compatible alias for older MAPoRL v1 configs."""

    apply_maporl_full_ppo_patch(config)
