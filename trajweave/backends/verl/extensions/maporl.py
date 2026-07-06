from __future__ import annotations

from typing import Any


def apply_maporl_single_model_patch(config: Any = None) -> None:
    """Install MAPoRL v1 single-model VERL compatibility.

    MAPoRL v1 currently uses the same agent-wise GRPO grouping machinery as
    DrMAS. The MAPoRL-specific behavior lives in TrajWeave orchestration and
    credit shaping; this adapter keeps the extension name separate so future
    MAPoRL value-head/adapter routing can evolve without changing user configs.
    """

    from trajweave.backends.verl.extensions.drmas import apply_drmas_agent_wise_grpo_patch

    apply_drmas_agent_wise_grpo_patch(config)
