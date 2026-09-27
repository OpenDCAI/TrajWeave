from __future__ import annotations

from typing import Any


def apply_hooked_grpo_patch(config: Any = None) -> None:
    """Install the shared VERL V1 hook bridge used by TrajWeave recipes."""
    from trajweave.backends.verl.extensions.drmas.agent_wise_grpo import apply_drmas_agent_wise_grpo_patch

    apply_drmas_agent_wise_grpo_patch(config)
