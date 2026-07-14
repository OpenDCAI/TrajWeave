from __future__ import annotations

from typing import Any


def apply_matpo_parent_broadcast_patch(config: Any = None) -> None:
    from trajweave.backends.verl.extensions.drmas import apply_drmas_agent_wise_grpo_patch

    apply_drmas_agent_wise_grpo_patch(config)
