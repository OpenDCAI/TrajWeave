from __future__ import annotations

from typing import Any


def apply_agentflow_planner_grpo_patch(config: Any = None) -> None:
    """Install VERL runtime compatibility needed by AgentFlow planner-only GRPO."""

    from trajweave.backends.verl.extensions.common.hooked_grpo import apply_hooked_grpo_patch

    apply_hooked_grpo_patch(config)
