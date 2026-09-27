from __future__ import annotations

from typing import Any


def apply_atgrpo_agent_turn_wise_grpo_patch(config: Any = None) -> None:
    """Install the TQ/actor-worker/weight-sync runtime needed for AT-GRPO.

    AT-GRPO reuses the same hook-aware runtime as GiGPO: the advantage
    computation itself is delegated to ``ATGRPOHooks`` via
    ``algorithm.extension_hooks_class`` (VERL's native hook dispatch) or via
    ``extension_hooks_for_config`` (TrajWeave's own dispatch for the TQ
    self-managed path), so no bespoke runtime patch is required beyond the
    shared TQ plumbing.
    """

    from trajweave.backends.verl.extensions.common.runtime import apply_hook_aware_tq_runtime

    apply_hook_aware_tq_runtime(config)
