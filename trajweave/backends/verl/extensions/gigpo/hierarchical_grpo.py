from __future__ import annotations

from typing import Any


def apply_gigpo_hierarchical_grpo_patch(config: Any = None) -> None:
    """安装 GiGPO 所需的 TQ、Actor worker 和权重回流能力。"""

    from trajweave.backends.verl.extensions.common.runtime import apply_hook_aware_tq_runtime

    apply_hook_aware_tq_runtime(config)
