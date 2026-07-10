from __future__ import annotations

from typing import Any

_NATIVE_HOOK_AWARE_COMPUTE_ADVANTAGE: Any = None


def apply_hook_aware_tq_runtime(config: Any = None) -> None:
    """启用 TrajWeave TQ/权重同步能力，同时保留 VERL 的通用 hook 入口。"""

    from trajweave.backends.verl.extensions.drmas import apply_drmas_agent_wise_grpo_patch
    from verl.trainer.ppo.v1 import trainer_base

    native_compute_advantage = _remember_native_compute_advantage(trainer_base.PPOTrainer)
    apply_drmas_agent_wise_grpo_patch(config)
    trainer_base.PPOTrainer._compute_advantage = native_compute_advantage


def _remember_native_compute_advantage(ppo_trainer: type) -> Any:
    global _NATIVE_HOOK_AWARE_COMPUTE_ADVANTAGE

    if _NATIVE_HOOK_AWARE_COMPUTE_ADVANTAGE is not None:
        return _NATIVE_HOOK_AWARE_COMPUTE_ADVANTAGE
    current = ppo_trainer._compute_advantage
    if current.__module__ != "verl.trainer.ppo.v1.trainer_base" or current.__name__ != "_compute_advantage":
        raise RuntimeError("VERL hook-aware PPOTrainer._compute_advantage is not available before TrajWeave patching.")
    _NATIVE_HOOK_AWARE_COMPUTE_ADVANTAGE = current
    return current
