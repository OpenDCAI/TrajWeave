from __future__ import annotations

from typing import Any

_NATIVE_PPO_V1_COMPUTE_ADVANTAGE: Any = None


def apply_maporl_full_ppo_patch(config: Any = None) -> None:
    """Install MAPoRL PPO compatibility around VERL.

    The algorithm-specific logic stays in TrajWeave. VERL receives only generic
    hook metadata through ``algorithm.extension_hooks_class`` plus the existing
    TransferQueue compatibility patch used by TrajWeave-managed agent loops.
    """

    from trajweave.backends.verl.extensions.drmas import apply_drmas_agent_wise_grpo_patch
    from verl.trainer.ppo.v1 import trainer_base

    _install_maporl_algorithm_config(config)
    _remember_native_compute_advantage(trainer_base.PPOTrainer)
    apply_drmas_agent_wise_grpo_patch(config)
    if _NATIVE_PPO_V1_COMPUTE_ADVANTAGE is None:
        raise RuntimeError("Cannot restore VERL's hook-aware PPO v1 advantage implementation for MAPoRL.")
    trainer_base.PPOTrainer._compute_advantage = _NATIVE_PPO_V1_COMPUTE_ADVANTAGE


def apply_maporl_single_model_patch(config: Any = None) -> None:
    """Backward compatible alias for older MAPoRL v1 configs."""

    apply_maporl_full_ppo_patch(config)


def _install_maporl_algorithm_config(config: Any) -> None:
    if config is None:
        return

    from trajweave.credit.maporl import resolve_maporl_score_rule_config

    algorithm = _config_get(config, "algorithm")
    if algorithm is None:
        return
    settings = resolve_maporl_score_rule_config(config)

    try:
        from omegaconf import DictConfig, OmegaConf, open_dict

        if isinstance(algorithm, DictConfig):
            with open_dict(algorithm):
                algorithm["maporl"] = OmegaConf.create(settings)
            return
    except ImportError:
        pass

    if isinstance(algorithm, dict):
        algorithm["maporl"] = settings
        return
    try:
        algorithm["maporl"] = settings
    except (KeyError, TypeError):
        algorithm.maporl = settings


def _remember_native_compute_advantage(ppo_trainer: type) -> None:
    global _NATIVE_PPO_V1_COMPUTE_ADVANTAGE

    current = ppo_trainer._compute_advantage
    if _NATIVE_PPO_V1_COMPUTE_ADVANTAGE is not None:
        return
    if current.__module__ == "verl.trainer.ppo.v1.trainer_base" and current.__name__ == "_compute_advantage":
        _NATIVE_PPO_V1_COMPUTE_ADVANTAGE = current


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    try:
        return config.get(key, default)
    except (AttributeError, TypeError):
        return getattr(config, key, default)
