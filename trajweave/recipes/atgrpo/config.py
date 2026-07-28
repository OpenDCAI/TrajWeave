from __future__ import annotations

from pathlib import Path
from typing import Any

TRAJWEAVE_AGENT_LOOP_MANAGER_FQN = "trajweave.backends.verl.agent_loop.TrajWeaveAgentLoopManager"
ATGRPO_HOOKS_FQN = "trajweave.backends.verl.extensions.common.hooks.ATGRPOHooks"


def resolve_atgrpo_settings(config: dict[str, Any]) -> dict[str, Any]:
    atgrpo = config.get("atgrpo", {}) or {}
    team = config.get("team", {}) or {}
    mixed_reward = atgrpo.get("mixed_reward", {}) or {}
    settings = {
        "agent_loop_backend": str(atgrpo.get("agent_loop_backend", "hf_local_tq")),
        "max_turns": int(atgrpo.get("max_turns", team.get("max_turns", 3))),
        "normalize_by_std": bool(atgrpo.get("normalize_by_std", True)),
        "mixed_reward_enabled": bool(mixed_reward.get("enabled", False)),
        "mixed_reward_alpha": float(mixed_reward.get("alpha", 1.0)),
        "mixed_reward_verifier_local_reward": float(mixed_reward.get("verifier_local_reward", 1.0)),
    }
    if settings["agent_loop_backend"] == "verl_tq":
        raise ValueError("AT-GRPO requires synthetic_tq or hf_local_tq so turn/agent metadata is preserved.")
    if settings["max_turns"] < 1:
        raise ValueError("atgrpo.max_turns must be at least 1.")
    return settings


def build_atgrpo_launch_overrides(config: dict[str, Any], *, config_path: str | None) -> tuple[str, ...]:
    settings = resolve_atgrpo_settings(config)
    source_config = config_path or str(Path.cwd())
    required = (
        "algorithm.adv_estimator=grpo",
        f"algorithm.norm_adv_by_std_in_grpo={'true' if settings['normalize_by_std'] else 'false'}",
        "++algorithm.group_by_agent_id=true",
        f"++algorithm.extension_hooks_class={ATGRPO_HOOKS_FQN}",
        "+agent.agent_ids=[\"Solver Agent\",\"Verifier Agent\"]",
        "+agent.model_ids=[\"shared\",\"shared\"]",
        "+agent.model_sharing=true",
        "+agent.orchestra_type=atgrpo_solver_verifier",
        f"+agent.orchestra.atgrpo.max_turns={settings['max_turns']}",
        "+trajweave.recipe=atgrpo_solver_verifier_math",
        f"+trajweave.config={_quote(source_config)}",
        "+trajweave.coordination_protocol=solver_verifier_loop",
        "+trajweave.trajectory_schema=agent_turn_wise_v1",
        "+trajweave.credit_allocator=atgrpo_agent_turn_wise_grpo",
        "+trajweave.verl_extensions=[trajweave_atgrpo_agent_turn_wise_grpo]",
        f"+trajweave.agent_loop_backend={settings['agent_loop_backend']}",
        "+trajweave.turn_padding_multiple=2",
        f"+actor_rollout_ref.rollout.agent.agent_loop_manager_class={TRAJWEAVE_AGENT_LOOP_MANAGER_FQN}",
        f"+agent.orchestra.atgrpo.mixed_reward.enabled={'true' if settings['mixed_reward_enabled'] else 'false'}",
    )
    overrides = tuple(str(item) for item in config.get("verl", {}).get("overrides", []))
    for item in required:
        overrides = _set_override(overrides, item)
    if settings["mixed_reward_enabled"]:
        for item in (
            f"+agent.orchestra.atgrpo.mixed_reward.alpha={settings['mixed_reward_alpha']}",
            "+agent.orchestra.atgrpo.mixed_reward.verifier_local_reward="
            f"{settings['mixed_reward_verifier_local_reward']}",
        ):
            overrides = _set_override(overrides, item)
    return overrides


def _set_override(overrides: tuple[str, ...], replacement: str) -> tuple[str, ...]:
    replacement_key = replacement.split("=", 1)[0].lstrip("+")
    filtered = tuple(item for item in overrides if item.split("=", 1)[0].lstrip("+") != replacement_key)
    return (*filtered, replacement)


def _quote(value: str) -> str:
    escaped = str(value).replace('"', '\\"')
    return f'"{escaped}"'
