from __future__ import annotations

from pathlib import Path
from typing import Any

TRAJWEAVE_AGENT_LOOP_MANAGER_FQN = "trajweave.backends.verl.agent_loop.TrajWeaveAgentLoopManager"
GIGPO_HOOKS_FQN = "trajweave.backends.verl.extensions.common.hooks.GiGPOHooks"


def resolve_gigpo_settings(config: dict[str, Any]) -> dict[str, Any]:
    gigpo = config.get("gigpo", {}) or {}
    team = config.get("team", {}) or {}
    settings = {
        "agent_loop_backend": str(gigpo.get("agent_loop_backend", "hf_local_tq")),
        "max_steps": int(gigpo.get("max_steps", team.get("max_turns", 2))),
        "gamma": float(gigpo.get("gamma", 0.95)),
        "step_advantage_weight": float(gigpo.get("step_advantage_weight", 1.0)),
        "mode": str(gigpo.get("mode", "mean_std_norm")),
        "enable_similarity": bool(gigpo.get("enable_similarity", False)),
        "similarity_threshold": float(gigpo.get("similarity_threshold", 0.95)),
    }
    if settings["agent_loop_backend"] == "verl_tq":
        raise ValueError("GiGPO requires synthetic_tq or hf_local_tq so step transition metadata is preserved.")
    if settings["max_steps"] < 1:
        raise ValueError("gigpo.max_steps must be at least 1.")
    if not 0.0 <= settings["gamma"] <= 1.0:
        raise ValueError("gigpo.gamma must be in [0, 1].")
    if settings["step_advantage_weight"] < 0.0:
        raise ValueError("gigpo.step_advantage_weight must be non-negative.")
    if settings["mode"] not in {"mean_norm", "mean_std_norm"}:
        raise ValueError("gigpo.mode must be mean_norm or mean_std_norm.")
    if settings["enable_similarity"] and not 0.0 < settings["similarity_threshold"] < 1.0:
        raise ValueError("gigpo.similarity_threshold must be in (0, 1) when similarity is enabled.")
    return settings


def build_gigpo_launch_overrides(config: dict[str, Any], *, config_path: str | None) -> tuple[str, ...]:
    settings = resolve_gigpo_settings(config)
    source_config = config_path or str(Path.cwd())
    required = (
        "algorithm.adv_estimator=grpo",
        "++algorithm.group_by_agent_id=false",
        f"algorithm.gamma={settings['gamma']}",
        f"++algorithm.gigpo.step_advantage_weight={settings['step_advantage_weight']}",
        f"++algorithm.gigpo.mode={settings['mode']}",
        f"++algorithm.gigpo.enable_similarity={str(settings['enable_similarity']).lower()}",
        f"++algorithm.gigpo.similarity_threshold={settings['similarity_threshold']}",
        f"++algorithm.extension_hooks_class={GIGPO_HOOKS_FQN}",
        "+agent.agent_ids=[\"Solver Agent\",\"Verifier Agent\"]",
        "+agent.model_ids=[\"shared\",\"shared\"]",
        "+agent.model_sharing=true",
        "+agent.orchestra_type=gigpo_solver_verifier",
        f"+agent.orchestra.gigpo.max_steps={settings['max_steps']}",
        "+trajweave.recipe=gigpo_solver_verifier_math",
        f"+trajweave.config={_quote(source_config)}",
        "+trajweave.coordination_protocol=solver_frozen_verifier_loop",
        "+trajweave.trajectory_schema=step_transition_v1",
        "+trajweave.credit_allocator=gigpo_hierarchical_grpo",
        "+trajweave.verl_extensions=[trajweave_gigpo_hierarchical_grpo]",
        f"+trajweave.agent_loop_backend={settings['agent_loop_backend']}",
        "+trajweave.turn_padding_multiple=2",
        f"+actor_rollout_ref.rollout.agent.agent_loop_manager_class={TRAJWEAVE_AGENT_LOOP_MANAGER_FQN}",
    )
    overrides = tuple(str(item) for item in config.get("verl", {}).get("overrides", []))
    for item in required:
        overrides = _set_override(overrides, item)
    return overrides


def _set_override(overrides: tuple[str, ...], replacement: str) -> tuple[str, ...]:
    replacement_key = replacement.split("=", 1)[0].lstrip("+")
    filtered = tuple(item for item in overrides if item.split("=", 1)[0].lstrip("+") != replacement_key)
    return (*filtered, replacement)


def _quote(value: str) -> str:
    escaped = str(value).replace('"', '\\"')
    return f'"{escaped}"'
