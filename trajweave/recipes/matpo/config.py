from __future__ import annotations

from math import isclose, isfinite
from pathlib import Path
from typing import Any

from trajweave.pipeline.config import hydra_path

TRAJWEAVE_AGENT_LOOP_MANAGER_FQN = "trajweave.backends.verl.agent_loop.TrajWeaveAgentLoopManager"
MATPO_HOOKS_FQN = "trajweave.backends.verl.extensions.matpo.parent_broadcast.MATPOParentBroadcastHooks"
DEFAULT_ACCURACY_REWARD_WEIGHT = 0.9
DEFAULT_TOOL_FORMAT_REWARD_WEIGHT = 0.1
_OLD_REWARD_FIELD = "tool_format_reward_scale"
_ALLOWED_TOOL_NAMES = {"search", "browse", "web_search", "search_and_browse"}


def matpo_reward_settings(config: dict[str, Any]) -> tuple[float, float]:
    matpo_cfg = _matpo_config(config)
    if _OLD_REWARD_FIELD in matpo_cfg:
        raise ValueError(
            "MATPO tool_format_reward_scale is no longer supported; use accuracy_reward_weight "
            "and tool_format_reward_weight."
        )
    accuracy_weight = _reward_weight(
        matpo_cfg.get("accuracy_reward_weight", DEFAULT_ACCURACY_REWARD_WEIGHT),
        field="accuracy_reward_weight",
    )
    tool_format_weight = _reward_weight(
        matpo_cfg.get("tool_format_reward_weight", DEFAULT_TOOL_FORMAT_REWARD_WEIGHT),
        field="tool_format_reward_weight",
    )
    if not isclose(accuracy_weight + tool_format_weight, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("MATPO accuracy_reward_weight and tool_format_reward_weight must sum to 1.0.")
    return accuracy_weight, tool_format_weight


def build_matpo_launch_overrides(config: dict[str, Any], *, config_path: str | None) -> tuple[str, ...]:
    matpo_cfg = _matpo_config(config)
    team_cfg = config.get("team", {}) or {}
    verl_cfg = config.get("verl", {}) or {}
    if not isinstance(team_cfg, dict) or not isinstance(verl_cfg, dict):
        raise ValueError("MATPO team and verl settings must be mappings.")

    agent_loop_backend = str(matpo_cfg.get("agent_loop_backend", "hf_local_tq")).strip()
    if agent_loop_backend == "verl_tq":
        raise ValueError("MATPO parent-broadcast training requires synthetic_tq or hf_local_tq, not verl_tq.")
    if agent_loop_backend not in {"synthetic_tq", "hf_local_tq"}:
        raise ValueError("MATPO agent_loop_backend must be synthetic_tq or hf_local_tq.")

    max_turns = int(matpo_cfg.get("max_turns", team_cfg.get("max_turns", 3)))
    if max_turns < 1:
        raise ValueError("MATPO max_turns must be at least 1.")
    planner_agent = _nonempty_setting(matpo_cfg, "planner_agent", "planner")
    worker_agent = _nonempty_setting(matpo_cfg, "worker_agent", "browsing_agent")
    tool_name = _nonempty_setting(matpo_cfg, "tool_name", "search_and_browse")
    if tool_name not in _ALLOWED_TOOL_NAMES:
        allowed = ", ".join(sorted(_ALLOWED_TOOL_NAMES))
        raise ValueError(f"MATPO tool_name must be one of: {allowed}.")
    if planner_agent == worker_agent:
        raise ValueError("MATPO planner_agent and worker_agent must be different.")
    accuracy_weight, tool_format_weight = matpo_reward_settings(config)
    source_config = config_path or str(Path.cwd())

    required = [
        "trainer.use_v1=true",
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=false",
        "critic.enable=false",
        "actor_rollout_ref.actor.policy_loss.loss_mode=vanilla",
        "actor_rollout_ref.actor.loss_agg_mode=token-mean",
        "actor_rollout_ref.actor.ppo_epochs=1",
        "actor_rollout_ref.actor.use_kl_loss=false",
        "++algorithm.group_by_agent_id=false",
        f"++algorithm.extension_hooks_class={MATPO_HOOKS_FQN}",
        f"+agent.agent_ids=[{_quote(planner_agent)},{_quote(worker_agent)}]",
        '+agent.model_ids=["shared","shared"]',
        "+agent.model_sharing=true",
        "+agent.orchestra_type=matpo",
        f"+agent.orchestra.matpo.max_turns={max_turns}",
        f"+agent.orchestra.matpo.planner_agent={_quote(planner_agent)}",
        f"+agent.orchestra.matpo.worker_agent={_quote(worker_agent)}",
        f"+agent.orchestra.matpo.tool_name={_quote(tool_name)}",
        f"+agent.orchestra.matpo.accuracy_reward_weight={accuracy_weight}",
        f"+agent.orchestra.matpo.tool_format_reward_weight={tool_format_weight}",
        "+trajweave.recipe=matpo_browse",
        f"+trajweave.config={hydra_path(source_config)}",
        "+trajweave.coordination_protocol=planner_worker_agent_tool",
        "+trajweave.trajectory_schema=matpo_parent_child_turn_v1",
        "+trajweave.credit_allocator=matpo_parent_broadcast_grpo",
        "+trajweave.verl_extensions=[trajweave_matpo_parent_broadcast]",
        f"+trajweave.agent_loop_backend={agent_loop_backend}",
        "+trajweave.turn_padding_multiple=2",
        f"+actor_rollout_ref.rollout.agent.agent_loop_manager_class={TRAJWEAVE_AGENT_LOOP_MANAGER_FQN}",
    ]

    configured_overrides = tuple(str(item) for item in verl_cfg.get("overrides", []))
    for override in configured_overrides:
        if _override_key(override).endswith(_OLD_REWARD_FIELD):
            raise ValueError(
                "MATPO tool_format_reward_scale override is no longer supported; use the explicit reward weights."
            )
    required_keys = {_override_key(item) for item in required}
    retained_overrides = tuple(item for item in configured_overrides if _override_key(item) not in required_keys)
    return retained_overrides + tuple(required)


def _matpo_config(config: dict[str, Any]) -> dict[str, Any]:
    matpo_cfg = config.get("matpo", {}) or {}
    if not isinstance(matpo_cfg, dict):
        raise ValueError("MATPO settings must be a mapping.")
    return matpo_cfg


def _reward_weight(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"MATPO {field} must be a finite number between 0 and 1.")
    try:
        weight = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"MATPO {field} must be a finite number between 0 and 1.") from exc
    if not isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError(f"MATPO {field} must be a finite number between 0 and 1.")
    return weight


def _nonempty_setting(config: dict[str, Any], field: str, default: str) -> str:
    value = str(config.get(field, default)).strip()
    if not value:
        raise ValueError(f"MATPO {field} must be non-empty.")
    return value


def _override_key(override: str) -> str:
    return override.split("=", 1)[0].lstrip("+")


def _quote(value: str) -> str:
    escaped = str(value).replace('"', '\\"')
    return f'"{escaped}"'
