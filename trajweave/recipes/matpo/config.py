from __future__ import annotations

from pathlib import Path
from typing import Any

TRAJWEAVE_AGENT_LOOP_MANAGER_FQN = "trajweave.backends.verl.agent_loop.TrajWeaveAgentLoopManager"


def build_matpo_launch_overrides(config: dict[str, Any], *, config_path: str | None) -> tuple[str, ...]:
    matpo_cfg = config.get("matpo", {})
    team_cfg = config.get("team", {})
    verl_cfg = config.get("verl", {})
    agent_loop_backend = str(matpo_cfg.get("agent_loop_backend", "hf_local_tq"))
    if agent_loop_backend == "verl_tq":
        raise ValueError("MATPO parent-broadcast training requires synthetic_tq or hf_local_tq, not verl_tq.")
    max_turns = int(matpo_cfg.get("max_turns", team_cfg.get("max_turns", 3)))
    worker_agent = str(matpo_cfg.get("worker_agent", "browsing_agent"))
    tool_name = str(matpo_cfg.get("tool_name", "search_and_browse"))
    source_config = config_path or str(Path.cwd())

    required = [
        "algorithm.adv_estimator=grpo",
        "++algorithm.group_by_agent_id=false",
        "++algorithm.extension_hooks_class=trajweave.backends.verl.extensions.common.hooks.MATPOParentBroadcastHooks",
        "+agent.orchestra_type=matpo",
        f"+agent.orchestra.matpo.max_turns={max_turns}",
        f"+agent.orchestra.matpo.worker_agent={_quote(worker_agent)}",
        f"+agent.orchestra.matpo.tool_name={_quote(tool_name)}",
        "+trajweave.recipe=matpo_browse",
        f"+trajweave.config={source_config}",
        "+trajweave.coordination_protocol=planner_worker_agent_tool",
        "+trajweave.trajectory_schema=matpo_parent_child_turn_v1",
        "+trajweave.credit_allocator=matpo_parent_broadcast_grpo",
        "+trajweave.verl_extensions=[trajweave_matpo_parent_broadcast]",
        f"+trajweave.agent_loop_backend={agent_loop_backend}",
        f"+actor_rollout_ref.rollout.agent.agent_loop_manager_class={TRAJWEAVE_AGENT_LOOP_MANAGER_FQN}",
    ]
    return tuple(str(item) for item in verl_cfg.get("overrides", [])) + tuple(required)


def _quote(value: str) -> str:
    escaped = str(value).replace('"', '\\"')
    return f'"{escaped}"'
