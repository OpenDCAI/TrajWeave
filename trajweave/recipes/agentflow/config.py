from __future__ import annotations

from pathlib import Path
from typing import Any

TRAJWEAVE_AGENT_LOOP_MANAGER_FQN = "trajweave.backends.verl.agent_loop.TrajWeaveAgentLoopManager"


def build_agentflow_launch_overrides(
    config: dict[str, Any],
    *,
    config_path: str | None,
) -> tuple[str, ...]:
    agentflow_cfg = config.get("agentflow", {})
    team_cfg = config.get("team", {})
    verl_cfg = config.get("verl", {})
    credit_cfg = config.get("credit", {}) or {}

    max_steps = int(agentflow_cfg.get("max_steps", team_cfg.get("max_turns", 3)))
    enabled_tools = tuple(agentflow_cfg.get("enabled_tools", ["base_generator", "python_stub"]))
    trainable_agent = str(agentflow_cfg.get("trainable_agent", "planner"))
    agent_loop_backend = str(agentflow_cfg.get("agent_loop_backend", "hf_local_tq"))
    if agent_loop_backend == "verl_tq":
        raise ValueError("AgentFlow planner-tool training requires synthetic_tq or hf_local_tq, not verl_tq.")
    source_config = config_path or str(Path.cwd())

    required = [
        "algorithm.adv_estimator=grpo",
        "++algorithm.group_by_agent_id=false",
        "++algorithm.extension_hooks_class=trajweave.backends.verl.extensions.common.hooks.AgentFlowPlannerGRPOHooks",
        "+agent.orchestra_type=agentflow",
        f"+agent.orchestra.agentflow.max_steps={max_steps}",
        f"+agent.orchestra.agentflow.trainable_agent={_quote(trainable_agent)}",
        f"+agent.orchestra.agentflow.enabled_tools={_hydra_list(enabled_tools)}",
        f"+agent.orchestra.agentflow.reward_scope={_quote(str(credit_cfg.get('reward_scope', 'final_outcome')))}",
        "+trajweave.recipe=agentflow_planner_tool",
        f"+trajweave.config={source_config}",
        "+trajweave.coordination_protocol=planner_executor_tool_verifier",
        "+trajweave.trajectory_schema=multi_agent_turn_v1",
        "+trajweave.credit_allocator=agentflow_planner_only_grpo",
        "+trajweave.verl_extensions=[trajweave_agentflow_planner_grpo]",
        f"+trajweave.agent_loop_backend={agent_loop_backend}",
        f"+actor_rollout_ref.rollout.agent.agent_loop_manager_class={TRAJWEAVE_AGENT_LOOP_MANAGER_FQN}",
    ]
    return tuple(str(item) for item in verl_cfg.get("overrides", [])) + tuple(required)


def _hydra_list(values: tuple[str, ...]) -> str:
    return "[" + ",".join(_quote(value) for value in values) + "]"


def _quote(value: str) -> str:
    escaped = str(value).replace('"', '\\"')
    return f'"{escaped}"'
