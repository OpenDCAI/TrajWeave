from __future__ import annotations

from typing import Any

from trajweave.pipeline.context import RunContext
from trajweave.pipeline.launch import maybe_run_verl_launch
from trajweave.recipes.agentflow import build_agentflow_launch_overrides, run_planner_tool_smoke


class AgentFlowRecipePlugin:
    name = "agentflow"

    def supports(self, context: RunContext) -> bool:
        return context.recipe_definition.family == "agentflow"

    def run(self, context: RunContext) -> dict:
        if context.mode == "verl_train":
            return self._run_verl_train(context)
        return self._run_smoke(context)

    def _run_smoke(self, context: RunContext) -> dict:
        config = context.config
        rollout_cfg = config.get("rollout", {})
        team_cfg = config.get("team", {})
        backend_cfg = config.get("backend", {})
        agentflow_cfg = config.get("agentflow", {})
        summary, result = run_planner_tool_smoke(
            backend=str(backend_cfg.get("type", "rule")),
            device=str(backend_cfg.get("device", "cpu")),
            rollouts_per_task=int(rollout_cfg.get("rollouts_per_task", 2)),
            max_steps=int(agentflow_cfg.get("max_steps", team_cfg.get("max_turns", int(team_cfg.get("max_turns", 2))))),
        )
        context.tracker.log_rollout_result(result, source="agentflow_planner_tool")
        output = {
            "run_id": context.run_id,
            "run_dir": str(context.run_dir),
            "config_path": context.config_path,
            "recipe": context.recipe,
            "mode": context.mode,
            "trajectories": summary.trajectories,
            "samples": summary.samples,
            "success_rate": summary.success_rate,
            "dataproto_status": summary.dataproto_status,
            "dataproto_rows": summary.dataproto_rows,
        }
        if context.recipe != context.recipe_definition.name:
            output["canonical_recipe"] = context.recipe_definition.name
        if context.prepared_assets:
            output["prepared_assets"] = context.prepared_assets
        return output

    def _run_verl_train(self, context: RunContext) -> dict:
        output: dict[str, Any] = context.base_output()
        output["agentflow"] = agentflow_summary(context.config)
        overrides = build_agentflow_launch_overrides(context.config, config_path=context.config_path)
        maybe_run_verl_launch(
            context.config,
            output,
            overrides=overrides,
            default_enabled=True,
            default_module="trajweave.backends.verl.main_ppo",
            tracker=context.tracker,
        )
        return output


def agentflow_summary(config: dict[str, Any]) -> dict[str, Any]:
    agentflow_cfg = config.get("agentflow", {})
    team_cfg = config.get("team", {})
    return {
        "task": "math",
        "runtime_recipe": "agentflow_planner_tool",
        "coordination_protocol": "planner_executor_tool_verifier",
        "communication_graph": "memory_blackboard",
        "aggregation": "verifier_stop_then_final_answer",
        "credit_allocator": "agentflow_planner_only_grpo",
        "trainable_agent": str(agentflow_cfg.get("trainable_agent", "planner")),
        "frozen_agents": ["executor", "verifier"],
        "enabled_tools": list(agentflow_cfg.get("enabled_tools", ["base_generator", "python_stub"])),
        "max_steps": int(agentflow_cfg.get("max_steps", team_cfg.get("max_turns", 3))),
        "training_backend": "verl_v1_single_actor_wg",
    }
