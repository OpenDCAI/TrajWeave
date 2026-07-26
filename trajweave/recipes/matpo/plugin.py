from __future__ import annotations

from typing import Any

from trajweave.pipeline.context import RunContext
from trajweave.pipeline.launch import maybe_run_verl_launch
from trajweave.recipes.matpo.config import build_matpo_launch_overrides
from trajweave.recipes.matpo.smoke import run_smoke


class MATPORecipePlugin:
    name = "matpo"

    def supports(self, context: RunContext) -> bool:
        return context.recipe_definition.family == "matpo"

    def run(self, context: RunContext) -> dict:
        if context.mode == "verl_train":
            return self._run_verl_train(context)
        return self._run_smoke(context)

    def _run_smoke(self, context: RunContext) -> dict:
        config = context.config
        rollout_cfg = config.get("rollout", {})
        team_cfg = config.get("team", {})
        matpo_cfg = config.get("matpo", {})
        summary, result = run_smoke(
            rollouts_per_task=int(rollout_cfg.get("rollouts_per_task", 2)),
            max_turns=int(team_cfg.get("max_turns", matpo_cfg.get("max_turns", 3))),
            planner_agent=str(matpo_cfg.get("planner_agent", "planner")),
            worker_agent=str(matpo_cfg.get("worker_agent", "browsing_agent")),
            tool_name=str(matpo_cfg.get("tool_name", "search_and_browse")),
        )
        context.tracker.log_rollout_result(result, source="matpo_browse")
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

    def _run_verl_train(self, context: RunContext) -> dict[str, Any]:
        output = context.base_output()
        output["matpo"] = matpo_summary(context.config)
        overrides = build_matpo_launch_overrides(context.config, config_path=context.config_path)
        maybe_run_verl_launch(
            context.config,
            output,
            overrides=overrides,
            default_enabled=True,
            default_module="trajweave.backends.verl.main_ppo",
            tracker=context.tracker,
        )
        return output


def matpo_summary(config: dict[str, Any]) -> dict[str, Any]:
    matpo_cfg = config.get("matpo", {})
    return {
        "task": "browse_qa",
        "runtime_recipe": "matpo_browse",
        "coordination_protocol": "planner_worker_agent_tool",
        "planner_agent": str(matpo_cfg.get("planner_agent", "planner")),
        "worker_agent": str(matpo_cfg.get("worker_agent", "browsing_agent")),
        "tool_name": str(matpo_cfg.get("tool_name", "search_and_browse")),
        "credit_allocator": "matpo_parent_broadcast_grpo",
        "training_backend": "verl_v1_single_actor_wg",
    }
