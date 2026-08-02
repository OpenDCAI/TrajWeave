from __future__ import annotations

from typing import Any

from trajweave.pipeline.context import RunContext
from trajweave.recipes.marti_mars2 import run_single_mcts_smoke


class MARTIMARS2RecipePlugin:
    name = "marti_mars2"

    def supports(self, context: RunContext) -> bool:
        return context.recipe_definition.family == "marti_mars2"

    def run(self, context: RunContext) -> dict:
        if context.mode == "verl_train":
            return self._run_verl_train(context)
        return self._run_smoke(context)

    def _run_smoke(self, context: RunContext) -> dict:
        config = context.config
        rollout_cfg = config.get("rollout", {})
        mars2_cfg = config.get("marti_mars2", {})
        summary, result = run_single_mcts_smoke(
            max_num_nodes=int(mars2_cfg.get("max_num_nodes", rollout_cfg.get("max_num_nodes", 2))),
            rollouts_per_task=int(rollout_cfg.get("rollouts_per_task", 1)),
        )
        context.tracker.log_rollout_result(result, source="marti_mars2_single_mcts")
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
            "marti_mars2": marti_mars2_summary(config),
        }
        if context.recipe != context.recipe_definition.name:
            output["canonical_recipe"] = context.recipe_definition.name
        if context.prepared_assets:
            output["prepared_assets"] = context.prepared_assets
        return output

    def _run_verl_train(self, context: RunContext) -> dict:
        output: dict[str, Any] = context.base_output()
        output["marti_mars2"] = marti_mars2_summary(context.config)
        output["verl_launch"] = {
            "status": "not_implemented",
            "reason": "MARTI-MARS2 VERL/vLLM training launch requires Phase 3/4 backend hooks.",
        }
        return output


def marti_mars2_summary(config: dict[str, Any]) -> dict[str, Any]:
    mars2_cfg = config.get("marti_mars2", {})
    rollout_cfg = config.get("rollout", {})
    max_num_nodes = int(mars2_cfg.get("max_num_nodes", rollout_cfg.get("max_num_nodes", 2)))
    agent_ids = list(mars2_cfg.get("agent_ids", ["generator"]))
    model_ids = list(mars2_cfg.get("model_ids", ["shared"] * len(agent_ids)))
    return {
        "task": "code",
        "runtime_recipe": "marti_mars2_single_mcts",
        "agent_ids": agent_ids,
        "model_ids": model_ids,
        "max_num_nodes": max_num_nodes,
        "coordination_protocol": "mcts_selection_expansion_refinement_termination",
        "communication_graph": "single_agent_tree" if len(agent_ids) == 1 else "multi_agent_shared_tree",
        "aggregation": "best_path_or_mcts_eval",
        "credit_allocator": "tree_group_norm",
        "tree_identity_required": True,
        "tis_hook": bool(mars2_cfg.get("enable_vllm_is_correction", False)),
        "training_backend": "pending_verl_vllm_backend_hook",
    }
