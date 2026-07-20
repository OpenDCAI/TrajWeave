from __future__ import annotations

from typing import Any

from trajweave.pipeline.context import RunContext
from trajweave.pipeline.launch import maybe_run_verl_launch
from trajweave.recipes.comas.config import build_comas_launch_overrides, resolve_comas_topology
from trajweave.recipes.comas.peer_review_math import run_comas_math_smoke


class CoMASRecipePlugin:
    name = "comas"

    def supports(self, context: RunContext) -> bool:
        return context.recipe_definition.family == "comas"

    def run(self, context: RunContext) -> dict[str, Any]:
        if context.mode == "verl_train":
            return self._run_verl_train(context)
        if context.mode == "smoke":
            return self._run_smoke(context)
        raise ValueError(f"Unsupported CoMAS mode: {context.mode!r}.")

    def _run_smoke(self, context: RunContext) -> dict[str, Any]:
        config = context.config
        comas_cfg = config.get("comas", {}) or {}
        topology = resolve_comas_topology(config)
        summary, result = run_comas_math_smoke(
            agent_ids=topology["agent_ids"],
            model_ids=topology["model_ids"],
            num_rounds=int(comas_cfg.get("num_rounds", 2)),
            num_references=int(comas_cfg.get("num_references", 2)),
            assignment_seed=int(comas_cfg.get("assignment_seed", 0)),
            rollouts_per_task=int(config.get("rollout", {}).get("rollouts_per_task", 1)),
        )
        context.tracker.log_rollout_result(result, source="comas_peer_review_math")
        output = context.base_output()
        output.update(
            {
                "comas": comas_summary(config),
                "trajectories": summary.trajectories,
                "samples": summary.samples,
                "success_rate": summary.success_rate,
                "dataproto_status": summary.dataproto_status,
                "dataproto_rows": summary.dataproto_rows,
            }
        )
        return output

    def _run_verl_train(self, context: RunContext) -> dict[str, Any]:
        output = context.base_output()
        output["comas"] = comas_summary(context.config)
        maybe_run_verl_launch(
            context.config,
            output,
            overrides=build_comas_launch_overrides(context.config, config_path=context.config_path),
            default_enabled=True,
            default_module="trajweave.backends.verl.main_ppo",
            tracker=context.tracker,
        )
        return output


def comas_summary(config: dict[str, Any]) -> dict[str, Any]:
    comas_cfg = config.get("comas", {}) or {}
    topology = resolve_comas_topology(
        config,
        require_worker_assets=str(config.get("mode", "")) == "verl_train",
    )
    return {
        "paper": "CoMAS: Co-Evolving Multi-Agent Systems via Interaction Rewards",
        "source_repository": "https://github.com/xxyQwQ/CoMAS",
        "source_commit": "0d98c97",
        "runtime_recipe": "comas_peer_review_math",
        "task": "math",
        "agent_ids": list(topology["agent_ids"]),
        "model_ids": list(topology["model_ids"]),
        "worker_groups": topology["worker_groups"],
        "shared_agents": topology["shared_agents"],
        "num_rounds": int(comas_cfg.get("num_rounds", 2)),
        "num_references": int(comas_cfg.get("num_references", 2)),
        "coordination_protocol": "solver_evaluator_scorer",
        "communication_graph": "peer_review",
        "credit_allocator": "comas_interaction_reward",
        "advantage_estimator": "per_worker_group_normalized_reinforce",
        "ground_truth_used_for_training": False,
        "training_backend": (
            "verl_v1_multi_actor_wg" if topology["multi_actor_training"] else "verl_v1_single_actor_wg"
        ),
        "multi_actor_validation": topology["multi_actor_validation"],
    }
