from __future__ import annotations

from typing import Any

from trajweave.pipeline.context import RunContext
from trajweave.pipeline.launch import maybe_run_verl_launch
from trajweave.recipes.gigpo import build_gigpo_launch_overrides, resolve_gigpo_settings, run_gigpo_smoke


class GiGPORecipePlugin:
    name = "gigpo"

    def supports(self, context: RunContext) -> bool:
        return context.recipe_definition.family == "gigpo"

    def run(self, context: RunContext) -> dict[str, Any]:
        if context.mode in {"verl_train", "verl_plan"}:
            return self._run_verl_train(context)
        if context.mode == "smoke":
            return self._run_smoke(context)
        raise ValueError(f"Unsupported GiGPO mode: {context.mode!r}.")

    def _run_smoke(self, context: RunContext) -> dict[str, Any]:
        settings = resolve_gigpo_settings(context.config)
        backend = context.config.get("backend", {}) or {}
        rollout = context.config.get("rollout", {}) or {}
        summary, result = run_gigpo_smoke(
            backend=str(backend.get("type", "rule")),
            device=str(backend.get("device", "cpu")),
            rollouts_per_task=int(rollout.get("rollouts_per_task", 2)),
            max_steps=settings["max_steps"],
            gamma=settings["gamma"],
            step_advantage_weight=settings["step_advantage_weight"],
            mode=settings["mode"],
            enable_similarity=settings["enable_similarity"],
            similarity_threshold=settings["similarity_threshold"],
        )
        context.tracker.log_rollout_result(result, source="gigpo_solver_verifier_math")
        return {
            **context.base_output(),
            "trajectories": summary.trajectories,
            "samples": summary.samples,
            "success_rate": summary.success_rate,
            "gigpo": gigpo_summary(context.config),
        }

    def _run_verl_train(self, context: RunContext) -> dict[str, Any]:
        output: dict[str, Any] = context.base_output()
        output["gigpo"] = gigpo_summary(context.config)
        maybe_run_verl_launch(
            context.config,
            output,
            overrides=build_gigpo_launch_overrides(context.config, config_path=context.config_path),
            default_enabled=True,
            default_module="trajweave.backends.verl.main_ppo",
            tracker=context.tracker,
            mode=context.mode,
        )
        return output


def gigpo_summary(config: dict[str, Any]) -> dict[str, Any]:
    settings = resolve_gigpo_settings(config)
    return {
        "task": "math",
        "runtime_recipe": "gigpo_solver_verifier_math",
        "coordination_protocol": "solver_frozen_verifier_loop",
        "communication_graph": "solver_verifier_loop",
        "trainable_agents": ["solver"],
        "frozen_agents": ["verifier"],
        "credit_allocator": "gigpo_hierarchical_grpo",
        "credit_levels": ["episode", "step"],
        "trajectory_schema": "step_transition_v1",
        "training_backend": "verl_v1_single_actor_wg",
        **settings,
    }
