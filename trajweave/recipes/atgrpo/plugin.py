from __future__ import annotations

from typing import Any

from trajweave.pipeline.context import RunContext
from trajweave.pipeline.launch import maybe_run_verl_launch
from trajweave.recipes.atgrpo import build_atgrpo_launch_overrides, resolve_atgrpo_settings, run_atgrpo_smoke


class ATGRPORecipePlugin:
    name = "atgrpo"

    def supports(self, context: RunContext) -> bool:
        return context.recipe_definition.family == "atgrpo"

    def run(self, context: RunContext) -> dict[str, Any]:
        if context.mode in {"verl_train", "verl_plan"}:
            return self._run_verl_train(context)
        if context.mode == "smoke":
            return self._run_smoke(context)
        raise ValueError(f"Unsupported AT-GRPO mode: {context.mode!r}.")

    def _run_smoke(self, context: RunContext) -> dict[str, Any]:
        settings = resolve_atgrpo_settings(context.config)
        backend = context.config.get("backend", {}) or {}
        rollout = context.config.get("rollout", {}) or {}
        summary, result = run_atgrpo_smoke(
            backend=str(backend.get("type", "rule")),
            device=str(backend.get("device", "cpu")),
            rollouts_per_task=int(rollout.get("rollouts_per_task", 4)),
            max_turns=settings["max_turns"],
            normalize_by_std=settings["normalize_by_std"],
            mixed_reward_enabled=settings["mixed_reward_enabled"],
            alpha=settings["mixed_reward_alpha"],
            verifier_local_reward=settings["mixed_reward_verifier_local_reward"],
        )
        context.tracker.log_rollout_result(result, source="atgrpo_solver_verifier_math")
        return {
            **context.base_output(),
            "trajectories": summary.trajectories,
            "samples": summary.samples,
            "success_rate": summary.success_rate,
            "atgrpo": atgrpo_summary(context.config),
        }

    def _run_verl_train(self, context: RunContext) -> dict[str, Any]:
        output: dict[str, Any] = context.base_output()
        output["atgrpo"] = atgrpo_summary(context.config)
        maybe_run_verl_launch(
            context.config,
            output,
            overrides=build_atgrpo_launch_overrides(context.config, config_path=context.config_path),
            default_enabled=True,
            default_module="trajweave.backends.verl.main_ppo",
            tracker=context.tracker,
            mode=context.mode,
        )
        return output


def atgrpo_summary(config: dict[str, Any]) -> dict[str, Any]:
    settings = resolve_atgrpo_settings(config)
    return {
        "task": "math",
        "runtime_recipe": "atgrpo_solver_verifier_math",
        "coordination_protocol": "solver_verifier_loop",
        "communication_graph": "solver_verifier_loop",
        "trainable_agents": ["solver", "verifier"],
        "frozen_agents": [],
        "credit_allocator": "atgrpo_agent_turn_wise_grpo",
        "credit_levels": ["agent_role", "turn"],
        "trajectory_schema": "agent_turn_wise_v1",
        "training_backend": "verl_v1_single_actor_wg",
        **settings,
    }
