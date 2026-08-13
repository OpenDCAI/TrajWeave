from __future__ import annotations

from typing import Any

from trajweave.pipeline.context import RunContext
from trajweave.pipeline.launch import maybe_run_verl_launch
from trajweave.recipes.c3 import build_c3_launch_overrides, resolve_c3_settings, run_c3_smoke


class C3RecipePlugin:
    name = "c3"

    def supports(self, context: RunContext) -> bool:
        return context.recipe_definition.family == "c3"

    def run(self, context: RunContext) -> dict[str, Any]:
        if context.mode == "smoke":
            return self._run_smoke(context)
        if context.mode in {"verl_train", "verl_plan"}:
            return self._run_verl(context)
        raise ValueError(f"Unsupported C3 mode: {context.mode!r}.")

    def _run_smoke(self, context: RunContext) -> dict[str, Any]:
        settings = resolve_c3_settings(context.config, mode=context.mode)
        backend = context.config.get("backend", {}) or {}
        backend_type = str(backend.get("type", "rule"))
        summary, result = run_c3_smoke(
            backend=backend_type,
            device=str(backend.get("device", "cpu")),
            fanout=settings["fanout"],
            variant=settings["credit_variant"],
            baseline_mode=settings["baseline_mode"],
            value_assisted_alpha=settings["value_assisted_alpha"],
            normalize=settings["normalize_advantages"],
        )
        context.tracker.log_rollout_result(result, source="c3_reasoner_actor_math")
        return {
            **context.base_output(),
            "trajectories": summary.trajectories,
            "samples": summary.samples,
            "success_rate": summary.success_rate,
            "c3": c3_summary(context.config, mode=context.mode, smoke_backend=backend_type),
        }

    def _run_verl(self, context: RunContext) -> dict[str, Any]:
        output = context.base_output()
        output["c3"] = c3_summary(context.config, mode=context.mode)
        maybe_run_verl_launch(
            context.config,
            output,
            overrides=build_c3_launch_overrides(
                context.config,
                config_path=context.config_path,
                mode=context.mode,
            ),
            default_enabled=True,
            default_module="trajweave.backends.verl.main_ppo",
            tracker=context.tracker,
            mode=context.mode,
        )
        return output


def c3_summary(
    config: dict[str, Any],
    *,
    mode: str | None = None,
    smoke_backend: str | None = None,
) -> dict[str, Any]:
    settings = resolve_c3_settings(config, mode=mode)
    if mode == "smoke":
        settings.pop("agent_loop_backend", None)
        settings.pop("hf_local_model_cache_size", None)
        settings["smoke_backend"] = str(smoke_backend or "rule")
    return {
        "task": "math",
        "runtime_recipe": "c3_reasoner_actor_math",
        "coordination_protocol": "contextual_counterfactual_prefix_replay",
        "communication_graph": "reasoner_to_actor",
        "trainable_agents": ["reasoner", "actor"],
        "credit_allocator": "c3_contextual_counterfactual",
        "trajectory_schema": "c3_prefix_tree_v1",
        "training_backend": "none" if mode == "smoke" else "verl_v1_multi_actor_q_critic",
        **settings,
    }
