from __future__ import annotations

from typing import Any

from trajweave.pipeline.context import RunContext
from trajweave.pipeline.launch import maybe_run_verl_launch
from trajweave.recipes.marft import (
    build_marft_launch_overrides,
    load_reward_callable,
    resolve_marft_settings,
    run_marft_smoke,
)


class MARFTRecipePlugin:
    name = "marft"

    def supports(self, context: RunContext) -> bool:
        return context.recipe_definition.family == "marft"

    def run(self, context: RunContext) -> dict[str, Any]:
        if context.mode == "smoke":
            return self._run_smoke(context)
        if context.mode in {"verl_train", "verl_plan"}:
            return self._run_verl(context)
        raise ValueError(f"Unsupported MARFT mode: {context.mode!r}.")

    def _run_smoke(self, context: RunContext) -> dict[str, Any]:
        settings = resolve_marft_settings(context.config)
        backend = context.config.get("backend", {}) or {}
        rollout = context.config.get("rollout", {}) or {}
        step_reward_fn = (
            load_reward_callable(settings["step_reward_fn"]) if settings["step_reward_fn"] is not None else None
        )
        per_agent_reward_fns = {
            role: load_reward_callable(path) for role, path in settings["per_agent_reward_fns"].items()
        }
        summary, result = run_marft_smoke(
            backend=str(backend.get("type", "rule")),
            device=str(backend.get("device", "cpu")),
            rollouts_per_task=int(rollout.get("rollouts_per_task", 2)),
            role_names=settings["role_names"],
            model_ids=settings["model_ids"],
            role_configs=settings["role_configs"],
            graph=settings["graph"],
            credit_strategy=settings["credit_strategy"],
            credit_discount=settings["credit_discount"],
            return_gamma=settings["return_gamma"],
            step_reward_fn=step_reward_fn,
            per_agent_reward_fns=per_agent_reward_fns,
        )
        context.tracker.log_rollout_result(result, source="marft_math_workflow")
        return {
            **context.base_output(),
            "trajectories": summary.trajectories,
            "samples": summary.samples,
            "success_rate": summary.success_rate,
            "agent_samples": summary.agent_samples,
            "dataproto_rows": summary.dataproto_rows,
            "dataproto_status": summary.dataproto_status,
            "marft": marft_summary(settings),
        }

    def _run_verl(self, context: RunContext) -> dict[str, Any]:
        settings = resolve_marft_settings(context.config, require_worker_assets=True)
        output = {**context.base_output(), "marft": marft_summary(settings)}
        maybe_run_verl_launch(
            context.config,
            output,
            overrides=build_marft_launch_overrides(context.config, config_path=context.config_path),
            default_enabled=True,
            default_module="trajweave.backends.verl.main_ppo",
            tracker=context.tracker,
            mode=context.mode,
        )
        return output


def marft_summary(settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": "math",
        "runtime_recipe": "marft_math_workflow",
        "coordination_protocol": "marft_static_dag",
        "communication_graph": {
            "nodes": [node.node_id for node in settings["graph"].nodes],
            "edges": [list(edge) for edge in settings["graph"].edges],
        },
        "roles": list(settings["role_names"]),
        "model_ids": list(settings["model_ids"]),
        "credit_allocator": "marft_ctde",
        "credit_strategy": settings["credit_strategy"],
        "critic_mode": settings["critic_mode"],
        "shared_policy": settings["shared_policy"],
        "shared_lora": settings["shared_lora"],
        "training_backend": ("verl_v1_multi_actor_wg" if settings["multi_actor_training"] else "verl_v1_shared_actor"),
    }
