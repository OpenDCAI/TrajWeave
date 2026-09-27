from __future__ import annotations

from typing import Any

from trajweave.pipeline.context import RunContext
from trajweave.pipeline.launch import maybe_run_verl_launch
from trajweave.recipes.wideseek_r1.broad_search import run_wideseek_r1_smoke
from trajweave.recipes.wideseek_r1.config import (
    build_wideseek_r1_launch_overrides,
    resolve_wideseek_r1_settings,
)
from trajweave.recipes.wideseek_r1.validation import validate_wideseek_training_result


class WideSeekR1RecipePlugin:
    name = "wideseek_r1"

    def supports(self, context: RunContext) -> bool:
        return context.recipe_definition.family == "wideseek_r1"

    def run(self, context: RunContext) -> dict[str, Any]:
        if context.mode == "smoke":
            return self._run_smoke(context)
        if context.mode in {"verl_train", "verl_plan"}:
            return self._run_verl(context)
        raise ValueError(f"Unsupported WideSeek-R1 mode: {context.mode!r}.")

    def _run_smoke(self, context: RunContext) -> dict[str, Any]:
        settings = resolve_wideseek_r1_settings(context.config)
        summary, result = run_wideseek_r1_smoke(
            rollouts_per_task=int((context.config.get("rollout", {}) or {}).get("rollouts_per_task", 2)),
            **{key: settings[key] for key in (
                "max_parallel_subagents",
                "lead_agent",
                "subagent_prefix",
                "shared_model_id",
                "format_reward",
                "search_reward",
                "length_limit",
                "max_length_limit",
                "length_penalty",
            )},
        )
        context.tracker.log_rollout_result(result, source="wideseek_r1_broad_search")
        output = context.base_output()
        output.update(
            {
                "trajectories": summary.trajectories,
                "samples": summary.samples,
                "success_rate": summary.success_rate,
                "lead_samples": summary.lead_samples,
                "subagent_samples": summary.subagent_samples,
                "active_subagents": summary.active_subagents,
                "dataproto_rows": summary.dataproto_rows,
                "dataproto_status": summary.dataproto_status,
                "wideseek_r1": wideseek_r1_summary(context.config),
            }
        )
        return output

    def _run_verl(self, context: RunContext) -> dict[str, Any]:
        output = context.base_output()
        output["wideseek_r1"] = wideseek_r1_summary(context.config)
        maybe_run_verl_launch(
            context.config,
            output,
            overrides=build_wideseek_r1_launch_overrides(context.config, config_path=context.config_path),
            default_enabled=True,
            default_module="trajweave.backends.verl.main_ppo",
            tracker=context.tracker,
            mode=context.mode,
        )
        validate_wideseek_training_result(output, run_dir=context.run_dir, mode=context.mode)
        return output


def wideseek_r1_summary(config: dict[str, Any]) -> dict[str, Any]:
    settings = resolve_wideseek_r1_settings(config)
    return {
        "task": "broad_information_seeking",
        "runtime_recipe": "wideseek_r1_broad_search",
        "coordination_protocol": "wideseek_r1_width_search",
        "lead_agent": settings["lead_agent"],
        "max_parallel_subagents": settings["max_parallel_subagents"],
        "shared_model_id": settings["shared_model_id"],
        "context_isolation": True,
        "credit_allocator": "wideseek_r1_multi_agent_grpo",
        "advantage_assignment": "trajectory_broadcast",
        "loss_reweighting": ["agent_level", "token_level"],
        "training_backend": "verl_v1_shared_actor",
    }
