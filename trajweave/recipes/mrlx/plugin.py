from __future__ import annotations

from typing import Any

from trajweave.pipeline.context import RunContext
from trajweave.pipeline.launch import maybe_run_verl_launch
from trajweave.recipes.mrlx.config import build_mrlx_launch_overrides, resolve_mrlx_settings
from trajweave.recipes.mrlx.research_qa import run_mrlx_smoke


class MrlXRecipePlugin:
    name = "mrlx"

    def supports(self, context: RunContext) -> bool:
        return context.recipe_definition.family == "mrlx"

    def run(self, context: RunContext) -> dict[str, Any]:
        if context.mode == "smoke":
            return self._run_smoke(context)
        if context.mode in {"verl_train", "verl_plan"}:
            return self._run_verl(context)
        raise ValueError(f"Unsupported MrlX mode: {context.mode!r}.")

    def _run_smoke(self, context: RunContext) -> dict[str, Any]:
        settings = resolve_mrlx_settings(context.config)
        rollout = context.config.get("rollout", {}) or {}
        summary, result = run_mrlx_smoke(
            rollouts_per_task=int(rollout.get("rollouts_per_task", 2)),
            explorer_agent=settings["explorer_agent"],
            adapter_agent=settings["adapter_agent"],
            explorer_model_id=settings["explorer_model_id"],
            adapter_model_id=settings["adapter_model_id"],
            tool_name=settings["tool_name"],
            research_rounds=settings["research_rounds"],
            explorer_format_bonus=settings["explorer_format_bonus"],
            adapter_format_bonus=settings["adapter_format_bonus"],
        )
        context.tracker.log_rollout_result(result, source="mrlx_research_qa")
        output = context.base_output()
        output.update(
            {
                "trajectories": summary.trajectories,
                "samples": summary.samples,
                "success_rate": summary.success_rate,
                "explorer_samples": summary.explorer_samples,
                "adapter_samples": summary.adapter_samples,
                "dataproto_rows": summary.dataproto_rows,
                "dataproto_status": summary.dataproto_status,
                "mrlx": mrlx_summary(context.config),
            }
        )
        return output

    def _run_verl(self, context: RunContext) -> dict[str, Any]:
        output = context.base_output()
        output["mrlx"] = mrlx_summary(context.config)
        maybe_run_verl_launch(
            context.config,
            output,
            overrides=build_mrlx_launch_overrides(context.config, config_path=context.config_path),
            default_enabled=True,
            default_module="trajweave.backends.verl.main_ppo",
            tracker=context.tracker,
            mode=context.mode,
        )
        return output


def mrlx_summary(config: dict[str, Any]) -> dict[str, Any]:
    settings = resolve_mrlx_settings(config)
    return {
        "runtime_recipe": "mrlx_research_qa",
        "coordination_protocol": "mrlx_async_research",
        "explorer_agent": settings["explorer_agent"],
        "adapter_agent": settings["adapter_agent"],
        "explorer_model_id": settings["explorer_model_id"],
        "adapter_model_id": settings["adapter_model_id"],
        "worker_groups": settings["worker_groups"],
        "credit_allocator": "mrlx_mgrpo",
        "explorer_update": "on_policy",
        "adapter_update": "off_policy_one_step_lag",
        "adapter_delay_steps": settings["adapter_delay_steps"],
        "explorer_format_bonus": settings["explorer_format_bonus"],
        "adapter_format_bonus": settings["adapter_format_bonus"],
        "training_backend": "verl_v1_mrlx_async_multi_actor",
    }
