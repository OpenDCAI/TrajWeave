from __future__ import annotations

from typing import Any

from trajweave.pipeline.context import RunContext
from trajweave.pipeline.launch import maybe_run_verl_launch
from trajweave.recipes.drmas_native import build_drmas_native_launch_overrides, recipe_spec


class DrMASNativeRecipePlugin:
    name = "drmas_native"

    def supports(self, context: RunContext) -> bool:
        return context.recipe_definition.family == "drmas_native"

    def run(self, context: RunContext) -> dict:
        if context.mode not in {"verl_train", "verl_plan"}:
            raise ValueError(f"Unsupported DrMAS native mode: {context.mode!r}.")
        output: dict[str, Any] = context.base_output()
        output["drmas_native"] = self._summary(context)
        overrides = build_drmas_native_launch_overrides(context.config, config_path=context.config_path)
        maybe_run_verl_launch(
            context.config,
            output,
            overrides=overrides,
            default_enabled=True,
            default_module="trajweave.backends.verl.main_ppo",
            tracker=context.tracker,
            mode=context.mode,
        )
        return output

    def _summary(self, context: RunContext) -> dict[str, Any]:
        spec = recipe_spec(context.recipe)
        native_cfg = context.config.get("drmas_native", {})
        agent_ids = list(native_cfg.get("agent_ids", spec.agent_ids))
        model_ids = list(native_cfg.get("model_ids", native_cfg.get("models", ["default"] * len(agent_ids))))
        return {
            "task": spec.task,
            "runtime_recipe": spec.runtime_recipe,
            "agent_ids": agent_ids,
            "model_ids": model_ids,
            "model_sharing": True,
            "training_backend": "verl_v1_single_actor_wg",
            "orchestra_type": spec.orchestra_type,
            "coordination_protocol": spec.coordination_protocol,
            "group_by_agent_id": True,
        }
