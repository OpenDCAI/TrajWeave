from __future__ import annotations

from typing import Any

from trajweave.pipeline.context import RunContext
from trajweave.pipeline.launch import maybe_run_verl_launch
from trajweave.recipes.marti_mars2.stable import (
    build_stable_launch_overrides,
    run_stable_cpu_fixture,
    stable_acceptance,
)


class MARTIMARS2StableRecipePlugin:
    """Independent MARS² stability recipe (GSPO/TIS/overlong)."""

    name = "marti_mars2_stable"

    def supports(self, context: RunContext) -> bool:
        return context.recipe_definition.family == "marti_mars2_stable"

    def run(self, context: RunContext) -> dict[str, Any]:
        output = context.base_output()
        stable = context.config.get("stable", {}) or {}
        output["stable"] = {
            "loss": str(stable.get("loss", "gspo")),
            "tis_level": str(stable.get("tis_level", "token")),
            "overlong_penalty": True,
            "recipe_isolated": True,
        }
        if context.mode == "smoke":
            output["stable_cpu_fixture"] = run_stable_cpu_fixture(
                context.config,
                require_two_actors=bool(context.config.get("acceptance", {}).get("require_two_actors", False)),
            )
        if context.mode == "verl_train":
            maybe_run_verl_launch(
                context.config,
                output,
                overrides=build_stable_launch_overrides(context.config, config_path=context.config_path),
                default_enabled=True,
                default_module="trajweave.backends.verl.main_ppo",
                tracker=context.tracker,
            )
        if context.config.get("acceptance", {}).get("enabled", False):
            metrics = context.config.get("acceptance", {}).get("metrics", {}) or {}
            acceptance = stable_acceptance(
                metrics=metrics,
                require_two_actors=bool(context.config.get("acceptance", {}).get("require_two_actors", True)),
                max_policy_lag=int(context.config.get("acceptance", {}).get("max_policy_lag", 1)),
            )
            acceptance["name"] = "marti_mars2_stable"
            acceptance["message"] = (
                "all configured stability checks passed"
                if acceptance.get("status") == "passed"
                else "one or more configured stability checks failed"
            )
            output["acceptance"] = acceptance
            output["stable_acceptance"] = acceptance
        return output


__all__ = ["MARTIMARS2StableRecipePlugin"]
