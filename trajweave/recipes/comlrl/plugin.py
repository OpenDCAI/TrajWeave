from __future__ import annotations

from typing import Any

from trajweave.pipeline.context import RunContext
from trajweave.pipeline.launch import maybe_run_verl_launch
from trajweave.recipes.comlrl.config import build_comlrl_launch_overrides, resolve_comlrl_topology
from trajweave.recipes.comlrl.joint_math import run_comlrl_joint_math_smoke

_RECIPE_ALGORITHM_ALIASES = {
    "comlrl.joint_math": "magrpo",
    "comlrl_joint_math": "magrpo",
}


class CoMLRLRecipePlugin:
    name = "comlrl"

    def supports(self, context: RunContext) -> bool:
        return context.recipe_definition.family == "comlrl"

    def run(self, context: RunContext) -> dict[str, Any]:
        _validate_recipe_algorithm(context)
        output = context.base_output()
        output["comlrl"] = comlrl_summary(context.config)
        if context.mode == "smoke":
            result = run_comlrl_joint_math_smoke(output["comlrl"]["algorithm"])
            output.update(result)
            context.tracker.log_event("comlrl_smoke", "CoMLRL smoke completed.", result)
            return output
        if context.mode not in {"verl_train", "verl_plan"}:
            raise ValueError(f"Unsupported CoMLRL mode: {context.mode!r}.")
        maybe_run_verl_launch(
            context.config,
            output,
            overrides=build_comlrl_launch_overrides(context.config, config_path=context.config_path),
            default_enabled=True,
            default_module="trajweave.backends.verl.main_ppo",
            tracker=context.tracker,
            mode=context.mode,
        )
        return output


def comlrl_summary(config: dict[str, Any]) -> dict[str, Any]:
    comlrl = config.get("comlrl", {}) or {}
    topology = resolve_comlrl_topology(
        config,
        require_worker_assets=str(config.get("mode", "")) in {"verl_train", "verl_plan"},
    )
    return {
        "source_repository": "https://github.com/OpenMLRL/CoMLRL",
        "source_version": "v1.4.1-5-g3c724af",
        "source_commit": "3c724afd",
        "runtime_recipe": "comlrl_joint_math",
        "task": "math",
        "algorithm": str(comlrl.get("algorithm", "magrpo")).lower(),
        "agent_ids": list(topology["agent_ids"]),
        "model_ids": list(topology["model_ids"]),
        "worker_groups": topology["worker_groups"],
        "joint_mode": str(comlrl.get("joint_mode", "aligned")),
        "max_turns": int(comlrl.get("max_turns", 1)),
        "multi_actor_validation": topology["validation"],
        "integration_constraints": {
            "minimum_agents": 2,
            "iac_critic": "separate_per_agent",
            "shared_actor_value_head": False,
        },
    }


def _validate_recipe_algorithm(context: RunContext) -> None:
    configured = str((context.config.get("comlrl", {}) or {}).get("algorithm", "magrpo")).strip().lower()
    expected = _RECIPE_ALGORITHM_ALIASES.get(
        context.recipe,
        context.recipe_definition.name.removeprefix("comlrl."),
    )
    if configured != expected:
        raise ValueError(
            f"CoMLRL recipe {context.recipe!r} requires comlrl.algorithm={expected!r}, got {configured!r}."
        )


__all__ = ["CoMLRLRecipePlugin", "comlrl_summary"]
