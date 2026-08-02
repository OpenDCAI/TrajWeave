from __future__ import annotations

from trajweave.pipeline.context import RunContext
from trajweave.pipeline.plugin import RecipePlugin
from trajweave.recipes.agentflow.plugin import AgentFlowRecipePlugin
from trajweave.recipes.doctor_mas.plugin import DoctorMASRecipePlugin
from trajweave.recipes.drmas_native.plugin import DrMASNativeRecipePlugin
from trajweave.recipes.marti_mars2.plugin import MARTIMARS2RecipePlugin
from trajweave.recipes.maporl.plugin import MAPoRLRecipePlugin


def recipe_plugins() -> tuple[RecipePlugin, ...]:
    return (
        DrMASNativeRecipePlugin(),
        MAPoRLRecipePlugin(),
        AgentFlowRecipePlugin(),
        MARTIMARS2RecipePlugin(),
        DoctorMASRecipePlugin(),
    )


def run_recipe(context: RunContext) -> dict:
    for plugin in recipe_plugins():
        if plugin.supports(context):
            return plugin.run(context)
    raise ValueError(
        f"No TrajWeave recipe plugin supports recipe={context.recipe!r}, "
        f"family={context.recipe_definition.family!r}, mode={context.mode!r}."
    )
