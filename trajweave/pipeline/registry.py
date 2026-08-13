from __future__ import annotations

from trajweave.pipeline.context import RunContext
from trajweave.pipeline.plugin import RecipePlugin
from trajweave.recipes.agentflow.plugin import AgentFlowRecipePlugin
from trajweave.recipes.atgrpo.plugin import ATGRPORecipePlugin
from trajweave.recipes.comas.plugin import CoMASRecipePlugin
from trajweave.recipes.comlrl.plugin import CoMLRLRecipePlugin
from trajweave.recipes.doctor_mas.plugin import DoctorMASRecipePlugin
from trajweave.recipes.drmas_native.plugin import DrMASNativeRecipePlugin
from trajweave.recipes.gigpo.plugin import GiGPORecipePlugin
from trajweave.recipes.maporl.plugin import MAPoRLRecipePlugin
from trajweave.recipes.matpo.plugin import MATPORecipePlugin


def recipe_plugins() -> tuple[RecipePlugin, ...]:
    return (
        DrMASNativeRecipePlugin(),
        CoMLRLRecipePlugin(),
        CoMASRecipePlugin(),
        MAPoRLRecipePlugin(),
        AgentFlowRecipePlugin(),
        GiGPORecipePlugin(),
        ATGRPORecipePlugin(),
        MATPORecipePlugin(),
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
