from __future__ import annotations

from trajweave.pipeline.context import RunContext
from trajweave.pipeline.plugin import RecipePlugin
from trajweave.recipes.agentflow.plugin import AgentFlowRecipePlugin
from trajweave.recipes.atgrpo.plugin import ATGRPORecipePlugin
from trajweave.recipes.c3.plugin import C3RecipePlugin
from trajweave.recipes.comas.plugin import CoMASRecipePlugin
from trajweave.recipes.comlrl.plugin import CoMLRLRecipePlugin
from trajweave.recipes.doctor_mas.plugin import DoctorMASRecipePlugin
from trajweave.recipes.drmas_native.plugin import DrMASNativeRecipePlugin
from trajweave.recipes.gigpo.plugin import GiGPORecipePlugin
from trajweave.recipes.maporl.plugin import MAPoRLRecipePlugin
from trajweave.recipes.marti_mars2.plugin import MARTIMARS2RecipePlugin
from trajweave.recipes.marti_mars2.stable_plugin import MARTIMARS2StableRecipePlugin
from trajweave.recipes.marft.plugin import MARFTRecipePlugin
from trajweave.recipes.marshal.plugin import MARSHALRecipePlugin
from trajweave.recipes.matpo.plugin import MATPORecipePlugin
from trajweave.recipes.mrlx.plugin import MrlXRecipePlugin
from trajweave.recipes.wideseek_r1.plugin import WideSeekR1RecipePlugin


def recipe_plugins() -> tuple[RecipePlugin, ...]:
    return (
        MARSHALRecipePlugin(),
        WideSeekR1RecipePlugin(),
        DrMASNativeRecipePlugin(),
        MARFTRecipePlugin(),
        MrlXRecipePlugin(),
        C3RecipePlugin(),
        CoMLRLRecipePlugin(),
        CoMASRecipePlugin(),
        MAPoRLRecipePlugin(),
        AgentFlowRecipePlugin(),
        GiGPORecipePlugin(),
        MARTIMARS2RecipePlugin(),
        MARTIMARS2StableRecipePlugin(),
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
