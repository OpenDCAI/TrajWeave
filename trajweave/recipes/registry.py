from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RecipeDefinition:
    name: str
    family: str
    task: str
    runtime_recipe: str
    allowed_modes: tuple[str, ...]
    aliases: tuple[str, ...] = ()

    def matches(self, value: str) -> bool:
        return value == self.name or value in self.aliases


RECIPES: tuple[RecipeDefinition, ...] = (
    RecipeDefinition(
        name="marft.cooperative_math",
        family="marft",
        task="math",
        runtime_recipe="marft_math_workflow",
        allowed_modes=("smoke", "verl_train", "verl_plan"),
        aliases=("marft_math_workflow", "marft.math"),
    ),
    RecipeDefinition(
        name="mrlx.mgrpo_research_qa",
        family="mrlx",
        task="browse_qa",
        runtime_recipe="mrlx_research_qa",
        allowed_modes=("smoke", "verl_train", "verl_plan"),
        aliases=("mrlx.deep_research", "mrlx_research_qa", "m-grpo"),
    ),
    RecipeDefinition(
        name="c3.reasoner_actor_math",
        family="c3",
        task="math",
        runtime_recipe="c3_reasoner_actor_math",
        allowed_modes=("smoke", "verl_train", "verl_plan"),
        aliases=("c3.math", "c3_reasoner_actor_math"),
    ),
    RecipeDefinition(
        name="comlrl.magrpo",
        family="comlrl",
        task="math",
        runtime_recipe="comlrl_joint_math",
        allowed_modes=("smoke", "verl_train", "verl_plan"),
        aliases=("comlrl.joint_math", "comlrl_joint_math"),
    ),
    RecipeDefinition("comlrl.mareinforce", "comlrl", "math", "comlrl_joint_math", ("smoke", "verl_train", "verl_plan")),
    RecipeDefinition("comlrl.marloo", "comlrl", "math", "comlrl_joint_math", ("smoke", "verl_train", "verl_plan")),
    RecipeDefinition("comlrl.maremax", "comlrl", "math", "comlrl_joint_math", ("smoke", "verl_train", "verl_plan")),
    RecipeDefinition("comlrl.iac", "comlrl", "math", "comlrl_joint_math", ("smoke", "verl_train", "verl_plan")),
    RecipeDefinition("comlrl.maac", "comlrl", "math", "comlrl_joint_math", ("smoke", "verl_train", "verl_plan")),
    RecipeDefinition("comlrl.madpo", "comlrl", "math", "comlrl_joint_math", ("smoke", "verl_train", "verl_plan")),
    RecipeDefinition("comlrl.marlhf", "comlrl", "math", "comlrl_joint_math", ("smoke", "verl_train", "verl_plan")),
    RecipeDefinition("comlrl.madpo_iter", "comlrl", "math", "comlrl_joint_math", ("smoke", "verl_train", "verl_plan")),
    RecipeDefinition("comlrl.marlhf_iter", "comlrl", "math", "comlrl_joint_math", ("smoke", "verl_train", "verl_plan")),
    RecipeDefinition(
        name="drmas.math.smoke",
        family="doctor_mas",
        task="math",
        runtime_recipe="doctor_mas_math",
        allowed_modes=("smoke", "train_tiny", "verl_train", "verl_plan"),
        aliases=("doctor_mas_math",),
    ),
    RecipeDefinition(
        name="drmas.search.smoke",
        family="doctor_mas",
        task="search",
        runtime_recipe="doctor_mas_search",
        allowed_modes=("smoke", "verl_train", "verl_plan"),
        aliases=("doctor_mas_search",),
    ),
    RecipeDefinition(
        name="drmas.math.verl_tiny",
        family="drmas_native",
        task="math",
        runtime_recipe="doctor_mas_math",
        allowed_modes=("verl_train", "verl_plan"),
        aliases=("drmas_native_math", "doctor_mas_native_math"),
    ),
    RecipeDefinition(
        name="drmas.search.verl_tiny",
        family="drmas_native",
        task="search",
        runtime_recipe="doctor_mas_search",
        allowed_modes=("verl_train", "verl_plan"),
        aliases=("drmas_native_search", "doctor_mas_native_search"),
    ),
    RecipeDefinition(
        name="maporl.debate_math.full_verl_tiny",
        family="maporl",
        task="math",
        runtime_recipe="maporl_debate_math",
        allowed_modes=("smoke", "verl_train", "verl_plan"),
        aliases=("maporl_debate_math", "maporl.debate_math.verl_tiny"),
    ),
    RecipeDefinition(
        name="agentflow.flow_grpo.planner_tool",
        family="agentflow",
        task="math",
        runtime_recipe="agentflow_planner_tool",
        allowed_modes=("smoke", "verl_train", "verl_plan"),
        aliases=("agentflow_planner_tool", "agentflow.flow_grpo"),
    ),
    RecipeDefinition(
        name="gigpo.solver_verifier_math",
        family="gigpo",
        task="math",
        runtime_recipe="gigpo_solver_verifier_math",
        allowed_modes=("smoke", "verl_train", "verl_plan"),
        aliases=("gigpo_solver_verifier_math", "gigpo.math"),
    ),
    RecipeDefinition(
        name="atgrpo.solver_verifier_math",
        family="atgrpo",
        task="math",
        runtime_recipe="atgrpo_solver_verifier_math",
        allowed_modes=("smoke", "verl_train", "verl_plan"),
        aliases=("atgrpo_solver_verifier_math", "atgrpo.math"),
    ),
    RecipeDefinition(
        name="comas.peer_review_math",
        family="comas",
        task="math",
        runtime_recipe="comas_peer_review_math",
        allowed_modes=("smoke", "verl_train", "verl_plan"),
        aliases=("comas_peer_review_math", "comas.math"),
    ),
    RecipeDefinition(
        name="matpo.browse_qa.parent_broadcast",
        family="matpo",
        task="browse_qa",
        runtime_recipe="matpo_browse",
        allowed_modes=("smoke", "verl_train", "verl_plan"),
        aliases=("matpo_browse", "matpo.browse_qa"),
    ),
)


def resolve_recipe(name: str) -> RecipeDefinition:
    for recipe in RECIPES:
        if recipe.matches(name):
            return recipe
    known = sorted({recipe.name for recipe in RECIPES} | {alias for recipe in RECIPES for alias in recipe.aliases})
    raise ValueError(f"Unknown recipe: {name}. Known recipes: {known}")


def is_recipe_family(name: str, family: str) -> bool:
    return resolve_recipe(name).family == family


def validate_recipe_mode(recipe: RecipeDefinition, mode: str) -> None:
    if mode not in recipe.allowed_modes:
        raise ValueError(
            f"Unsupported mode {mode!r} for recipe {recipe.name!r}. Allowed modes: {list(recipe.allowed_modes)}"
        )
