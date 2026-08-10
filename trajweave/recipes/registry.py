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
        name="comas.peer_review_math",
        family="comas",
        task="math",
        runtime_recipe="comas_peer_review_math",
        allowed_modes=("smoke", "verl_train", "verl_plan"),
        aliases=("comas_peer_review_math", "comas.math"),
    ),
    RecipeDefinition(
        name="marti_mars2.single_mcts.smoke",
        family="marti_mars2",
        task="code",
        runtime_recipe="marti_mars2_single_mcts",
        allowed_modes=("smoke", "verl_train", "verl_plan", "marti_eval"),
        aliases=("marti_mars2_single_mcts", "marti-mars2-single-mcts", "mars2.single_mcts"),
    ),
    RecipeDefinition(
        name="marti_mars2.single_mcts.fidelity",
        family="marti_mars2",
        task="code",
        runtime_recipe="marti_mars2_single_mcts",
        aliases=("marti_mars2_fidelity", "marti-mars2-fidelity"),
    ),
    RecipeDefinition(
        name="marti_mars2.vanilla_grpo.baseline",
        family="marti_mars2",
        task="code",
        runtime_recipe="marti_mars2_single_mcts",
        aliases=("marti_mars2_vanilla_grpo", "marti-mars2-vanilla-grpo"),
    ),
    RecipeDefinition(
        name="marti_mars2.single_mcts.tree_credit_experimental",
        family="marti_mars2",
        task="code",
        runtime_recipe="marti_mars2_single_mcts",
        aliases=("marti_mars2_tree_credit_experimental", "marti-mars2-tree-credit-experimental"),
    ),
    RecipeDefinition(
        name="marti_mars2.stable.gspo_token_tis",
        family="marti_mars2_stable",
        task="code",
        runtime_recipe="marti_mars2_stable",
        aliases=("marti_mars2_stable", "mars2.stable.token"),
    ),
    RecipeDefinition(
        name="marti_mars2.stable.gspo_sequence_tis",
        family="marti_mars2_stable",
        task="code",
        runtime_recipe="marti_mars2_stable",
        aliases=("mars2.stable.sequence",),
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
