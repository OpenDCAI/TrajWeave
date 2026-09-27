from trajweave.recipes.registry import resolve_recipe


def test_recipe_registry_keeps_legacy_drmas_aliases():
    assert resolve_recipe("doctor_mas_math").name == "drmas.math.smoke"
    assert resolve_recipe("drmas_native_math").name == "drmas.math.verl_tiny"
    assert resolve_recipe("doctor_mas_native_search").name == "drmas.search.verl_tiny"


def test_recipe_registry_exposes_maporl_namespace():
    recipe = resolve_recipe("maporl_debate_math")

    assert recipe.name == "maporl.debate_math.full_verl_tiny"
    assert recipe.family == "maporl"
    assert recipe.runtime_recipe == "maporl_debate_math"


def test_recipe_registry_exposes_agentflow_namespace():
    recipe = resolve_recipe("agentflow_planner_tool")

    assert recipe.name == "agentflow.flow_grpo.planner_tool"
    assert recipe.family == "agentflow"
    assert recipe.runtime_recipe == "agentflow_planner_tool"


def test_recipe_registry_exposes_gigpo_namespace():
    recipe = resolve_recipe("gigpo_solver_verifier_math")

    assert recipe.name == "gigpo.solver_verifier_math"
    assert recipe.family == "gigpo"
