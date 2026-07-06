from trajweave.recipes.registry import resolve_recipe


def test_recipe_registry_keeps_legacy_drmas_aliases():
    assert resolve_recipe("doctor_mas_math").name == "drmas.math.smoke"
    assert resolve_recipe("drmas_native_math").name == "drmas.math.verl_tiny"
    assert resolve_recipe("doctor_mas_native_search").name == "drmas.search.verl_tiny"


def test_recipe_registry_exposes_maporl_namespace():
    recipe = resolve_recipe("maporl_debate_math")

    assert recipe.name == "maporl.debate_math.verl_tiny"
    assert recipe.family == "maporl"
    assert recipe.runtime_recipe == "maporl_debate_math"
