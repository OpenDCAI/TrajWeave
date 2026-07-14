from trajweave.recipes.matpo import run_smoke
from trajweave.recipes.registry import resolve_recipe


def test_matpo_recipe_registry_alias():
    recipe = resolve_recipe("matpo_browse")
    assert recipe.name == "matpo.browse_qa.parent_broadcast"
    assert recipe.runtime_recipe == "matpo_browse"


def test_matpo_smoke_builds_parent_child_samples():
    summary, result = run_smoke(rollouts_per_task=2, max_turns=3)

    assert summary.trajectories == 4
    assert summary.samples == 12
    assert summary.success_rate == 1.0
    assert all(sample.metadata["credit"] == "matpo_parent_broadcast_grpo" for sample in result.samples)

    child_samples = [sample for sample in result.samples if sample.metadata.get("is_from_subagent_tool")]
    main_samples = [sample for sample in result.samples if not sample.metadata.get("is_from_subagent_tool")]
    assert child_samples
    assert main_samples
    parent_ids = {sample.metadata["reqs_id"] for sample in main_samples}
    assert all(sample.metadata["parent_reqs_id"] in parent_ids for sample in child_samples)
    assert all(sample.metadata.get("parent_advantage_broadcast") for sample in child_samples)
