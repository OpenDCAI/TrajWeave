import json
from pathlib import Path

import pytest

from trajweave.recipes.marti_mars2 import run_single_mcts_smoke
from trajweave.recipes.registry import resolve_recipe
from trajweave.runner import run_from_config


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_recipe_registry_exposes_marti_mars2_single_mcts():
    recipe = resolve_recipe("marti_mars2_single_mcts")

    assert recipe.name == "marti_mars2.single_mcts.smoke"
    assert recipe.family == "marti_mars2"
    assert recipe.runtime_recipe == "marti_mars2_single_mcts"


def test_marti_mars2_single_mcts_smoke_builds_tree_group_samples():
    _summary, result = run_single_mcts_smoke(max_num_nodes=2, rollouts_per_task=1)

    assert len(result.trajectories) == 2
    assert len(result.samples) == 4
    assert {sample.metadata["credit"] for sample in result.samples} == {"tree_group_norm"}
    assert {sample.metadata["rollout_logprob_available"] for sample in result.samples} == {True}
    assert [sample.advantage for sample in result.samples[:2]] == pytest.approx([0.70710678, -0.70710678])
    assert [sample.advantage for sample in result.samples[2:]] == [0.0, 0.0]
    assert all("tree_id" in sample.metadata for sample in result.samples)
    assert all("node_id" in sample.metadata for sample in result.samples)
    assert result.success_rate == 1.0


def test_yaml_runner_runs_marti_mars2_single_mcts_smoke(tmp_path):
    result = run_from_config(
        {
            "recipe": "marti_mars2.single_mcts.smoke",
            "mode": "smoke",
            "run": {"root_dir": str(tmp_path), "name": "unit-marti-mars2-smoke"},
            "logging": {"console": False},
            "marti_mars2": {"max_num_nodes": 2},
            "rollout": {"rollouts_per_task": 1},
        }
    )

    run_dir = Path(result["run_dir"])
    samples = read_jsonl(run_dir / "trajectories" / "samples.jsonl")
    assert result["recipe"] == "marti_mars2.single_mcts.smoke"
    assert result["trajectories"] == 2
    assert result["samples"] == 4
    assert result["marti_mars2"]["tree_identity_required"] is True
    assert {row["metadata"]["tree_id"] for row in samples} == {"code-add-one:rollout-0", "code-square:rollout-0"}
    assert {row["metadata"]["credit"] for row in samples} == {"tree_group_norm"}
