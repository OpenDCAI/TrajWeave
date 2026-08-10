import json

from trajweave.backends.verl.tiny_assets import prepare_tiny_verl_assets, prepare_verl_dataset
from trajweave.pipeline.assets import maybe_prepare_assets


def test_tiny_verl_assets_record_recipe_name(tmp_path):
    output = prepare_tiny_verl_assets(
        tmp_path / "assets",
        train_size=1,
        val_size=1,
        task_family="math",
        recipe_name="agentflow_planner_tool",
    )

    with open(output["train_file"], encoding="utf-8") as handle:
        row = json.loads(handle.readline())

    assert output["recipe_name"] == "agentflow_planner_tool"
    assert row["extra_info"]["trajweave_recipe"] == "agentflow_planner_tool"


def test_tiny_verl_assets_default_output_dir_is_isolated_by_recipe(monkeypatch):
    from trajweave.backends.verl import tiny_assets

    monkeypatch.setattr(tiny_assets, "prepare_tiny_verl_assets", lambda **kwargs: kwargs)

    agentflow = maybe_prepare_assets(
        {
            "recipe": "agentflow.flow_grpo.planner_tool",
            "prepare": {"tiny_verl_assets": {"enabled": True}},
        }
    )
    maporl = maybe_prepare_assets(
        {
            "recipe": "maporl.debate_math.full_verl_tiny",
            "prepare": {"tiny_verl_assets": {"enabled": True}},
        }
    )

    assert agentflow["output_dir"] == "outputs/agentflow.flow_grpo.planner_tool_tiny_assets"
    assert maporl["output_dir"] == "outputs/maporl.debate_math.full_verl_tiny_tiny_assets"
    assert agentflow["output_dir"] != maporl["output_dir"]


def test_verl_dataset_preparation_does_not_create_a_tiny_model(tmp_path):
    output = prepare_verl_dataset(
        tmp_path / "assets",
        train_size=1,
        val_size=1,
        task_family="math",
        recipe_name="comas_peer_review_math",
    )

    assert not (tmp_path / "assets/model").exists()
    with open(output["train_file"], encoding="utf-8") as handle:
        row = json.loads(handle.readline())
    assert row["extra_info"]["trajweave_recipe"] == "comas_peer_review_math"


def test_tiny_verl_assets_support_marti_mars2_code(tmp_path):
    output = prepare_tiny_verl_assets(
        tmp_path / "assets",
        train_size=1,
        val_size=1,
        task_family="code",
        recipe_name="marti_mars2_single_mcts",
    )

    with open(output["train_file"], encoding="utf-8") as handle:
        row = json.loads(handle.readline())

    assert output["task_family"] == "code"
    assert row["ability"] == "code"
    assert row["extra_info"]["trajweave_recipe"] == "marti_mars2_single_mcts"


def test_tiny_verl_assets_support_controlled_mixed_reward_code(tmp_path):
    output = prepare_tiny_verl_assets(
        tmp_path / "assets",
        train_size=1,
        val_size=1,
        task_family="controlled_code",
        recipe_name="marti_mars2_fidelity",
    )

    with open(output["train_file"], encoding="utf-8") as handle:
        row = json.loads(handle.readline())

    assert output["task_family"] == "controlled_code"
    assert row["data_source"] == "trajweave_controlled_code"
    assert row["reward_model"]["test_cases"]["fn_name"] == "zigzag_code"
    assert row["extra_info"]["acceptance_target"] == "mixed_verifier_rewards_within_tree"
