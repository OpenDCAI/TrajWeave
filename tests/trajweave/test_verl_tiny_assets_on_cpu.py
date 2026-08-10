import json

from trajweave.backends.verl.tiny_assets import prepare_tiny_verl_assets


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
