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
