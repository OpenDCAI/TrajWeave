import json

from transformers import AutoConfig, AutoTokenizer

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


def test_mrlx_tiny_assets_use_mrlx_prompt_and_protocol_vocabulary(tmp_path):
    output = prepare_tiny_verl_assets(
        tmp_path / "mrlx-assets",
        train_size=4,
        val_size=1,
        task_family="browse_qa",
        recipe_name="mrlx.mgrpo_research_qa",
    )

    with open(output["train_file"], encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    system_prompt = rows[0]["prompt"][0]["content"]
    assert "MrlX main explorer" in system_prompt
    assert "MATPO" not in system_prompt

    tokenizer = AutoTokenizer.from_pretrained(output["model_path"])
    model_config = AutoConfig.from_pretrained(output["model_path"])
    assert model_config.max_position_embeddings >= 128 + 96
    for text in (
        "CALL sub_adapter: capital France",
        "CALL search_and_browse: capital France",
        "Research result: Paris",
        "Final answer: Paris",
    ):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        assert tokenizer.unk_token_id not in token_ids
        assert tokenizer.decode(token_ids) == text
    for row in rows:
        answer_ids = tokenizer.encode(row["reward_model"]["ground_truth"], add_special_tokens=False)
        assert tokenizer.unk_token_id not in answer_ids

    question_ids = {
        row["prompt"][-1]["content"]: tokenizer.encode(row["prompt"][-1]["content"], add_special_tokens=False)
        for row in rows
    }
    assert question_ids["What is the capital of Japan?"] != question_ids["What is the capital of Canada?"]


def test_m_grpo_alias_uses_mrlx_browse_prompt(tmp_path):
    output = prepare_verl_dataset(
        tmp_path / "m-grpo-data",
        train_size=1,
        val_size=1,
        task_family="browse_qa",
        recipe_name="m-grpo",
    )

    with open(output["train_file"], encoding="utf-8") as handle:
        row = json.loads(handle.readline())
    assert "MrlX main explorer" in row["prompt"][0]["content"]
