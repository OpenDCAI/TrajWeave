from pathlib import Path


def test_readme_registers_marti_recipe_status_and_entrypoints():
    repo = Path(__file__).resolve().parents[2]
    readme = (repo / "README.md").read_text(encoding="utf-8")
    catalog = (repo / "docs/recipes.md").read_text(encoding="utf-8")
    validation = (repo / "docs/validation.md").read_text(encoding="utf-8")

    # 首页保留入口，细节移至目录；检查可运行路径与限制，不绑定中文标题。
    assert "MARTI-MARS²" in readme
    assert "docs/recipes.md" in readme
    assert "docs/validation.md" in readme
    for component in ("CodeExecutionEnvironment", "TreeSearchProtocol", "trajweave_multi_actor_sync"):
        assert component in catalog
    for name in ("single_mcts_smoke.yaml", "single_mcts_verl_tiny.yaml", "stage1e_multi_agent_vllm_smoke.yaml"):
        path = Path("configs/marti_mars2") / name
        assert str(path) in catalog
        assert (repo / path).is_file()
    assert "cross-step asynchronous buffering is not enabled" in validation
