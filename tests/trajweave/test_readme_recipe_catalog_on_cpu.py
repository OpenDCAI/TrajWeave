from pathlib import Path


def test_readme_registers_marti_recipe_status_and_entrypoints():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "五条可运行的 MAS 路径" in readme
    assert "### MARTI-MARS² Code" in readme
    assert "`CodeExecutionEnvironment`" in readme
    assert "`TreeSearchProtocol`" in readme
    assert "`trajweave_multi_actor_sync`" in readme
    assert "configs/marti_mars2/single_mcts_smoke.yaml" in readme
    assert "configs/marti_mars2/single_mcts_verl_tiny.yaml" in readme
    assert "configs/marti_mars2/stage1e_multi_agent_vllm_async_smoke.yaml" in readme
    assert "0.5B 双卡 native vLLM async 3-step 已验证" in readme
