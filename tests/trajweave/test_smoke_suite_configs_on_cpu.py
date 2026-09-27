"""验证最小训练配置通过正式 recipe 入口组合，覆盖中文路径和算法分发。"""

from pathlib import Path

from hydra import compose, initialize_config_dir
import pytest

from trajweave.cli.prepare_smoke_suite import CASES, build_case
from trajweave.backends.verl.multi_actor import reconcile_multi_actor_global_assets
from trajweave.backends.verl.multi_actor.config import validate_trajweave_config
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from trajweave.runner import result_exit_code, run_from_config


@pytest.mark.parametrize("name,template,gpus,task", CASES, ids=[case[0] for case in CASES])
def test_real_training_suite_composes_through_recipe(name, template, gpus, task, tmp_path):
    repo = Path(__file__).resolve().parents[2]
    output = tmp_path / "最小训练 验收"
    model = tmp_path / "Qwen 模型"
    if name in {"maporl", "marft", "c3", "comlrl_iac", "comlrl_maac"}:
        from trajweave.backends.verl.tiny_assets import _write_tiny_model
        model.mkdir()
        _write_tiny_model(model)
    config = build_case(name=name, template=template, gpus=gpus, task=task,
                        model=model, output=output, python="python", repo=repo)
    assert config["mode"] == "verl_train"
    assert config["verl"]["execute"] is True
    ray_tmp = config["verl"]["env"]["RAY_TMPDIR"]
    assert len((ray_tmp + "/session_2026-09-27_18-19-36_859879_40560/sockets/plasma_store").encode()) <= 107
    assert "tiny_verl_assets" not in config["prepare"]
    assert config["prepare"]["verl_dataset"]["task_family"] == task
    if name in {"comlrl_madpo_iter", "comlrl_marlhf_iter"}:
        assert config["comlrl"]["iterative"]["num_target_candidates"] == 4
        assert config["comlrl"]["iterative"]["num_train_epochs"] == 1
    config["mode"] = "verl_plan"
    config["verl"]["execute"] = False
    config["prepare"] = {}
    result = run_from_config(config, config_path=str(output / f"{name}.yaml"))
    assert result_exit_code(result) == 0
    assert result["verl_launch"]["status"] == "dry_run"
    with initialize_config_dir(config_dir=str(repo / "verl/trainer/config"), version_base=None):
        composed = compose(config_name="ppo_trainer", overrides=result["verl_launch"]["command"][3:])
    assets = reconcile_multi_actor_global_assets(composed)
    validate_trajweave_config(composed, use_reference_policy=need_reference_policy(composed),
                             use_critic=need_critic(composed))
    if assets is not None:
        assert assets["model_path"] == str(model)
        assert assets["tokenizer_path"] == str(model)
    assert composed.actor_rollout_ref.model.path == str(model)
    if name in {"maporl", "marft", "c3", "comlrl_iac", "comlrl_maac"}:
        assert composed.critic.model.path == str(model)
        assert composed.critic.model.tokenizer_path == str(model)
    assert composed.trainer.total_training_steps == 2
    assert composed.trainer.n_gpus_per_node == gpus
    assert composed.trajweave.agent_loop_backend == ("vllm_marti_tq" if name == "marti_vllm" else "hf_local_tq")
    assert Path(composed.trajweave.config).parent == output
    assert composed.trainer.default_local_dir.startswith(str(output))
