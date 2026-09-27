"""验证公开配置在不同模型路径与外部 GPU 分配下保持一致。"""

from __future__ import annotations

import copy
import shutil
from pathlib import Path

import pytest
import yaml
from hydra import compose, initialize_config_dir

from trajweave.pipeline.config import load_yaml_config, resolve_model_references
from trajweave.runner import run_from_config


def test_model_environment_resolution_preserves_deferred_hydra_and_input(monkeypatch):
    monkeypatch.setenv("TRAJWEAVE_MODEL_TEST", '/tmp/模型, with "quotes"')
    monkeypatch.delenv("TRAJWEAVE_RUN_DIR", raising=False)
    expression = "${oc.env:TRAJWEAVE_MODEL_TEST,Qwen/Qwen2.5-0.5B-Instruct}"
    config = {
        "agents": [{"model_path": expression, "tokenizer_path": expression}],
        "verl": {"overrides": ["trainer.default_local_dir=${oc.env:TRAJWEAVE_RUN_DIR}/checkpoints"]},
    }
    before = copy.deepcopy(config)
    result = resolve_model_references(config)
    assert config == before
    assert result["agents"][0]["model_path"] == '/tmp/模型, with "quotes"'
    assert result["agents"][0]["tokenizer_path"] == '/tmp/模型, with "quotes"'
    assert result["verl"]["overrides"] == config["verl"]["overrides"]


def test_model_reference_uses_public_default(monkeypatch):
    monkeypatch.delenv("TRAJWEAVE_MODEL_TEST", raising=False)
    config = {"model_path": "${oc.env:TRAJWEAVE_MODEL_TEST,Qwen/Qwen2.5-0.5B-Instruct}"}
    assert resolve_model_references(config)["model_path"] == "Qwen/Qwen2.5-0.5B-Instruct"


def test_required_model_environment_fails_explicitly(monkeypatch):
    from omegaconf.errors import InterpolationResolutionError

    monkeypatch.delenv("TRAJWEAVE_MODEL_TEST", raising=False)
    with pytest.raises(InterpolationResolutionError, match="TRAJWEAVE_MODEL_TEST"):
        resolve_model_references({"model_path": "${oc.env:TRAJWEAVE_MODEL_TEST}"})


def test_maporl_model_environment_reaches_tokenizer_validation_and_hydra(tmp_path, monkeypatch):
    from trajweave.backends.verl.tiny_assets import _write_tiny_model

    repo = Path(__file__).resolve().parents[2]
    model_a = tmp_path / "模型 A,revision"
    model_b = tmp_path / "模型 B"
    model_a.mkdir()
    _write_tiny_model(model_a)
    shutil.copytree(model_a, model_b)
    monkeypatch.setenv("TRAJWEAVE_QWEN05B_INSTRUCT_PATH", str(model_a))
    monkeypatch.setenv("TRAJWEAVE_QWEN15B_INSTRUCT_PATH", str(model_b))
    path = repo / "configs/maporl/debate_math_multi_actor_qwen05b_qwen15b_2gpu.yaml"
    config = load_yaml_config(path)
    config.update(mode="verl_plan", prepare={}, run={"root_dir": str(tmp_path / "runs")})
    config["verl"]["execute"] = False
    result = run_from_config(config, config_path=str(path))
    assert result["verl_launch"]["status"] == "dry_run"
    with initialize_config_dir(config_dir=str(repo / "verl/trainer/config"), version_base=None):
        composed = compose(config_name="ppo_trainer", overrides=result["verl_launch"]["command"][3:])
    assert composed.actor_rollout_ref.model.path == str(model_a)
    assert [g.model_path for g in composed.agent.worker_groups] == [str(model_a), str(model_b)]


def test_public_presets_preserve_caller_gpu_selection(monkeypatch):
    from trajweave.backends.verl.launcher import VerlTrainerLaunchConfig, VerlTrainerLauncher

    repo = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6,7")
    for path in (repo / "configs").rglob("*.yaml"):
        raw = yaml.safe_load(path.read_text())
        env = (raw.get("verl") or {}).get("env") or {}
        assert "CUDA_VISIBLE_DEVICES" not in env, path
        if env:
            launcher = VerlTrainerLauncher(VerlTrainerLaunchConfig(env=env))
            assert launcher._subprocess_env()["CUDA_VISIBLE_DEVICES"] == "6,7"
