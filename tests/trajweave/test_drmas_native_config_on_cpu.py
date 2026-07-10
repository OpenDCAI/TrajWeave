from pathlib import Path

import pytest
import yaml

from trajweave.recipes.drmas_native.config import build_drmas_native_launch_overrides
from trajweave.runner import run_from_config


def test_drmas_native_math_config_prepares_agent_wise_verl_launch():
    result = run_from_config(
        {
            "recipe": "drmas_native_math",
            "mode": "verl_train",
            "prepare": {"tiny_verl_assets": {"enabled": False}},
            "drmas_native": {
                "agent_ids": ["Solver Agent", "Verifier Agent"],
                "model_ids": ["shared", "shared"],
                "agent_loop_backend": "synthetic_tq",
                "max_loop_num": 2,
            },
            "verl": {
                "enabled": True,
                "execute": False,
                "overrides": ["trainer.use_v1=true"],
            },
        },
        config_path="configs/drmas/math_verl_tiny.yaml",
    )

    command = result["verl_launch"]["command"]
    command_text = " ".join(command)
    assert result["drmas_native"]["task"] == "math"
    assert "trajweave.backends.verl.main_ppo" in command
    assert "++algorithm.group_by_agent_id=true" in command
    assert '+agent.agent_ids=["Solver Agent","Verifier Agent"]' in command
    assert "+trajweave.verl_extensions=[drmas_agent_wise_grpo]" in command
    assert "+agent.orchestra.math.max_loop_num=2" in command
    assert "+trajweave.turn_padding_multiple=2" in command
    assert "+trajweave.agent_loop_backend=synthetic_tq" in command
    assert "trajweave.backends.verl.agent_loop.TrajWeaveAgentLoopManager" in command_text


def test_drmas_native_config_rejects_native_verl_tq_backend():
    try:
        run_from_config(
            {
                "recipe": "drmas_native_math",
                "mode": "verl_train",
                "prepare": {"tiny_verl_assets": {"enabled": False}},
                "drmas_native": {
                    "agent_ids": ["Solver Agent", "Verifier Agent"],
                    "model_ids": ["shared", "shared"],
                    "agent_loop_backend": "verl_tq",
                },
                "verl": {"enabled": True, "execute": False},
            }
        )
    except ValueError as exc:
        assert "not verl_tq" in str(exc)
    else:
        raise AssertionError("DrMAS native should reject native verl_tq.")


def test_drmas_native_config_rejects_agent_ids_not_supported_by_runtime():
    with pytest.raises(ValueError, match="fixed agent_ids"):
        run_from_config(
            {
                "recipe": "drmas_native_math",
                "mode": "verl_train",
                "prepare": {"tiny_verl_assets": {"enabled": False}},
                "drmas_native": {
                    "agent_ids": ["custom_solver", "custom_verifier"],
                    "model_ids": ["shared", "shared"],
                    "agent_loop_backend": "synthetic_tq",
                    "max_loop_num": 2,
                },
                "verl": {"enabled": True, "execute": False},
            }
        )


def test_drmas_native_rejects_silent_heterogeneous_model_configuration():
    with pytest.raises(ValueError, match="heterogeneous"):
        run_from_config(
            {
                "recipe": "drmas_native_math",
                "mode": "verl_train",
                "prepare": {"tiny_verl_assets": {"enabled": False}},
                "drmas_native": {
                    "agent_ids": ["Solver Agent", "Verifier Agent"],
                    "model_ids": ["solver_model", "verifier_model"],
                    "agent_loop_backend": "synthetic_tq",
                    "max_loop_num": 2,
                },
                "verl": {"enabled": True, "execute": False},
            }
        )


@pytest.mark.parametrize(
    ("recipe", "native_config"),
    [
        (
            "drmas_native_math",
            {
                "agent_ids": ["Solver Agent", "Verifier Agent"],
                "model_ids": ["shared", "shared"],
                "max_loop_num": 3,
            },
        ),
        (
            "drmas_native_search",
            {
                "agent_ids": ["Verifier Agent", "Search Agent", "Answer Agent"],
                "model_ids": ["shared", "shared", "shared"],
                "max_loop_num": 1,
            },
        ),
        (
            "drmas_native_math",
            {
                "agent_ids": ["Solver Agent", "Verifier Agent"],
                "model_ids": ["shared", "shared"],
                "max_loop_num": 2.5,
            },
        ),
    ],
)
def test_drmas_native_config_rejects_unsupported_max_loop_values(recipe, native_config):
    native_config["agent_loop_backend"] = "synthetic_tq"
    with pytest.raises(ValueError, match="max_loop_num"):
        run_from_config(
            {
                "recipe": recipe,
                "mode": "verl_train",
                "prepare": {"tiny_verl_assets": {"enabled": False}},
                "drmas_native": native_config,
                "verl": {"enabled": True, "execute": False},
            }
        )


@pytest.mark.parametrize(
    ("config_path", "recipe", "padding_multiple"),
    [
        ("configs/drmas/math_qwen05b_2gpu.yaml", "drmas.math.verl_tiny", 2),
        ("configs/drmas/search_qwen05b_2gpu.yaml", "drmas.search.verl_tiny", 4),
    ],
)
def test_drmas_qwen_configs_use_native_training_recipe(config_path, recipe, padding_multiple):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    overrides = build_drmas_native_launch_overrides(config, config_path=config_path)
    override_keys = [item.split("=", 1)[0].lstrip("+") for item in overrides]

    assert config["recipe"] == recipe
    assert config["mode"] == "verl_train"
    assert f"+trajweave.turn_padding_multiple={padding_multiple}" in overrides
    assert len(override_keys) == len(set(override_keys))
