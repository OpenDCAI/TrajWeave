import json
from pathlib import Path

import pytest

from trajweave.backends.verl.async_buffer import run_asymmetric_three_step_fixture
from trajweave.recipes.marti_mars2 import build_marti_mars2_launch_overrides, run_single_mcts_smoke
from trajweave.recipes.registry import resolve_recipe
from trajweave.runner import load_yaml_config, run_from_config


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_recipe_registry_exposes_marti_mars2_single_mcts():
    recipe = resolve_recipe("marti_mars2_single_mcts")

    assert recipe.name == "marti_mars2.single_mcts.smoke"
    assert recipe.family == "marti_mars2"
    assert recipe.runtime_recipe == "marti_mars2_single_mcts"


def test_recipe_registry_exposes_marti_mars2_vanilla_grpo_baseline():
    recipe = resolve_recipe("marti_mars2_vanilla_grpo")

    assert recipe.name == "marti_mars2.vanilla_grpo.baseline"
    assert recipe.family == "marti_mars2"
    assert recipe.runtime_recipe == "marti_mars2_single_mcts"


def test_marti_mars2_single_mcts_smoke_builds_tree_group_samples():
    _summary, result = run_single_mcts_smoke(max_num_nodes=2, rollouts_per_task=1)

    assert len(result.trajectories) == 2
    assert len(result.samples) == 4
    assert {sample.metadata["credit"] for sample in result.samples} == {"marti_mars2_fidelity_group_grpo"}
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
    assert {row["metadata"]["credit"] for row in samples} == {"marti_mars2_fidelity_group_grpo"}


def test_async_buffer_fixture_exercises_stale_sample_and_sync_boundary():
    result = run_asymmetric_three_step_fixture()

    assert result["acceptance"]["status"] == "passed"
    assert result["updates"] == {"policy_a": 3, "policy_b": 2}
    assert result["metrics"]["buffer/samples_stale"] == 1
    assert result["metrics"]["buffer/policy_a/policy_lag"] == 0
    assert result["metrics"]["buffer/policy_b/policy_lag"] == 0
    assert result["pending"] == 0


def test_yaml_runner_exposes_async_buffer_fixture(tmp_path):
    result = run_from_config(
        {
            "recipe": "marti_mars2.single_mcts.fidelity",
            "mode": "smoke",
            "run": {"root_dir": str(tmp_path), "name": "unit-marti-mars2-async"},
            "logging": {"console": False},
            "marti_mars2": {"max_num_nodes": 2, "async_fixture": True},
            "rollout": {"rollouts_per_task": 1},
        }
    )

    assert result["async_buffer_fixture"]["acceptance"]["status"] == "passed"


def test_marti_mars2_verl_overrides_keep_correction_hook_gated():
    disabled = build_marti_mars2_launch_overrides(
        {
            "marti_mars2": {"max_num_nodes": 2, "agent_loop_backend": "synthetic_tq"},
            "verl": {"overrides": []},
        },
        config_path="configs/marti_mars2/single_mcts_verl_vllm_dryrun.yaml",
    )
    enabled = build_marti_mars2_launch_overrides(
        {
            "marti_mars2": {
                "max_num_nodes": 2,
                "agent_loop_backend": "hf_local_tq",
                "enable_vllm_is_correction": True,
            },
            "verl": {"overrides": []},
        },
        config_path="configs/marti_mars2/single_mcts_verl_tiny.yaml",
    )

    assert "+trajweave.credit_allocator=marti_mars2_fidelity_group_grpo" in disabled
    assert "+agent.orchestra.marti_mars2.credit_mode=fidelity" in disabled
    assert "+agent.orchestra.marti_mars2.initial_candidates=2" in disabled
    assert "+trajweave.correction_hook.enabled=false" in disabled
    assert not any(item.startswith("algorithm.rollout_correction.rollout_is=") for item in disabled)
    assert "+trajweave.correction_hook.enabled=true" in enabled
    assert "algorithm.rollout_correction.rollout_is=token" in enabled
    assert "actor_rollout_ref.rollout.calculate_log_probs=true" in enabled


def test_marti_mars2_async_updates_are_opt_in():
    disabled = build_marti_mars2_launch_overrides(
        {"marti_mars2": {"max_num_nodes": 2}, "verl": {"overrides": []}},
        config_path=None,
    )
    enabled = build_marti_mars2_launch_overrides(
        {"marti_mars2": {"max_num_nodes": 2, "async_updates": True}, "verl": {"overrides": []}},
        config_path=None,
    )

    assert "+trajweave.async_buffer.enabled=false" in disabled
    assert "+trajweave.async_buffer.enabled=true" in enabled


def test_marti_mars2_vanilla_grpo_overrides_use_independent_roots_without_correction():
    overrides = build_marti_mars2_launch_overrides(
        {
            "marti_mars2": {
                "max_num_nodes": 3,
                "initial_candidates": 3,
                "search_mode": "vanilla_grpo",
                "agent_loop_backend": "vllm_marti_tq",
                "enable_vllm_is_correction": False,
            },
            "credit": {"mode": "fidelity"},
            "verl": {"overrides": []},
        },
        config_path="configs/marti_mars2/vanilla_grpo_verl_vllm_baseline.yaml",
    )

    assert "+agent.orchestra.marti_mars2.initial_candidates=3" in overrides
    assert "+agent.orchestra.marti_mars2.search_mode=vanilla_grpo" in overrides
    assert "+trajweave.coordination_protocol=independent_group_rollouts" in overrides
    assert "+trajweave.credit_allocator=vanilla_group_grpo" in overrides
    assert "+trajweave.correction_hook.enabled=false" in overrides
    assert not any(item.startswith("algorithm.rollout_correction.rollout_is=") for item in overrides)


def test_marti_mars2_multi_agent_overrides_preserve_agent_policy_bindings():
    overrides = build_marti_mars2_launch_overrides(
        {
            "marti_mars2": {
                "max_num_nodes": 4,
                "initial_candidates": 2,
                "agent_ids": ["generator", "critic"],
                "model_ids": ["policy_a", "policy_b"],
                "agent_loop_backend": "synthetic_tq",
                "multi_actor_training": True,
            },
            "verl": {"overrides": []},
        },
        config_path="configs/marti_mars2/stage1e_multi_agent_contract.yaml",
    )

    assert '+agent.agent_ids=["generator","critic"]' in overrides
    assert '+agent.model_ids=["policy_a","policy_b"]' in overrides
    assert "+agent.model_sharing=false" in overrides
    assert '+agent.orchestra.marti_mars2.agent_ids=["generator","critic"]' in overrides
    assert '+agent.orchestra.marti_mars2.model_ids=["policy_a","policy_b"]' in overrides
    assert "trainer.v1.trainer_mode=trajweave_multi_actor_sync" in overrides
    assert "+trajweave.multi_actor.enabled=true" in overrides
    assert '+agent.worker_group_ids=["policy_a","policy_b"]' in overrides


def test_marti_mars2_rejects_multi_actor_vllm_without_explicit_lifecycle_opt_in():
    with pytest.raises(ValueError, match="multi_actor_vllm.enabled=true"):
        build_marti_mars2_launch_overrides(
            {
                "marti_mars2": {
                    "max_num_nodes": 3,
                    "agent_ids": ["generator", "critic"],
                    "model_ids": ["policy_a", "policy_b"],
                    "agent_loop_backend": "vllm_marti_tq",
                    "multi_actor_training": True,
                },
                "verl": {"overrides": []},
            },
            config_path=None,
        )


def test_marti_mars2_multi_actor_vllm_lifecycle_opt_in_builds_routing_overrides():
    overrides = build_marti_mars2_launch_overrides(
        {
            "marti_mars2": {
                "max_num_nodes": 2,
                "agent_ids": ["generator", "critic"],
                "model_ids": ["policy_a", "policy_b"],
                "agent_loop_backend": "vllm_marti_tq",
                "multi_actor_training": True,
                "multi_actor_vllm": {"enabled": True},
            },
            "verl": {"overrides": []},
        },
        config_path=None,
    )

    assert "+trajweave.multi_actor.vllm.enabled=true" in overrides


def test_stage1e_native_vllm_async_smoke_preserves_launch_contract():
    config_path = Path("configs/marti_mars2/stage1e_multi_agent_vllm_async_smoke.yaml")
    config = load_yaml_config(config_path)

    overrides = build_marti_mars2_launch_overrides(config, config_path=str(config_path))

    assert "trainer.v1.trainer_mode=trajweave_multi_actor_sync" in overrides
    assert "+trajweave.multi_actor.vllm.enabled=true" in overrides
    assert "+trajweave.async_buffer.enabled=true" in overrides
    assert "trainer.total_training_steps=3" in overrides


def test_marti_mars2_rejects_mismatched_multi_agent_policy_bindings():
    with pytest.raises(ValueError, match="same length"):
        build_marti_mars2_launch_overrides(
            {
                "marti_mars2": {
                    "max_num_nodes": 3,
                    "agent_ids": ["generator", "critic"],
                    "model_ids": ["shared"],
                },
                "verl": {"overrides": []},
            },
            config_path=None,
        )


def test_yaml_runner_builds_marti_mars2_verl_vllm_dryrun(tmp_path):
    command_file = tmp_path / "run_verl_ppo.sh"
    result = run_from_config(
        {
            "recipe": "marti_mars2.single_mcts.smoke",
            "mode": "verl_train",
            "run": {"root_dir": str(tmp_path), "name": "unit-marti-mars2-verl-dryrun"},
            "logging": {"console": False},
            "marti_mars2": {"max_num_nodes": 2, "agent_loop_backend": "synthetic_tq"},
            "verl": {
                "enabled": True,
                "execute": False,
                "command_file": str(command_file),
                "module": "trajweave.backends.verl.main_ppo",
                "overrides": ["actor_rollout_ref.rollout.name=vllm"],
            },
        }
    )

    assert result["verl_launch"]["status"] == "dry_run"
    assert result["marti_mars2"]["training_backend"] == "verl_v1_single_actor_wg"
    assert command_file.exists()
    command = command_file.read_text(encoding="utf-8")
    assert "actor_rollout_ref.rollout.name=vllm" in command
    assert "trajweave_marti_mars2_tree_grpo" in command
