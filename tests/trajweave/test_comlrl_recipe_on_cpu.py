from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from trajweave.backends.verl.multi_actor import reconcile_multi_actor_global_assets
from trajweave.backends.verl.trainers.comlrl_staged import prepare_marlhf_online_config
from trajweave.backends.verl.trainers.multi_actor_sync import (
    TrajWeaveMultiActorSyncTrainer,
    WorkerGroupConfig,
)
from trajweave.pipeline.config import load_yaml_config
from trajweave.recipes.comlrl.config import build_comlrl_launch_overrides
from trajweave.runner import run_from_config
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.workers.config.model import HFModelConfig


def _base_config(algorithm: str) -> dict:
    config = {
        "recipe": f"comlrl.{algorithm}",
        "mode": "verl_plan",
        "comlrl": {
            "algorithm": algorithm,
            "agent_ids": ["alice", "bob"],
            "model_ids": ["actor-a", "actor-b"],
            "joint_mode": "aligned",
            "max_turns": 1,
            "num_candidates": 4,
            "agent_loop_backend": "synthetic_tq",
            "worker_groups": {
                "actor-a": {
                    "model_path": "/models/a",
                    "tokenizer_path": "/tokenizer",
                    "gpus": 1,
                },
                "actor-b": {
                    "model_path": "/models/b",
                    "tokenizer_path": "/tokenizer",
                    "gpus": 1,
                },
            },
        },
        "prepare": {"tiny_verl_assets": {"enabled": False}},
        "verl": {"enabled": True, "execute": False, "overrides": ["trainer.use_v1=true"]},
    }
    if algorithm in {"iac", "maac"}:
        topology = "independent" if algorithm == "iac" else "centralized"
        routes = (
            [
                {
                    "critic_group": f"critic-{actor}",
                    "actor_groups": [actor],
                    "model_path": f"/models/critic-{actor}",
                    "tokenizer_path": "/tokenizer",
                    "gpus": 1,
                }
                for actor in ("actor-a", "actor-b")
            ]
            if algorithm == "iac"
            else [
                {
                    "critic_group": "team-critic",
                    "actor_groups": ["actor-a", "actor-b"],
                    "model_path": "/models/team-critic",
                    "tokenizer_path": "/tokenizer",
                    "gpus": 1,
                }
            ]
        )
        config["comlrl"]["actor_critic"] = {
            "topology": topology,
            "critic_type": "v",
            "critic_routes": routes,
        }
    return config


def test_multi_actor_init_defers_hf_weight_sync_until_agent_loop_exists():
    calls = []

    class CheckpointManager:
        @staticmethod
        def update_weights(global_steps):
            calls.append(global_steps)

    trainer = object.__new__(TrajWeaveMultiActorSyncTrainer)
    trainer.checkpoint_manager = CheckpointManager()
    trainer.global_steps = 3

    trainer.on_init_end()
    assert calls == []
    trainer.agent_loop_manager = object()
    trainer.on_init_end()
    assert calls == [3]


def test_multi_actor_tokenizer_init_uses_primary_group_assets(monkeypatch):
    import verl.utils
    import verl.utils.fs

    copied = []
    loaded = []

    def fake_copy(source, **_kwargs):
        copied.append(str(source))
        return f"/local{source}"

    monkeypatch.setattr(verl.utils.fs, "copy_to_local", fake_copy)
    monkeypatch.setattr(verl.utils, "hf_tokenizer", lambda source, **_kwargs: loaded.append(("tokenizer", source)))
    monkeypatch.setattr(verl.utils, "hf_processor", lambda source, **_kwargs: loaded.append(("processor", source)))

    trainer = object.__new__(TrajWeaveMultiActorSyncTrainer)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {
                    "path": "~/models/deepseek-llm-7b-chat",
                    "tokenizer_path": None,
                    "use_shm": False,
                }
            },
            "data": {"trust_remote_code": False},
        }
    )
    trainer.multi_actor_worker_group_specs = {
        "actor-a": WorkerGroupConfig(
            group_id="actor-a",
            role_key="actor-a",
            model_path="/models/actor-a",
            tokenizer_path="/tokenizers/shared",
            trainable=True,
            gpus=1,
        )
    }
    trainer.multi_actor_trainable_group_ids = ["actor-a"]

    trainer._align_global_actor_assets()
    trainer._init_tokenizer()

    assert trainer.config.actor_rollout_ref.model.path == "/models/actor-a"
    assert trainer.config.actor_rollout_ref.model.tokenizer_path == "/tokenizers/shared"
    assert copied == ["/tokenizers/shared"]
    assert loaded == [
        ("tokenizer", "/local/tokenizers/shared"),
        ("processor", "/local/tokenizers/shared"),
    ]


def test_multi_actor_auto_resume_allows_fresh_run_but_rejects_checkpoint(tmp_path):
    trainer = object.__new__(TrajWeaveMultiActorSyncTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "resume_mode": "auto",
                "resume_from_path": None,
                "default_local_dir": str(tmp_path),
            }
        }
    )

    trainer._load_checkpoint()
    assert trainer.global_steps == 0

    (tmp_path / "latest_checkpointed_iteration.txt").write_text("1")
    with pytest.raises(ValueError, match="checkpoint resume is not implemented"):
        trainer._load_checkpoint()


def test_madpo_standard_recipe_entrypoint_builds_complete_launch_contract():
    result = run_from_config(_base_config("madpo"))

    command = result["verl_launch"]["command"]
    assert "trainer.v1.trainer_mode=trajweave_joint_preference_sync" in command
    assert "+trajweave.recipe=comlrl_joint_math" in command
    assert "+trajweave.comlrl.algorithm=madpo" in command
    assert "+trajweave.credit_allocator=comlrl_madpo" in command
    assert "actor_rollout_ref.actor.ppo_epochs=1" in command
    assert '+agent.model_ids=["actor-a","actor-b"]' in command
    assert result["comlrl"]["source_version"] == "v1.4.1-5-g3c724af"
    assert result["comlrl"]["source_commit"] == "3c724afd"


def test_reinforce_recipe_uses_ratio_free_sequence_policy_gradient():
    command = run_from_config(_base_config("magrpo"))["verl_launch"]["command"]

    assert "actor_rollout_ref.actor.policy_loss.loss_mode=gpg" in command
    assert "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum" in command
    assert "actor_rollout_ref.actor.ppo_epochs=1" in command
    assert "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1" in command
    assert "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1" in command
    assert "actor_rollout_ref.rollout.calculate_log_probs=false" in command
    assert "critic.enable=false" in command
    assert not any("early_stop_threshold" in item for item in command)
    assert command.count('actor_rollout_ref.model.path="/models/a"') == 1
    assert command.count('actor_rollout_ref.model.tokenizer_path="/tokenizer"') == 1


def test_recipe_rejects_reference_kl_until_rollout_reference_models_are_wired():
    config = _base_config("magrpo")
    config["comlrl"]["sequence_kl_coefficient"] = 0.1

    with pytest.raises(ValueError, match="reference-policy KL is not wired"):
        run_from_config(config)


@pytest.mark.parametrize("algorithm", ["iac", "maac"])
def test_actor_critic_recipe_uses_source_loss_scale_and_learning_rate(algorithm):
    command = run_from_config(_base_config(algorithm))["verl_launch"]["command"]

    assert "actor_rollout_ref.actor.policy_loss.loss_mode=gpg" in command
    assert "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum" in command
    assert "actor_rollout_ref.actor.ppo_epochs=1" in command
    assert "critic.ppo_micro_batch_size_per_gpu=1" in command
    assert "critic.optim.lr=5e-06" in command
    assert any("value_loss_coef:0.6" in item for item in command)


@pytest.mark.parametrize("algorithm", ["iac", "maac"])
def test_actor_critic_verl_plan_requires_explicit_routed_critics(algorithm):
    config = _base_config(algorithm)
    config["comlrl"].pop("actor_critic")

    with pytest.raises(ValueError, match="critic route is required"):
        build_comlrl_launch_overrides(config, config_path="config.yaml")


def test_madpo_iter_recipe_builds_iterative_runtime_contract():
    config = _base_config("madpo_iter")
    config["comlrl"]["iterative"] = {
        "num_iterations": 3,
        "pair_selection": "comparator_reward",
        "pairs_per_sample": 2,
        "replay": {"mode": "nearest_k", "k": 2, "sample_size": 4},
        "comparator": {
            "policy": "history",
            "generation_mode": "decentralized",
            "num_candidates": 4,
            "history_k": 1,
        },
    }

    command = run_from_config(config)["verl_launch"]["command"]

    assert "trainer.v1.trainer_mode=trajweave_joint_preference_sync" in command
    assert "+trajweave.comlrl.algorithm=madpo" in command
    assert "+trajweave.comlrl.iterative.enabled=true" in command
    assert "+trajweave.comlrl.iterative.algorithm=madpo_iter" in command
    assert any("num_iterations" in item and "nearest_k" in item for item in command)


def test_marlhf_iter_recipe_builds_iterative_runtime_contract():
    config = _base_config("marlhf_iter")
    config["comlrl"]["marlhf"] = {
        "rl_algorithm": "magrpo",
        "reward_model_name": "/models/reward",
    }
    config["comlrl"]["iterative"] = {
        "num_iterations": 2,
        "comparator": {"policy": "current", "num_candidates": 4},
    }

    command = run_from_config(config)["verl_launch"]["command"]

    assert "+trajweave.comlrl.algorithm=magrpo" in command
    assert "+trajweave.comlrl.iterative.enabled=true" in command
    assert "+trajweave.comlrl.iterative.algorithm=marlhf_iter" in command
    assert "+trajweave.comlrl.marlhf.enabled=true" in command


def test_marlhf_standard_recipe_entrypoint_builds_staged_launch_contract():
    config = _base_config("marlhf")
    config["comlrl"]["marlhf"] = {
        "rl_algorithm": "magrpo",
        "reward_model_name": "/models/reward",
        "preference_num_candidates": 8,
        "preference_collection_batches": 3,
        "reward_freeze_backbone": True,
        "reward_learning_rate": 2.0e-5,
        "reward_torch_dtype": "bf16",
        "reward_num_train_epochs": 2,
        "reward_train_batch_size": 4,
        "reward_max_length": 512,
        "reward_model_checkpoint": "/checkpoints/reward.pt",
    }

    result = run_from_config(config)

    command = result["verl_launch"]["command"]
    assert "+trajweave.comlrl.algorithm=marlhf" in command
    assert "+trajweave.comlrl.marlhf.enabled=true" in command
    assert "+trajweave.comlrl.marlhf.rl_algorithm=magrpo" in command
    assert '+trajweave.comlrl.marlhf.reward_model_name="/models/reward"' in command
    assert "+trajweave.comlrl.marlhf.preference_num_candidates=8" in command
    assert "+trajweave.comlrl.marlhf.preference_collection_batches=3" in command
    assert "+trajweave.comlrl.marlhf.reward_freeze_backbone=true" in command
    assert "+trajweave.comlrl.marlhf.reward_learning_rate=2e-05" in command
    assert "+trajweave.comlrl.marlhf.reward_torch_dtype=bf16" in command
    assert "+trajweave.comlrl.marlhf.reward_num_train_epochs=2" in command
    assert "+trajweave.comlrl.marlhf.reward_train_batch_size=4" in command
    assert "+trajweave.comlrl.marlhf.reward_max_length=512" in command
    assert '+trajweave.comlrl.marlhf.reward_model_checkpoint="/checkpoints/reward.pt"' in command


def test_marlhf_iac_recipe_builds_single_candidate_critic_launch_contract():
    config = _base_config("marlhf")
    config["comlrl"]["marlhf"] = {
        "rl_algorithm": "iac",
        "reward_model_name": "/models/reward",
        "critic_model_name": "/models/critic",
        "critic_tokenizer_name": "/tokenizers/critic",
        "critic_type": "q",
        "critic_gpus": 1,
        "critic_max_length": 1024,
    }

    command = run_from_config(config)["verl_launch"]["command"]

    assert "trainer.v1.trainer_mode=trajweave_multi_actor_critic_sync" in command
    assert "actor_rollout_ref.rollout.n=1" in command
    assert '+trajweave.comlrl.marlhf.critic_model_name="/models/critic"' in command
    assert '+trajweave.comlrl.marlhf.critic_tokenizer_name="/tokenizers/critic"' in command
    assert "+trajweave.comlrl.marlhf.critic_type=q" in command
    assert "+trajweave.comlrl.marlhf.critic_gpus=1" in command
    assert "+trajweave.comlrl.marlhf.critic_max_length=1024" in command


@pytest.mark.parametrize(
    ("filename", "trainer_mode", "critic_required", "expected_batch_size"),
    [
        ("iac_verl_tiny.yaml", "trajweave_multi_actor_critic_sync", True, 4),
        ("maac_verl_tiny.yaml", "trajweave_multi_actor_critic_sync", True, 3),
        ("madpo_iter_verl_tiny.yaml", "trajweave_joint_preference_sync", False, 2),
        ("madpo_verl_tiny.yaml", "trajweave_joint_preference_sync", False, 2),
        ("magrpo_verl_tiny.yaml", "trajweave_multi_actor_sync", False, 2),
        ("mareinforce_verl_tiny.yaml", "trajweave_multi_actor_sync", False, 2),
        ("maremax_verl_tiny.yaml", "trajweave_multi_actor_sync", False, 2),
        ("marlhf_iter_verl_tiny.yaml", "trajweave_multi_actor_sync", False, 2),
        ("marlhf_verl_tiny.yaml", "trajweave_multi_actor_sync", False, 2),
        ("marloo_verl_tiny.yaml", "trajweave_multi_actor_sync", False, 2),
    ],
)
def test_all_comlrl_verl_plans_really_compose_with_safe_runtime_defaults(
    filename,
    trainer_mode,
    critic_required,
    expected_batch_size,
    monkeypatch,
):
    config_path = Path("configs/comlrl") / filename
    raw = load_yaml_config(config_path)
    overrides = build_comlrl_launch_overrides(raw, config_path=str(config_path))
    hydra_root = Path("verl/trainer/config").resolve()

    with initialize_config_dir(config_dir=str(hydra_root), version_base=None):
        config = compose(config_name="ppo_trainer", overrides=list(overrides))
    prepare_marlhf_online_config(config)
    reconcile_multi_actor_global_assets(config)

    assert config.trainer.v1.trainer_mode == trainer_mode
    assert need_critic(config) is critic_required
    assert config.actor_rollout_ref.model.path == "/path/to/actor_0"
    assert config.actor_rollout_ref.model.tokenizer_path == "/path/to/tokenizer"
    assert config.data.train_files == "/path/to/train.parquet"
    assert config.data.val_files == "/path/to/val.parquet"
    assert config.data.train_batch_size == expected_batch_size
    assert config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu == 1
    assert config.actor_rollout_ref.rollout.calculate_log_probs is False
    assert config.trainer.total_epochs == 1
    assert list(config.trainer.logger) == ["console"]
    if critic_required:
        expected_critic = "/path/to/critic_0" if filename.startswith("iac_") else "/path/to/centralized_critic"
        assert config.critic.model.path == expected_critic
        assert config.critic.model.tokenizer_path == "/path/to/tokenizer"
        assert config.critic.ppo_micro_batch_size_per_gpu == 1

    # VERL eagerly instantiates its singleton critic model during config validation.
    # The plan intentionally contains placeholders, so suppress only that filesystem load.
    monkeypatch.setattr(HFModelConfig, "__post_init__", lambda self: None)
    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )


def test_comlrl_package_import_is_order_independent():
    result = subprocess.run(
        [sys.executable, "-c", "import trajweave.recipes.comlrl"],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_comlrl_hydra_overrides_quote_paths_with_structural_characters():
    config = _base_config("magrpo")
    model_path = "/models/actor a:{revision},one"
    tokenizer_path = "/tokenizers/shared one:{revision},two"
    config["comlrl"]["worker_groups"]["actor-a"]["model_path"] = model_path
    for group in config["comlrl"]["worker_groups"].values():
        group["tokenizer_path"] = tokenizer_path
    overrides = build_comlrl_launch_overrides(
        config,
        config_path="/configs/CoMLRL plan:{draft},v1.yaml",
    )

    with initialize_config_dir(config_dir=str(Path("verl/trainer/config").resolve()), version_base=None):
        composed = compose(config_name="ppo_trainer", overrides=list(overrides))

    assert composed.actor_rollout_ref.model.path == model_path
    assert composed.actor_rollout_ref.model.tokenizer_path == tokenizer_path
    assert composed.trajweave.config == "/configs/CoMLRL plan:{draft},v1.yaml"
