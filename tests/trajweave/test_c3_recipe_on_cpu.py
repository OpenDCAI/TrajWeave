from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from hydra import compose, initialize_config_dir

from trajweave.backends.policy import PolicyRequest, PolicyResponse
from trajweave.backends.verl.extensions.c3 import C3ContextualCounterfactualHooks
from trajweave.backends.verl.multi_actor import reconcile_multi_actor_global_assets
from trajweave.credit.c3 import compute_c3_scalar_credit
from trajweave.envs.math import C3MathEnvironment, C3MathTask, MathTask, SolverVerifierMathEnvironment
from trajweave.orchestration.c3 import C3PrefixTreeOrchestra
from trajweave.recipes.c3 import default_c3_team, run_c3_smoke
from trajweave.recipes.c3.config import build_c3_launch_overrides, resolve_c3_settings
from trajweave.recipes.registry import resolve_recipe
from trajweave.runner import run_from_config
from verl import DataProto
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.workers.config.model import HFModelConfig


class _BranchBackend:
    def generate(self, request: PolicyRequest) -> PolicyResponse:
        branch = int(request.metadata["c3_branch_index"])
        depth = int(request.metadata["c3_depth"])
        if depth == 0:
            text = f"plan-{branch}"
        else:
            parent = str(request.metadata["c3_parent_node_id"])
            parent_branch = int(parent.split(":")[-1])
            answer = 4 if parent_branch == 0 and branch == 0 else 9
            text = f"Final answer: {answer}"
        return PolicyResponse(text=text, token_ids=[depth + 1, branch + 3], logprobs=[0.0, 0.0])


def test_c3_registry_configs_and_launch_contract_are_exposed():
    recipe = resolve_recipe("c3.math")
    assert recipe.name == "c3.reasoner_actor_math"
    assert recipe.family == "c3"
    assert recipe.runtime_recipe == "c3_reasoner_actor_math"

    config_path = Path("configs/c3/reasoner_actor_math_verl_tiny.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings = resolve_c3_settings(config)
    overrides = build_c3_launch_overrides(config, config_path=str(config_path))
    assert settings["fanout"] == (2, 2)
    assert settings["credit_variant"] == "value_assisted"
    assert "trainer.v1.trainer_mode=trajweave_c3_critic_sync" in overrides
    assert "critic.enable=true" in overrides
    assert any("C3ContextualCounterfactualHooks" in item for item in overrides)
    assert any("trajweave_c3_contextual_counterfactual" in item for item in overrides)
    assert "actor_rollout_ref.rollout.n=1" in overrides
    assert "actor_rollout_ref.rollout.val_kwargs.n=1" in overrides
    assert "+trajweave.hf_local_model_cache_size=2" in overrides
    actor_critic_override = next(item for item in overrides if "actor_critic=" in item)
    assert 'topology:"centralized"' in actor_critic_override
    assert 'critic_type:"q"' in actor_critic_override
    assert 'actor_groups:["reasoner","actor"]' in actor_critic_override
    assert "value_loss_coef:1.0" in actor_critic_override
    hook_override = next(item for item in overrides if "extension_hooks_class=" in item)
    hook_fqn = hook_override.split("=", 1)[1]
    from verl.trainer.ppo.v1.extension_hooks import get_ppo_v1_extension_hooks

    assert isinstance(
        get_ppo_v1_extension_hooks({"algorithm": {"extension_hooks_class": hook_fqn}}),
        C3ContextualCounterfactualHooks,
    )


def test_c3_verl_plan_really_composes_with_safe_runtime_defaults(monkeypatch):
    config_path = Path("configs/c3/reasoner_actor_math_verl_tiny.yaml")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    overrides = build_c3_launch_overrides(raw, config_path=str(config_path))
    hydra_root = Path("verl/trainer/config").resolve()

    with initialize_config_dir(config_dir=str(hydra_root), version_base=None):
        config = compose(config_name="ppo_trainer", overrides=list(overrides))
    reconcile_multi_actor_global_assets(config)

    assert config.trainer.v1.trainer_mode == "trajweave_c3_critic_sync"
    assert need_critic(config) is True
    assert need_reference_policy(config) is False
    assert config.actor_rollout_ref.model.path == "/path/to/reasoner_actor_model"
    assert config.actor_rollout_ref.model.tokenizer_path == "/path/to/tokenizer"
    assert config.critic.model.path == "/path/to/q_critic"
    assert config.critic.model.tokenizer_path == "/path/to/tokenizer"
    assert config.actor_rollout_ref.rollout.name == "hf"
    assert config.actor_rollout_ref.rollout.calculate_log_probs is False
    assert config.actor_rollout_ref.rollout.agent.num_workers == 1
    assert config.actor_rollout_ref.model.override_config.attn_implementation == "eager"
    assert config.actor_rollout_ref.model.use_remove_padding is False
    assert config.actor_rollout_ref.model.enable_gradient_checkpointing is False
    assert config.actor_rollout_ref.actor.use_torch_compile is False
    assert config.actor_rollout_ref.actor.fsdp_config.use_torch_compile is False
    assert config.critic.model.override_config.attn_implementation == "eager"
    assert config.critic.model.use_remove_padding is False
    assert config.critic.model.enable_gradient_checkpointing is False
    assert config.data.train_batch_size == 3
    assert config.trainer.nnodes == 1
    assert config.trainer.n_gpus_per_node == 3
    assert config.trainer.total_training_steps == 1
    assert config.trajweave.hf_local_model_cache_size == 2

    # VERL eagerly instantiates model configs during validation. The sample plan
    # deliberately uses placeholder paths, so suppress only that filesystem load.
    monkeypatch.setattr(HFModelConfig, "__post_init__", lambda self: None)
    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )


def test_c3_launch_padding_tracks_actor_gpu_world_size():
    config = {
        "c3": {
            "credit_variant": "reward_only",
            "model_ids": ["reasoner", "actor"],
            "worker_groups": {
                "reasoner": {"model_path": "/r", "tokenizer_path": "/t", "gpus": 3},
                "actor": {"model_path": "/a", "tokenizer_path": "/t", "gpus": 2},
            },
        }
    }
    overrides = build_c3_launch_overrides(config, config_path=None)
    assert "+trajweave.turn_padding_multiple=6" in overrides


@pytest.mark.parametrize(
    ("config", "error"),
    [
        ({"c3": {"credit_variant": "reward_only", "fanout": [2.5, 2]}}, "c3.fanout\\[0\\]"),
        ({"c3": {"credit_variant": "reward_only", "fanout": [2.0, 2]}}, "c3.fanout\\[0\\]"),
        (
            {
                "c3": {
                    "credit_variant": "reward_only",
                    "worker_groups": {"reasoner": {"gpus": 1.5}},
                }
            },
            "c3.worker_groups.reasoner.gpus",
        ),
        (
            {
                "c3": {
                    "credit_variant": "value_assisted",
                    "critic": {"model_path": "/q", "tokenizer_path": "/t", "gpus": 1.5},
                }
            },
            "c3.critic.gpus",
        ),
        (
            {
                "c3": {
                    "credit_variant": "value_assisted",
                    "critic": {"model_path": "/q", "tokenizer_path": "/t", "max_length": 12.5},
                }
            },
            "c3.critic.max_length",
        ),
    ],
)
def test_c3_config_rejects_fractional_integer_fields(config, error):
    with pytest.raises(ValueError, match=error):
        build_c3_launch_overrides(config, config_path=None)


def test_c3_config_rejects_unknown_backend_and_synthetic_training():
    with pytest.raises(ValueError, match="agent_loop_backend must be"):
        resolve_c3_settings({"c3": {"agent_loop_backend": "bogus", "credit_variant": "reward_only"}})
    with pytest.raises(ValueError, match="requires c3.agent_loop_backend=hf_local_tq"):
        resolve_c3_settings(
            {
                "mode": "verl_train",
                "c3": {"agent_loop_backend": "synthetic_tq", "credit_variant": "reward_only"},
            }
        )
    with pytest.raises(ValueError, match="requires c3.agent_loop_backend=hf_local_tq"):
        resolve_c3_settings(
            {
                "run": {"mode": "verl_train"},
                "c3": {"agent_loop_backend": "synthetic_tq", "credit_variant": "reward_only"},
            }
        )


def test_c3_runtime_rejects_fractional_fanout():
    from trajweave.backends.verl.emitters.c3 import C3EmitterMixin

    worker = C3EmitterMixin()
    worker.config = SimpleNamespace(trajweave=SimpleNamespace(c3=SimpleNamespace(fanout=[2.5, 2])))
    with pytest.raises(ValueError, match="integers of at least 2"):
        worker._c3_fanout()

    with pytest.raises(ValueError, match="integer fanout"):
        C3PrefixTreeOrchestra(fanout=(2.5, 2)).run_tree(
            episode_id="invalid",
            rollout_group="invalid",
            task=MathTask(task_id="invalid", question="What is 2 + 2?", answer=4),
            team=default_c3_team(),
            observation="What is 2 + 2?",
            policy_backend=_BranchBackend(),
            environment=SolverVerifierMathEnvironment(),
        )


def test_c3_rule_smoke_reports_the_backend_it_actually_used(tmp_path):
    result = run_from_config(
        {
            "recipe": "c3.reasoner_actor_math",
            "mode": "smoke",
            "run": {"root_dir": str(tmp_path / "runs")},
            "logging": {"console": False},
            "c3": {"fanout": [2, 2], "credit_variant": "reward_only"},
            "backend": {"type": "rule", "device": "cpu"},
        }
    )

    assert result["c3"]["smoke_backend"] == "rule"
    assert result["c3"]["training_backend"] == "none"
    assert "agent_loop_backend" not in result["c3"]
    assert "hf_local_model_cache_size" not in result["c3"]


def test_c3_verl_summary_keeps_the_actual_agent_loop_backend(tmp_path):
    result = run_from_config(
        {
            "recipe": "c3.reasoner_actor_math",
            "mode": "verl_plan",
            "run": {"root_dir": str(tmp_path / "runs")},
            "logging": {"console": False},
            "c3": {"credit_variant": "reward_only", "agent_loop_backend": "synthetic_tq"},
            "verl": {"enabled": True, "execute": False},
        }
    )

    assert result["c3"]["agent_loop_backend"] == "synthetic_tq"
    assert result["c3"]["training_backend"] == "verl_v1_multi_actor_q_critic"
    assert result["verl_launch"]["status"] == "dry_run"


def test_c3_prefix_tree_freezes_sibling_context_and_materializes_subtree_returns():
    team = default_c3_team()
    trajectory = C3PrefixTreeOrchestra(fanout=(2, 2)).run_tree(
        episode_id="episode",
        rollout_group="task",
        task=MathTask(task_id="task", question="What is 2 + 2?", answer=4),
        team=team,
        observation="What is 2 + 2?",
        policy_backend=_BranchBackend(),
        environment=SolverVerifierMathEnvironment(),
    )
    assert len(trajectory.turns) == 6
    groups = {}
    for turn in trajectory.turns:
        groups.setdefault(turn.metadata["c3_group_id"], []).append(turn)
    assert sorted(len(rows) for rows in groups.values()) == [2, 2, 2]
    for rows in groups.values():
        assert len({row.prompt for row in rows}) == 1
        assert len({row.parent_node_id for row in rows}) == 1
    reasoners = [turn for turn in trajectory.turns if turn.agent_name == "reasoner"]
    assert sorted(turn.metadata["c3_leaf_count"] for turn in reasoners) == [2, 2]
    assert sorted(turn.metadata["c3_subtree_return"] for turn in reasoners) == [0.0, 0.5]
    assert trajectory.success is True
    assert trajectory.final_answer == "Final answer: 4"


def test_c3_rule_b_credit_matches_reward_value_and_value_assisted_variants():
    kwargs = {
        "subtree_returns": [1.0, 0.0, 0.25, 0.75],
        "group_ids": ["a", "a", "b", "b"],
        "baseline_mode": "loo",
        "normalize": False,
    }
    reward = compute_c3_scalar_credit(**kwargs, variant="reward_only")
    assert reward.advantages == pytest.approx([1.0, -1.0, -0.5, 0.5])
    value = compute_c3_scalar_credit(**kwargs, q_values=[0.2, 0.8, 0.5, 0.5], variant="value_only")
    assert value.advantages == pytest.approx([-0.6, 0.6, 0.0, 0.0])
    assisted = compute_c3_scalar_credit(
        **kwargs,
        q_values=[0.2, 0.8, 0.5, 0.5],
        variant="value_assisted",
        value_assisted_alpha=0.5,
    )
    assert assisted.baselines == pytest.approx([0.4, 0.6, 0.625, 0.375])
    assert assisted.advantages == pytest.approx([0.6, -0.6, -0.375, 0.375])


def test_c3_credit_fails_fast_for_singleton_loo_and_missing_q():
    with pytest.raises(ValueError, match="at least two"):
        compute_c3_scalar_credit(subtree_returns=[1.0], group_ids=["only"], baseline_mode="loo")
    with pytest.raises(ValueError, match="requires one q_value"):
        compute_c3_scalar_credit(
            subtree_returns=[1.0, 0.0],
            group_ids=["g", "g"],
            variant="value_assisted",
        )


def test_c3_smoke_runs_real_prefix_tree_and_emits_credit_samples():
    summary, result = run_c3_smoke(fanout=(2, 2), normalize=False)
    assert summary.trajectories == 2
    assert summary.samples == 12
    assert summary.success_rate == 1.0
    assert {sample.agent_name for sample in result.samples} == {"reasoner", "actor"}
    assert all(sample.metadata["credit"] == "c3_contextual_counterfactual" for sample in result.samples)
    assert all(sample.metadata["c3_group_size"] == 2 for sample in result.samples)


def test_c3_smoke_uses_c3_math_tasks_and_string_answers(monkeypatch):
    observed: list[tuple[type[object], type[object]]] = []
    original_evaluate = C3MathEnvironment.evaluate

    def recording_evaluate(self, task, final_answer):
        observed.append((type(task), type(task.answer)))
        return original_evaluate(self, task, final_answer)

    monkeypatch.setattr(C3MathEnvironment, "evaluate", recording_evaluate)
    summary, _result = run_c3_smoke(fanout=(2, 2), normalize=False)

    assert summary.success_rate == 1.0
    assert observed
    assert set(observed) == {(C3MathTask, str)}


def test_c3_reward_only_hook_uses_subtree_returns_and_true_sibling_groups():
    data = DataProto.from_dict(
        tensors={
            "response_mask": torch.ones(4, 1),
            "token_level_rewards": torch.zeros(4, 1),
        },
        non_tensors={
            "uid": np.array(["task"] * 4, dtype=object),
            "c3_group_id": np.array(["a", "a", "b", "b"], dtype=object),
            "c3_depth": np.array([0, 0, 1, 1], dtype=object),
            "c3_role_index": np.array([0, 0, 1, 1], dtype=object),
            "c3_parent_id": np.array(["", "", "p", "p"], dtype=object),
            "c3_is_leaf": np.array([False, False, True, True], dtype=object),
            "c3_prefix_text": np.array(["a0", "a1", "b0", "b1"], dtype=object),
            "c3_subtree_return": np.array([1.0, 0.0, 0.25, 0.75], dtype=object),
            "c3_leaf_count": np.array([2, 2, 1, 1], dtype=object),
        },
    )
    result = C3ContextualCounterfactualHooks().compute_advantage(
        data,
        config={"c3": {"credit_variant": "reward_only", "baseline_mode": "loo", "normalize_advantages": False}},
    )
    torch.testing.assert_close(result.batch["advantages"].squeeze(-1), torch.tensor([1.0, -1.0, -0.5, 0.5]))
    torch.testing.assert_close(result.batch["returns"].squeeze(-1), torch.tensor([1.0, 0.0, 0.25, 0.75]))


def test_c3_value_smoke_requires_q_scorer_and_config_requires_critic_paths():
    with pytest.raises(ValueError, match="require a configured VERL Q critic"):
        run_c3_smoke(variant="value_assisted")
    with pytest.raises(ValueError, match="requires c3.critic"):
        resolve_c3_settings({"c3": {"credit_variant": "value_assisted", "fanout": [2, 2]}})
    with pytest.raises(ValueError, match="value_loss_coef must be positive and finite"):
        resolve_c3_settings(
            {
                "c3": {
                    "credit_variant": "value_assisted",
                    "fanout": [2, 2],
                    "critic": {
                        "model_path": "/critic",
                        "tokenizer_path": "/tokenizer",
                        "value_loss_coef": 0.0,
                    },
                }
            }
        )


def test_c3_weighted_soft_target_bce_matches_explicit_leaf_views_and_backpropagates():
    import torch.nn.functional as F
    from tensordict import TensorDict

    from trajweave.backends.verl.trainers.c3_critic_sync import verl_c3_weighted_bce_critic_loss
    from verl.utils import tensordict_utils as tu

    logits = torch.nn.Parameter(torch.tensor([-0.4, 0.7]))
    targets = torch.tensor([0.25, 0.5])
    weights = torch.tensor([4.0, 2.0])
    data = TensorDict(
        {
            "returns": targets,
            "loss_mask": torch.ones(2, dtype=torch.bool),
            "c3_leaf_weights": weights,
        },
        batch_size=2,
    )
    tu.assign_non_tensor(
        data,
        c3_total_leaf_weight=6.0,
        c3_positive_leaf_weight=2.0,
        dp_size=1,
    )

    loss, metrics = verl_c3_weighted_bce_critic_loss(None, {"values": logits}, data)
    prior = (2.0 + 0.5) / (6.0 + 1.0)
    bias = torch.logit(torch.tensor(prior))
    repeated_logits = torch.stack([logits[0]] * 4 + [logits[1]] * 2)
    repeated_targets = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    expected = F.binary_cross_entropy_with_logits(repeated_logits + bias, repeated_targets)

    torch.testing.assert_close(loss, expected)
    assert metrics["critic/pos_prior"] == pytest.approx(prior)
    assert metrics["critic/logit_bias"] == pytest.approx(float(bias))
    loss.backward()
    assert logits.grad is not None
    assert bool(torch.isfinite(logits.grad).all())
    assert bool((logits.grad != 0.0).all())


def test_c3_weighted_bce_dp_scaling_matches_global_expanded_leaf_mean():
    import torch.nn.functional as F
    from tensordict import TensorDict

    from trajweave.backends.verl.trainers.c3_critic_sync import verl_c3_weighted_bce_critic_loss
    from verl.utils import tensordict_utils as tu

    logits = torch.tensor([-0.4, 0.7])
    targets = torch.tensor([0.25, 0.5])
    weights = torch.tensor([4.0, 2.0])
    local_losses = []
    for rank in range(2):
        local_data = TensorDict(
            {
                "returns": targets[rank : rank + 1],
                "loss_mask": torch.ones(1, dtype=torch.bool),
                "c3_leaf_weights": weights[rank : rank + 1],
            },
            batch_size=1,
        )
        tu.assign_non_tensor(
            local_data,
            c3_total_leaf_weight=6.0,
            c3_positive_leaf_weight=2.0,
            dp_size=2,
        )
        local_loss, _ = verl_c3_weighted_bce_critic_loss(
            None,
            {"values": logits[rank : rank + 1]},
            local_data,
        )
        local_losses.append(local_loss)

    prior_bias = torch.logit(torch.tensor((2.0 + 0.5) / (6.0 + 1.0)))
    expanded_logits = torch.tensor([-0.4] * 4 + [0.7] * 2)
    expanded_targets = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    expected = F.binary_cross_entropy_with_logits(expanded_logits + prior_bias, expanded_targets)
    torch.testing.assert_close(torch.stack(local_losses).mean(), expected)


def test_c3_worker_setup_replaces_parent_mse_with_c3_bce(monkeypatch):
    from trajweave.backends.verl.extensions.comlrl.actor_critic import verl_unclipped_mse_critic_loss
    from trajweave.backends.verl.trainers.c3_critic_sync import (
        TrajWeaveC3CriticSyncTrainer,
        verl_c3_weighted_bce_critic_loss,
    )
    from trajweave.backends.verl.trainers.multi_actor_critic_sync import TrajWeaveMultiActorCriticSyncTrainer

    class _Worker:
        loss_fn = None

        def set_loss_fn(self, loss_fn):
            self.loss_fn = loss_fn

    worker = _Worker()

    def fake_parent_setup(trainer):
        trainer.critic_wgs = {"critic": worker}
        trainer._critic_cfgs = {"critic": object()}

    monkeypatch.setattr(TrajWeaveMultiActorCriticSyncTrainer, "_create_worker_groups", fake_parent_setup)
    trainer = object.__new__(TrajWeaveC3CriticSyncTrainer)
    trainer.config = {"trajweave": {"actor_critic": {"value_loss_coef": 1.0}}}
    trainer._create_worker_groups()

    assert worker.loss_fn.func is verl_c3_weighted_bce_critic_loss
    assert worker.loss_fn.func is not verl_unclipped_mse_critic_loss
    assert worker.loss_fn.keywords["value_loss_coef"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("targets", "weights", "error", "message"),
    [
        ([1.1, 0.0], [1.0, 1.0], ValueError, r"targets must be in \[0, 1\]"),
        ([float("nan"), 0.0], [1.0, 1.0], FloatingPointError, "targets must be finite"),
        ([1.0, 0.0], [0.0, 1.0], ValueError, "leaf weights must be positive"),
        ([1.0, 0.0], [float("inf"), 1.0], FloatingPointError, "leaf weights must be finite"),
    ],
)
def test_c3_bce_rejects_invalid_targets_and_leaf_weights(targets, weights, error, message):
    from tensordict import TensorDict

    from trajweave.backends.verl.trainers.c3_critic_sync import verl_c3_weighted_bce_critic_loss
    from verl.utils import tensordict_utils as tu

    data = TensorDict(
        {
            "returns": torch.tensor(targets),
            "loss_mask": torch.ones(2, dtype=torch.bool),
            "c3_leaf_weights": torch.tensor(weights),
        },
        batch_size=2,
    )
    tu.assign_non_tensor(data, c3_total_leaf_weight=2.0, c3_positive_leaf_weight=1.0, dp_size=1)
    with pytest.raises(error, match=message):
        verl_c3_weighted_bce_critic_loss(None, {"values": torch.zeros(2)}, data)


def test_c3_bce_rejects_non_finite_logits():
    from tensordict import TensorDict

    from trajweave.backends.verl.trainers.c3_critic_sync import verl_c3_weighted_bce_critic_loss
    from verl.utils import tensordict_utils as tu

    data = TensorDict(
        {
            "returns": torch.tensor([1.0, 0.0]),
            "loss_mask": torch.ones(2, dtype=torch.bool),
            "c3_leaf_weights": torch.ones(2),
        },
        batch_size=2,
    )
    tu.assign_non_tensor(data, c3_total_leaf_weight=2.0, c3_positive_leaf_weight=1.0, dp_size=1)
    with pytest.raises(FloatingPointError, match="logits must be finite"):
        verl_c3_weighted_bce_critic_loss(None, {"values": torch.tensor([float("inf"), 0.0])}, data)


def test_c3_trainer_registration_and_value_assisted_prefix_targets_use_real_tq():
    import uuid

    tq = pytest.importorskip("transfer_queue")
    from transfer_queue import KVBatchMeta

    from trajweave.backends.verl.multi_actor.critic_config import normalize_critic_route_specs
    from trajweave.backends.verl.trainers import register_trajweave_trainers
    from trajweave.backends.verl.trainers.c3_critic_sync import _materialize_c3_critic_tq_batch
    from verl.trainer.ppo.v1.trainer_base import get_trainer_cls
    from verl.utils.tensordict_utils import list_of_dict_to_tensordict

    register_trajweave_trainers()
    trainer_cls = get_trainer_cls("trajweave_c3_critic_sync")
    assert trainer_cls.__name__ == "TrajWeaveC3CriticSyncTrainer"

    [route] = normalize_critic_route_specs(
        [
            {
                "id": "c3-q-critic",
                "actor_groups": ["reasoner", "actor"],
                "topology": "centralized",
                "critic_type": "q",
                "model_path": "/models/critic",
                "tokenizer_path": "/tokenizer",
                "gpus": 1,
                "max_length": 32,
            }
        ],
        trainable_actor_groups=["reasoner", "actor"],
    )
    trainer = object.__new__(trainer_cls)
    trainer.config = {
        "trajweave": {
            "c3": {
                "credit_variant": "value_assisted",
                "baseline_mode": "loo",
                "value_assisted_alpha": 1.0,
                "normalize_advantages": False,
            }
        }
    }
    trainer.critic_route_specs = {route.critic_group: route}

    class _FakeCriticWorker:
        world_size = 1

        @staticmethod
        def infer_batch(critic_batch):
            critic_fields = tq.kv_batch_get(
                keys=critic_batch.keys,
                partition_id=critic_batch.partition_id,
                select_fields=["loss_mask"],
            )
            inferred = []
            for row, value in enumerate(q_logits):
                mask = torch.as_tensor(critic_fields["loss_mask"][row], dtype=torch.float32)
                inferred.append({"values": mask * value})
            tq.kv_batch_put(
                keys=critic_batch.keys,
                partition_id=critic_batch.partition_id,
                fields=list_of_dict_to_tensordict(inferred),
            )
            return None

    trainer.critic_wgs = {route.critic_group: _FakeCriticWorker()}
    trainer._critic_tokenizers = {route.tokenizer_path: object()}

    returns = [1.0, 0.0, 0.25, 0.75]
    q_logits = [-2.0, 2.0, 0.0, 0.0]
    rows = []
    for row, (group_id, subtree_return) in enumerate(zip(["a", "a", "b", "b"], returns, strict=True)):
        rows.append(
            {
                "worker_group": "reasoner" if row < 2 else "actor",
                "agent_id": "reasoner" if row < 2 else "actor",
                "traj_uid": "trajectory",
                "turn_id": row // 2,
                "response_mask": torch.ones((row % 2) + 1, dtype=torch.long),
                "node_id": f"node-{row}",
                "c3_group_id": group_id,
                "c3_depth": row // 2,
                "c3_role_index": row // 2,
                "c3_parent_id": "" if row < 2 else "reasoner-parent",
                "c3_is_leaf": row >= 2,
                "c3_prefix_text": f"Question: 2 + 2\nPrefix {row}",
                "c3_subtree_return": subtree_return,
                "c3_leaf_count": 2 if row < 2 else 1,
                "critic_group": route.critic_group,
                "critic_type": "q",
                "critic_input_ids": [10 + row, 20 + row],
                "critic_attention_mask": [1, 1],
                "critic_position_ids": [0, 1],
                "critic_loss_mask": 1.0,
            }
        )

    suffix = uuid.uuid4().hex
    keys = [f"c3-trainer-{suffix}-{row}" for row in range(len(rows))]
    tq.init()
    critic_batch = None
    try:
        tq.kv_batch_put(
            keys=keys,
            partition_id="train",
            fields=list_of_dict_to_tensordict(rows),
            tags=[{"seq_len": 2} for _ in rows],
        )
        batch = KVBatchMeta(keys=keys, tags=[{} for _ in rows], partition_id="train")
        metrics = {}
        trainer._compute_values(batch, metrics)
        value_output = tq.kv_batch_get(
            keys=keys,
            partition_id="train",
            select_fields=["old_values", "values"],
        )
        q_probabilities = torch.sigmoid(torch.tensor(q_logits))
        torch.testing.assert_close(torch.as_tensor(value_output["old_values"]), q_probabilities)
        nested_values = value_output["values"]
        for row, expected in enumerate(q_probabilities):
            active_values = torch.as_tensor(nested_values[row])
            torch.testing.assert_close(active_values, torch.full_like(active_values, expected))

        trainer._compute_advantage(batch, metrics)
        output = tq.kv_batch_get(
            keys=keys,
            partition_id="train",
            select_fields=["advantages", "returns", "critic_returns"],
        )
        from trajweave.backends.verl.trainers.multi_actor_critic_sync import _unpack_advantage_tq_field

        advantages = _unpack_advantage_tq_field(output["advantages"])
        actor_returns = _unpack_advantage_tq_field(output["returns"])
        expected_advantages = torch.tensor([1.0 - q_probabilities[1], -q_probabilities[0], -0.25, 0.25])
        torch.testing.assert_close(advantages[:, 0], expected_advantages)
        assert float(advantages[0, 0]) > 0.0
        assert float(advantages[1, 0]) < 0.0
        torch.testing.assert_close(actor_returns[:, 0], torch.tensor(returns))
        torch.testing.assert_close(torch.as_tensor(output["critic_returns"]), torch.tensor(returns))
        assert metrics == {"trajweave/c3/prefix_rows": 4, "trajweave/c3/sibling_groups": 2}

        [routed] = trainer._route_critic_batch(batch)
        assert routed.batch.keys == keys  # Every prefix node is a distinct Q target; do not apply MAAC dedupe.
        critic_batch = _materialize_c3_critic_tq_batch(routed.batch)
        critic_data = tq.kv_batch_get(
            keys=critic_batch.keys,
            partition_id="train",
            select_fields=["returns", "loss_mask", "response_mask", "c3_leaf_weights"],
        )
        assert torch.as_tensor(critic_data["loss_mask"][0]).dtype == torch.int64
        assert torch.as_tensor(critic_data["response_mask"][0]).dtype == torch.int64
        materialized_targets = []
        materialized_weights = []
        for row in range(len(rows)):
            loss_mask = torch.as_tensor(critic_data["loss_mask"][row]).bool()
            materialized_targets.append(float(torch.as_tensor(critic_data["returns"][row])[loss_mask].item()))
            materialized_weights.append(float(torch.as_tensor(critic_data["c3_leaf_weights"][row])[loss_mask].item()))
        assert materialized_targets == pytest.approx(returns)
        assert materialized_weights == pytest.approx([2.0, 2.0, 1.0, 1.0])
        assert critic_batch.extra_info["c3_total_leaf_weight"] == pytest.approx(6.0)
        assert critic_batch.extra_info["c3_positive_leaf_weight"] == pytest.approx(3.0)
    finally:
        if critic_batch is not None:
            tq.kv_clear(keys=critic_batch.keys, partition_id="train")
        tq.kv_clear(keys=keys, partition_id="train")
        tq.close()


def test_c3_agent_loop_marks_outer_rollout_repetition_as_failure(monkeypatch):
    import asyncio

    tq = pytest.importorskip("transfer_queue")
    from trajweave.backends.verl.agent_loop import TrajWeaveSyntheticAgentLoopWorkerTQ
    from trajweave.backends.verl.emitters.c3 import C3EmitterMixin

    actor_class = TrajWeaveSyntheticAgentLoopWorkerTQ.__ray_actor_class__

    class _Worker(C3EmitterMixin):
        _run_prompt = actor_class._run_prompt

        def __init__(self):
            self.config = SimpleNamespace(
                trajweave=SimpleNamespace(
                    recipe="c3_reasoner_actor_math",
                    agent_loop_backend="synthetic_tq",
                    c3=SimpleNamespace(fanout=[2, 2]),
                ),
                actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(n=2, val_kwargs=SimpleNamespace(n=1))),
            )

    statuses = []

    async def fake_put(*_args, **kwargs):
        statuses.append(kwargs["tag"])
        return None

    monkeypatch.setattr(tq, "async_kv_put", fake_put)
    asyncio.run(
        _Worker()._run_prompt(
            {
                "uid": "c3-invalid-rollout-n",
                "raw_prompt": [{"role": "user", "content": "What is 2 + 2?"}],
                "reward_model": {"ground_truth": "4"},
                "global_steps": 0,
            },
            trajectory={"validate": False},
        )
    )
    assert statuses[0] == {"status": "running"}
    assert statuses[-1]["status"] == "failure"
    assert statuses[-1]["error_type"] == "ValueError"
    assert "requires rollout.n=1" in statuses[-1]["error_message"]


def test_c3_trainer_rejects_actor_plus_critic_gpu_overcommit():
    from omegaconf import OmegaConf

    from trajweave.backends.verl.multi_actor.critic_config import normalize_critic_route_specs
    from trajweave.backends.verl.trainers.c3_critic_sync import TrajWeaveC3CriticSyncTrainer
    from trajweave.backends.verl.trainers.multi_actor_sync import WorkerGroupConfig

    [route] = normalize_critic_route_specs(
        [
            {
                "id": "critic",
                "actor_groups": ["reasoner", "actor"],
                "topology": "centralized",
                "critic_type": "q",
                "model_path": "/critic",
                "tokenizer_path": "/tokenizer",
                "gpus": 1,
            }
        ],
        trainable_actor_groups=["reasoner", "actor"],
    )
    trainer = object.__new__(TrajWeaveC3CriticSyncTrainer)
    trainer.config = OmegaConf.create(
        {
            "trajweave": {"multi_actor": {"tokenizer_mode": "shared"}},
            "actor_rollout_ref": {
                "actor": {"policy_loss": {"loss_mode": "vanilla_no_dual_clip"}},
                "model": {},
            },
            "trainer": {"nnodes": 1, "n_gpus_per_node": 2},
        }
    )
    trainer.use_reference_policy = False
    trainer.use_critic = True
    trainer.multi_actor_trainable_group_ids = ["reasoner", "actor"]
    trainer.multi_actor_worker_group_specs = {
        group_id: WorkerGroupConfig(
            group_id=group_id,
            role_key=f"actor-{group_id}",
            model_path=f"/{group_id}",
            tokenizer_path="/tokenizer",
            trainable=True,
            gpus=1,
        )
        for group_id in trainer.multi_actor_trainable_group_ids
    }
    trainer.critic_route_specs = {route.critic_group: route}
    trainer.critic_topology = "centralized"
    trainer.critic_type = "q"
    with pytest.raises(ValueError, match="prefix-Q critic.*requested=3, available=2"):
        trainer._validate_multi_actor_specs()


def test_c3_trainer_rejects_synthetic_rollout_even_when_recipe_validation_is_bypassed():
    from omegaconf import OmegaConf

    from trajweave.backends.verl.trainers.multi_actor_sync import (
        TrajWeaveMultiActorSyncTrainer,
        WorkerGroupConfig,
    )

    trainer = object.__new__(TrajWeaveMultiActorSyncTrainer)
    trainer.config = OmegaConf.create(
        {
            "trajweave": {
                "recipe": "c3_reasoner_actor_math",
                "agent_loop_backend": "synthetic_tq",
                "multi_actor": {"tokenizer_mode": "shared"},
            },
            "actor_rollout_ref": {"model": {}},
            "trainer": {"nnodes": 1, "n_gpus_per_node": 2},
        }
    )
    trainer.use_reference_policy = False
    trainer.multi_actor_trainable_group_ids = ["reasoner", "actor"]
    trainer.multi_actor_worker_group_specs = {
        group_id: WorkerGroupConfig(
            group_id=group_id,
            role_key=f"actor-{group_id}",
            model_path=f"/{group_id}",
            tokenizer_path="/tokenizer",
            trainable=True,
            gpus=1,
        )
        for group_id in trainer.multi_actor_trainable_group_ids
    }

    with pytest.raises(ValueError, match="requires trajweave.agent_loop_backend=hf_local_tq"):
        trainer._validate_multi_actor_specs()


def test_c3_agent_loop_writes_complete_nested_tree_through_real_tq():
    import asyncio
    import uuid

    tq = pytest.importorskip("transfer_queue")
    from trajweave.backends.verl.agent_loop import TrajWeaveSyntheticAgentLoopWorkerTQ
    from trajweave.backends.verl.emitters.c3 import C3EmitterMixin

    actor_class = TrajWeaveSyntheticAgentLoopWorkerTQ.__ray_actor_class__

    class _Tokenizer:
        pad_token_id = 0
        eos_token_id = 1

    class _Worker(C3EmitterMixin):
        _run_prompt = actor_class._run_prompt
        _put_outputs = actor_class._put_outputs

        def __init__(self):
            self.config = SimpleNamespace(
                agent=SimpleNamespace(model_ids=["reasoner-policy", "actor-policy"]),
                trajweave=SimpleNamespace(
                    recipe="c3_reasoner_actor_math",
                    agent_loop_backend="synthetic_tq",
                    turn_padding_multiple=2,
                    c3=SimpleNamespace(fanout=[2, 2], credit_variant="reward_only"),
                ),
                actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(n=1, val_kwargs=SimpleNamespace(n=1))),
            )
            self.rollout_config = SimpleNamespace(prompt_length=64, response_length=32, temperature=1.0)
            self.tokenizer = _Tokenizer()
            self.online_turn_rows = []

        @staticmethod
        def _encode_prompt(value):
            return [10, 11]

        @staticmethod
        def _encode_prompt_text(value):
            return [10, 11]

        @staticmethod
        def _encode_text(value):
            return [ord(char) % 127 for char in value] or [1]

        @staticmethod
        def _local_policy_version():
            return 0

        @staticmethod
        def _compute_multi_modal_inputs(output, input_ids):
            return None

        @staticmethod
        def _compute_position_ids(input_ids, attention_mask, multi_modal_inputs):
            return torch.arange(input_ids.shape[-1], dtype=torch.long).unsqueeze(0)

        @staticmethod
        def _attach_worker_group_stats(rows):
            return None

        def _write_online_turns(self, runtime, rows):
            self.online_turn_rows.extend(rows)

    tq.init()
    uid = uuid.uuid4().hex
    worker = _Worker()
    try:
        asyncio.run(
            worker._run_prompt(
                {
                    "uid": uid,
                    "raw_prompt": [{"role": "user", "content": "What is 2 + 2?"}],
                    "reward_model": {"ground_truth": "4"},
                    "global_steps": 0,
                },
                trajectory={"validate": False},
            )
        )
        metadata = tq.kv_list(partition_id="train")["train"]
        keys = sorted(key for key in metadata if key.startswith(f"{uid}_"))
        assert len(keys) == 6  # two reasoner rows and four actor rows; both routes are already aligned
        selected = (
            "response_mask",
            "worker_group",
            "node_id",
            "c3_group_id",
            "c3_parent_id",
            "c3_depth",
            "c3_prefix_text",
            "c3_subtree_return",
            "c3_leaf_count",
        )
        data = tq.kv_batch_get(keys=keys, partition_id="train", select_fields=selected)
        real = [row for row, mask in enumerate(data["response_mask"]) if bool(torch.as_tensor(mask).any())]
        assert len(real) == 6
        groups = [str(data["c3_group_id"][row]) for row in real]
        assert sorted(groups.count(group) for group in set(groups)) == [2, 2, 2]
        assert {str(data["worker_group"][row]) for row in real} == {"reasoner-policy", "actor-policy"}
        assert {int(data["c3_depth"][row]) for row in real} == {0, 1}
        assert all(str(data["c3_prefix_text"][row]).startswith("Question:") for row in real)
        assert all(int(data["c3_leaf_count"][row]) >= 1 for row in real)

        validation_uid = f"{uid}-validation"
        asyncio.run(
            worker._run_prompt(
                {
                    "uid": validation_uid,
                    "raw_prompt": [{"role": "user", "content": "What is 2 + 2?"}],
                    "reward_model": {"ground_truth": "4"},
                    "global_steps": 0,
                },
                trajectory={"validate": True},
            )
        )
        validation_metadata = tq.kv_list(partition_id="val")["val"]
        validation_keys = sorted(key for key in validation_metadata if key.startswith(f"{validation_uid}_"))
        # One real row per worker group, plus one zero-mask alignment row per group.
        assert len(validation_keys) == 4
        validation_data = tq.kv_batch_get(
            keys=validation_keys,
            partition_id="val",
            select_fields=["worker_group", "c3_is_leaf", "c3_depth", "response_mask"],
        )
        validation_real = [
            row for row, mask in enumerate(validation_data["response_mask"]) if bool(torch.as_tensor(mask).any())
        ]
        assert len(validation_real) == 2
        assert [str(validation_data["worker_group"][row]) for row in validation_real] == [
            "reasoner-policy",
            "actor-policy",
        ]
        assert [bool(validation_data["c3_is_leaf"][row]) for row in validation_real] == [False, True]
        assert [int(validation_data["c3_depth"][row]) for row in validation_real] == [0, 1]
    finally:
        for partition in ("train", "val"):
            keys = list(tq.kv_list(partition_id=partition).get(partition, {}).keys())
            owned = [
                key for key in keys if key == uid or key.startswith(f"{uid}_") or key.startswith(f"{uid}-validation_")
            ]
            if owned:
                tq.kv_clear(keys=owned, partition_id=partition)
        tq.close()
