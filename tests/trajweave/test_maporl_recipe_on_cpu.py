from types import SimpleNamespace

import pytest

from trajweave.backends.policy import PolicyResponse
from trajweave.envs.math import MathTask
from trajweave.orchestration.maporl_debate import MAPoRLDebateOrchestra
from trajweave.recipes.maporl import run_debate_math_smoke
from trajweave.recipes.maporl.debate_math import default_debate_team
from trajweave.runner import run_from_config


def test_maporl_smoke_assigns_full_ppo_turn_rewards():
    _summary, result = run_debate_math_smoke(
        backend="rule",
        agent_count=2,
        rollouts_per_task=1,
        max_turns=2,
        consensus_threshold=2,
        reward_feedback=True,
    )

    assert len(result.samples) == 4
    assert {sample.policy_group for sample in result.samples} == {"shared"}
    assert {sample.metadata["credit"] for sample in result.samples} == {"maporl_ppo_score_rule"}
    assert {sample.metadata["raw_score"] for sample in result.samples} == {1.0}
    assert {sample.metadata["correctness"] for sample in result.samples} == {1.0}
    assert {sample.metadata["reward_feedback"] for sample in result.samples} == {True}
    assert {sample.reward for sample in result.samples} == {1.0}


def test_maporl_smoke_supports_logical_multi_policy_groups():
    _summary, result = run_debate_math_smoke(
        backend="rule",
        agent_count=2,
        agent_ids=("agent_0", "agent_1"),
        model_ids=("policy_a", "policy_b"),
        rollouts_per_task=1,
        max_turns=2,
        consensus_threshold=2,
    )

    assert {sample.agent_name for sample in result.samples} == {"agent_0", "agent_1"}
    assert {sample.policy_group for sample in result.samples} == {"policy_a", "policy_b"}


class _RoundAnswerBackend:
    def generate(self, request):
        answers = {
            (0, "agent_0"): 2,
            (0, "agent_1"): 2,
            (1, "agent_0"): 2,
            (1, "agent_1"): 3,
        }
        answer = answers[(request.metadata["round_id"], request.agent.name)]
        return PolicyResponse(text=f"Final answer: {answer}", token_ids=[answer], logprobs=[0.0])


def test_maporl_full_vote_meets_one_hundred_percent_threshold_and_finished_round_resets():
    task = MathTask(task_id="maporl_threshold", question="1 + 1", answer=2)
    trajectory = MAPoRLDebateOrchestra(
        early_stop=False,
        criteria_for_consensus_percentage=1.0,
    ).run(
        episode_id="episode",
        rollout_group="group",
        task=task,
        team=default_debate_team(agent_count=2, max_turns=2),
        observation=task.question,
        policy_backend=_RoundAnswerBackend(),
    )

    first_round = [turn for turn in trajectory.turns if turn.metadata["round_id"] == 0]
    assert all(turn.metadata["consensus_reached"] is True for turn in first_round)
    assert trajectory.metadata["consensus_reached"] is False
    assert trajectory.metadata["finished_round"] == -1
    assert all(turn.metadata["finished_round"] == -1 for turn in trajectory.turns)


def test_maporl_smoke_and_train_use_credit_section_precedence(monkeypatch):
    from trajweave.recipes.maporl import plugin as maporl_plugin

    captured = {}

    def fake_smoke(**kwargs):
        captured.update(kwargs)
        summary = SimpleNamespace(
            trajectories=0,
            samples=0,
            success_rate=0.0,
            dataproto_status="skipped",
            dataproto_rows=None,
        )
        result = SimpleNamespace(trajectories=[], samples=[], success_rate=0.0)
        return summary, result

    monkeypatch.setattr(maporl_plugin, "run_debate_math_smoke", fake_smoke)
    shared_config = {
        "maporl": {
            "agent_count": 2,
            "rule_horizon": "last",
            "rule_agent_share": "individual",
            "rule_discount": 0.9,
            "alpha": [9, 9, 9, 9],
        },
        "credit": {
            "rule_horizon": "current",
            "rule_agent_share": "all",
            "rule_discount": 0.25,
            "alpha": [1, 2, 3, 4],
        },
        "prepare": {"tiny_verl_assets": {"enabled": False}},
    }

    run_from_config(
        {
            **shared_config,
            "recipe": "maporl_debate_math",
            "mode": "smoke",
            "backend": {"type": "rule"},
        }
    )
    assert captured["rule_horizon"] == "current"
    assert captured["rule_agent_share"] == "all"
    assert captured["rule_discount"] == 0.25
    assert captured["alpha"] == (1.0, 2.0, 3.0, 4.0)

    result = run_from_config(
        {
            **shared_config,
            "recipe": "maporl.debate_math.full_verl_tiny",
            "mode": "verl_plan",
            "maporl": {
                **shared_config["maporl"],
                "agent_loop_backend": "synthetic_tq",
            },
            "verl": {"enabled": True, "execute": False},
        }
    )
    command = result["verl_launch"]["command"]
    assert "+agent.orchestra.maporl.rule_horizon=current" in command
    assert "+agent.orchestra.maporl.rule_agent_share=all" in command
    assert "+agent.orchestra.maporl.rule_discount=0.25" in command
    assert "+agent.orchestra.maporl.alpha=[1.0,2.0,3.0,4.0]" in command


def test_yaml_runner_runs_maporl_debate_math_smoke_config():
    result = run_from_config(
        {
            "recipe": "maporl_debate_math",
            "mode": "smoke",
            "backend": {"type": "rule"},
            "team": {"agent_count": 2, "max_turns": 2},
            "maporl": {"agent_count": 2, "consensus_threshold": 2},
            "rollout": {"rollouts_per_task": 1},
        }
    )

    assert result["recipe"] == "maporl_debate_math"
    assert result["trajectories"] == 2
    assert result["samples"] == 4
    assert result["success_rate"] == 1.0


def test_maporl_verl_config_prepares_namespaced_launch():
    result = run_from_config(
        {
            "recipe": "maporl.debate_math.full_verl_tiny",
            "mode": "verl_plan",
            "prepare": {"tiny_verl_assets": {"enabled": False}},
            "maporl": {
                "agent_count": 2,
                "agent_ids": ["agent_0", "agent_1"],
                "model_ids": ["shared", "shared"],
                "agent_loop_backend": "synthetic_tq",
                "hf_local_dtype": "fp16",
                "hf_local_model_cache_size": 1,
                "max_rounds": 2,
                "worker_groups": {
                    "shared": {
                        "model_path": "outputs/maporl_debate_math_tiny_assets/model",
                        "trainable": True,
                        "gpus": 2,
                    }
                },
            },
            "protocol": {"consensus_threshold": 2},
            "verl": {
                "enabled": True,
                "execute": False,
                "overrides": ["trainer.use_v1=true"],
            },
        },
        config_path="configs/maporl/debate_math_verl_tiny.yaml",
    )

    command = result["verl_launch"]["command"]
    command_text = " ".join(command)
    assert result["canonical_recipe"] == "maporl.debate_math.full_verl_tiny"
    assert result["maporl"]["runtime_recipe"] == "maporl_debate_math"
    assert "trajweave.backends.verl.main_ppo" in command
    assert "+trajweave.recipe=maporl_debate_math" in command
    assert "+trajweave.verl_extensions=[trajweave_maporl_full_ppo]" in command
    assert (
        "++algorithm.extension_hooks_class=trajweave.backends.verl.extensions.common.hooks.MAPoRLFullPPOHooks"
        in command
    )
    assert '+agent.agent_ids=["agent_0","agent_1"]' in command
    assert '+agent.worker_group_ids=["shared"]' in command
    assert (
        '+agent.worker_groups=[{id:"shared",trainable:true,model_path:"outputs/maporl_debate_math_tiny_assets/model",gpus:2}]'
        in command
    )
    assert "+agent.orchestra.maporl.max_rounds=2" in command
    assert "+agent.orchestra.maporl.early_stop=true" in command
    assert "+agent.orchestra.maporl.rule_horizon=discounted_sum" in command
    assert "+trajweave.credit_allocator=maporl_ppo_score_rule" in command
    assert "+trajweave.agent_loop_backend=synthetic_tq" in command
    assert "+trajweave.hf_local_dtype=fp16" in command
    assert "+trajweave.hf_local_model_cache_size=1" in command
    assert "trajweave.backends.verl.agent_loop.TrajWeaveAgentLoopManager" in command_text


def test_maporl_verl_config_keeps_multi_policy_metadata():
    result = run_from_config(
        {
            "recipe": "maporl.debate_math.full_verl_tiny",
            "mode": "verl_plan",
            "prepare": {"tiny_verl_assets": {"enabled": False}},
            "maporl": {
                "agent_count": 2,
                "agent_ids": ["agent_0", "agent_1"],
                "model_ids": ["model_a", "model_b"],
                "agent_loop_backend": "synthetic_tq",
                "worker_groups": {
                    "model_a": {"model_path": "/models/model-a", "trainable": True, "gpus": 1},
                    "model_b": {"model_path": "/models/model-b", "trainable": False, "gpus": 0},
                },
            },
            "verl": {"enabled": True, "execute": False},
        }
    )

    command = result["verl_launch"]["command"]
    assert '+agent.model_ids=["model_a","model_b"]' in command
    assert "+agent.model_sharing=false" in command
    assert '+agent.worker_group_ids=["model_a","model_b"]' in command
    assert (
        '+agent.worker_groups=[{id:"model_a",trainable:true,model_path:"/models/model-a",gpus:1},{id:"model_b",trainable:false,model_path:"/models/model-b",gpus:0}]'
        in command
    )
    assert result["maporl"]["single_model_only"] is False
    assert result["maporl"]["native_multi_actor_training"] is False


def test_maporl_verl_config_enables_multi_actor_trainer_for_trainable_groups():
    result = run_from_config(
        {
            "recipe": "maporl.debate_math.full_verl_tiny",
            "mode": "verl_plan",
            "prepare": {"tiny_verl_assets": {"enabled": False}},
            "maporl": {
                "agent_count": 2,
                "agent_ids": ["agent_0", "agent_1"],
                "model_ids": ["model_a", "model_b"],
                "agent_loop_backend": "synthetic_tq",
                "multi_actor_training": True,
                "worker_groups": {
                    "model_a": {
                        "model_path": "/models/model-a",
                        "tokenizer_path": "/models/shared-tokenizer",
                        "trainable": True,
                        "gpus": 1,
                    },
                    "model_b": {
                        "model_path": "/models/model-b",
                        "tokenizer_path": "/models/shared-tokenizer",
                        "trainable": True,
                        "gpus": 1,
                    },
                },
            },
            "verl": {
                "enabled": True,
                "execute": False,
                "overrides": ["trainer.use_v1=true", "trainer.v1.trainer_mode=sync"],
            },
        }
    )

    command = result["verl_launch"]["command"]
    assert "trainer.v1.trainer_mode=sync" not in command
    assert "trainer.v1.trainer_mode=trajweave_maporl_multi_actor_sync" in command
    assert "+trajweave.multi_actor.tokenizer_mode=shared" in command
    assert "+trajweave.multi_actor.enabled=true" in command
    assert "+trajweave.multi_actor.routing_field=worker_group" in command
    assert "+trajweave.turn_padding_multiple=2" in command
    assert (
        '+agent.worker_groups=[{id:"model_a",trainable:true,model_path:"/models/model-a",'
        'tokenizer_path:"/models/shared-tokenizer",gpus:1},{id:"model_b",trainable:true,'
        'model_path:"/models/model-b",tokenizer_path:"/models/shared-tokenizer",gpus:1}]' in command
    )
    assert result["maporl"]["native_multi_actor_training"] is True
    assert result["maporl"]["training_backend"] == "verl_v1_multi_actor_wg"
    assert result["maporl"]["tokenizer_mode"] == "shared"
    assert result["maporl"]["multi_actor_validation_status"] == "passed"


def test_maporl_verl_config_allows_compatible_tokenizer_paths(monkeypatch):
    from trajweave.recipes.maporl import config as maporl_config

    monkeypatch.setattr(
        maporl_config,
        "assert_compatible_tokenizers",
        lambda paths: {path: type("Fingerprint", (), {"digest": "same-digest"})() for path in paths},
    )

    result = run_from_config(
        {
            "recipe": "maporl.debate_math.full_verl_tiny",
            "mode": "verl_plan",
            "prepare": {"tiny_verl_assets": {"enabled": False}},
            "maporl": {
                "agent_count": 2,
                "agent_ids": ["agent_0", "agent_1"],
                "model_ids": ["model_a", "model_b"],
                "agent_loop_backend": "synthetic_tq",
                "tokenizer_mode": "compatible",
                "multi_actor_training": True,
                "worker_groups": {
                    "model_a": {
                        "model_path": "/models/model-a",
                        "tokenizer_path": "/models/tokenizer-a",
                        "trainable": True,
                    },
                    "model_b": {
                        "model_path": "/models/model-b",
                        "tokenizer_path": "/models/tokenizer-b",
                        "trainable": True,
                    },
                },
            },
            "verl": {"enabled": True, "execute": False},
        }
    )

    command = result["verl_launch"]["command"]
    assert "+trajweave.multi_actor.tokenizer_mode=compatible" in command
    assert result["maporl"]["native_multi_actor_training"] is True
    assert result["maporl"]["tokenizer_mode"] == "compatible"
    assert result["maporl"]["multi_actor_validation_status"] == "passed"
    assert result["maporl"]["multi_actor_validation"]["compatible_tokenizer_paths"] == [
        "/models/tokenizer-a",
        "/models/tokenizer-b",
    ]


def test_maporl_verl_config_uses_safe_worker_group_list_override_for_special_ids():
    result = run_from_config(
        {
            "recipe": "maporl.debate_math.full_verl_tiny",
            "mode": "verl_plan",
            "prepare": {"tiny_verl_assets": {"enabled": False}},
            "maporl": {
                "agent_count": 2,
                "agent_ids": ["agent_0", "agent_1"],
                "model_ids": ["qwen2.5/0.5b", "qwen2.5/0.5b"],
                "agent_loop_backend": "synthetic_tq",
                "worker_groups": {
                    "qwen2.5/0.5b": {
                        "model_path": "/models/qwen2.5-0.5b",
                        "tokenizer_path": "/models/qwen2.5-0.5b-tokenizer",
                        "trainable": True,
                    }
                },
            },
            "verl": {"enabled": True, "execute": False},
        }
    )

    command = result["verl_launch"]["command"]
    assert '+agent.worker_group_ids=["qwen2.5/0.5b"]' in command
    assert (
        '+agent.worker_groups=[{id:"qwen2.5/0.5b",trainable:true,'
        'model_path:"/models/qwen2.5-0.5b",tokenizer_path:"/models/qwen2.5-0.5b-tokenizer"}]' in command
    )
    assert not any("agent.worker_groups.qwen2.5/0.5b" in item for item in command)


def test_maporl_verl_config_rejects_native_verl_tq_backend():
    try:
        run_from_config(
            {
                "recipe": "maporl.debate_math.full_verl_tiny",
                "mode": "verl_train",
                "prepare": {"tiny_verl_assets": {"enabled": False}},
                "maporl": {
                    "agent_count": 2,
                    "agent_ids": ["agent_0", "agent_1"],
                    "model_ids": ["shared", "shared"],
                    "agent_loop_backend": "verl_tq",
                },
                "verl": {"enabled": True, "execute": False},
            }
        )
    except ValueError as exc:
        assert "not verl_tq" in str(exc)
    else:
        raise AssertionError("MAPoRL full PPO should reject native verl_tq.")


def test_maporl_multi_actor_requires_two_trainable_worker_groups():
    try:
        run_from_config(
            {
                "recipe": "maporl.debate_math.full_verl_tiny",
                "mode": "verl_train",
                "prepare": {"tiny_verl_assets": {"enabled": False}},
                "maporl": {
                    "agent_count": 2,
                    "agent_ids": ["agent_0", "agent_1"],
                    "model_ids": ["model_a", "model_b"],
                    "agent_loop_backend": "synthetic_tq",
                    "multi_actor_training": True,
                    "worker_groups": {
                        "model_a": {
                            "model_path": "/models/model-a",
                            "tokenizer_path": "/models/shared-tokenizer",
                            "trainable": True,
                        },
                        "model_b": {
                            "model_path": "/models/model-b",
                            "tokenizer_path": "/models/shared-tokenizer",
                            "trainable": False,
                        },
                    },
                },
                "verl": {"enabled": True, "execute": False},
            }
        )
    except ValueError as exc:
        assert "at least two trainable worker groups" in str(exc)
    else:
        raise AssertionError("MAPoRL multi_actor_training should require two trainable groups.")


def test_maporl_multi_actor_requires_explicit_model_and_tokenizer_paths():
    try:
        run_from_config(
            {
                "recipe": "maporl.debate_math.full_verl_tiny",
                "mode": "verl_train",
                "prepare": {"tiny_verl_assets": {"enabled": False}},
                "maporl": {
                    "agent_count": 2,
                    "agent_ids": ["agent_0", "agent_1"],
                    "model_ids": ["model_a", "model_b"],
                    "agent_loop_backend": "synthetic_tq",
                    "multi_actor_training": True,
                    "worker_groups": {
                        "model_a": {"model_path": "/models/model-a", "trainable": True},
                        "model_b": {"tokenizer_path": "/models/shared-tokenizer", "trainable": True},
                    },
                },
                "verl": {"enabled": True, "execute": False},
            }
        )
    except ValueError as exc:
        message = str(exc)
        assert "model_a.tokenizer_path" in message
        assert "model_b.model_path" in message
    else:
        raise AssertionError("MAPoRL multi_actor_training should require explicit model/tokenizer paths.")


def test_maporl_multi_actor_rejects_different_tokenizers_for_stable_path():
    try:
        run_from_config(
            {
                "recipe": "maporl.debate_math.full_verl_tiny",
                "mode": "verl_train",
                "prepare": {"tiny_verl_assets": {"enabled": False}},
                "maporl": {
                    "agent_count": 2,
                    "agent_ids": ["agent_0", "agent_1"],
                    "model_ids": ["model_a", "model_b"],
                    "agent_loop_backend": "synthetic_tq",
                    "tokenizer_mode": "shared",
                    "multi_actor_training": True,
                    "worker_groups": {
                        "model_a": {
                            "model_path": "/models/model-a",
                            "tokenizer_path": "/models/tokenizer-a",
                            "trainable": True,
                        },
                        "model_b": {
                            "model_path": "/models/model-b",
                            "tokenizer_path": "/models/tokenizer-b",
                            "trainable": True,
                        },
                    },
                },
                "verl": {"enabled": True, "execute": False},
            }
        )
    except ValueError as exc:
        assert "identical tokenizer_path" in str(exc)
    else:
        raise AssertionError("MAPoRL stable multi_actor_training should reject different tokenizers.")


@pytest.mark.parametrize(
    ("maporl", "message"),
    [
        (
            {"agent_count": 3, "agent_ids": ["agent_0", "agent_1"], "model_ids": ["shared", "shared"]},
            "does not match",
        ),
        (
            {"agent_count": 2, "agent_ids": ["agent_0", "agent_0"], "model_ids": ["shared", "shared"]},
            "must be unique",
        ),
        ({"agent_count": 2, "max_rounds": 0}, "max_rounds"),
        ({"agent_count": 2, "consensus_threshold": 3}, "consensus_threshold"),
        ({"agent_count": 2, "criteria_for_consensus_percentage": 1.1}, "consensus_percentage"),
        ({"agent_count": 2, "criteria_for_consensus_reward_threshold": -0.1}, "reward_threshold"),
    ],
)
def test_maporl_verl_config_rejects_invalid_protocol_shapes(maporl, message):
    with pytest.raises(ValueError, match=message):
        run_from_config(
            {
                "recipe": "maporl.debate_math.full_verl_tiny",
                "mode": "verl_train",
                "prepare": {"tiny_verl_assets": {"enabled": False}},
                "maporl": {"agent_loop_backend": "synthetic_tq", **maporl},
                "verl": {"enabled": True, "execute": False},
            }
        )
