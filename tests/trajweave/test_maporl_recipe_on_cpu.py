from trajweave.runner import run_from_config
from trajweave.recipes.maporl import run_debate_math_smoke


def test_maporl_smoke_shapes_rewards_by_shared_policy_group():
    _summary, result = run_debate_math_smoke(
        backend="rule",
        agent_count=2,
        rollouts_per_task=1,
        max_turns=2,
        consensus_threshold=2,
    )

    assert len(result.samples) == 4
    assert {sample.policy_group for sample in result.samples} == {"shared"}
    assert {sample.metadata["advantage_group"].split(":")[-1] for sample in result.samples} == {"shared"}
    assert {sample.metadata["credit"] for sample in result.samples} == {"maporl_score_bonus"}
    assert {sample.metadata["raw_global_reward"] for sample in result.samples} == {1.0}
    assert {sample.metadata["shaped_reward"] for sample in result.samples} == {1.5}


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
            "recipe": "maporl.debate_math.verl_tiny",
            "mode": "verl_train",
            "prepare": {"tiny_verl_assets": {"enabled": False}},
            "maporl": {
                "agent_count": 2,
                "agent_ids": ["agent_0", "agent_1"],
                "model_ids": ["shared", "shared"],
                "agent_loop_backend": "synthetic_tq",
                "max_rounds": 2,
            },
            "protocol": {"consensus_threshold": 2},
            "verl": {
                "enabled": True,
                "execute": False,
                "overrides": ["trainer.use_v1=true"],
            },
        },
        config_path="examples/trajweave/configs/maporl/debate_math_verl_tiny.yaml",
    )

    command = result["verl_launch"]["command"]
    command_text = " ".join(command)
    assert result["canonical_recipe"] == "maporl.debate_math.verl_tiny"
    assert result["maporl"]["runtime_recipe"] == "maporl_debate_math"
    assert "trajweave.backends.verl.main_ppo" in command
    assert "+trajweave.recipe=maporl_debate_math" in command
    assert "+trajweave.verl_extensions=[trajweave_maporl_single_model]" in command
    assert "+agent.agent_ids=[\"agent_0\",\"agent_1\"]" in command
    assert "+agent.orchestra.maporl.max_rounds=2" in command
    assert "+agent.orchestra.maporl.early_stop=true" in command
    assert "+agent.orchestra.maporl.baseline_scope=policy_group" in command
    assert "+trajweave.agent_loop_backend=synthetic_tq" in command
    assert "trajweave.backends.verl.agent_loop.TrajWeaveAgentLoopManager" in command_text


def test_maporl_verl_config_rejects_multi_model_v1():
    try:
        run_from_config(
            {
                "recipe": "maporl.debate_math.verl_tiny",
                "mode": "verl_train",
                "prepare": {"tiny_verl_assets": {"enabled": False}},
                "maporl": {
                    "agent_count": 2,
                    "agent_ids": ["agent_0", "agent_1"],
                    "model_ids": ["model_a", "model_b"],
                    "agent_loop_backend": "synthetic_tq",
                },
                "verl": {"enabled": True, "execute": False},
            }
        )
    except ValueError as exc:
        assert "single-model/shared-policy" in str(exc)
    else:
        raise AssertionError("MAPoRL v1 should reject heterogeneous model_ids.")


def test_maporl_verl_config_rejects_native_verl_tq_backend():
    try:
        run_from_config(
            {
                "recipe": "maporl.debate_math.verl_tiny",
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
        raise AssertionError("MAPoRL v1 should reject native verl_tq.")
