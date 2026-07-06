import pytest

from trajweave.recipes.agentflow import run_planner_tool_smoke
from trajweave.runner import run_from_config


def test_agentflow_smoke_records_full_flow_but_trains_planner_only():
    _summary, result = run_planner_tool_smoke(
        backend="rule",
        rollouts_per_task=1,
        max_steps=3,
    )

    assert len(result.trajectories) == 2
    assert len(result.samples) == 2
    assert result.success_rate == 1.0
    assert {sample.agent_name for sample in result.samples} == {"planner"}
    assert {sample.policy_group for sample in result.samples} == {"planner"}
    assert {sample.metadata["credit"] for sample in result.samples} == {"agentflow_planner_only_grpo"}
    assert {sample.metadata["reward_scope"] for sample in result.samples} == {"final_outcome"}

    stages = {turn.metadata["agentflow_stage"] for turn in result.trajectories[0].turns}
    assert stages == {"planner_next_step", "executor_command", "tool_result", "verifier_decision"}
    assert result.trajectories[0].metadata["agentflow_protocol"]["trainable_agent"] == "planner"


def test_yaml_runner_runs_agentflow_smoke_config():
    result = run_from_config(
        {
            "recipe": "agentflow_planner_tool",
            "mode": "smoke",
            "backend": {"type": "rule"},
            "team": {"max_turns": 3},
            "rollout": {"rollouts_per_task": 1},
        }
    )

    assert result["recipe"] == "agentflow_planner_tool"
    assert result["trajectories"] == 2
    assert result["samples"] == 2
    assert result["success_rate"] == 1.0


def test_agentflow_verl_config_prepares_planner_grpo_launch():
    result = run_from_config(
        {
            "recipe": "agentflow.flow_grpo.planner_tool",
            "mode": "verl_train",
            "prepare": {"tiny_verl_assets": {"enabled": False}},
            "agentflow": {
                "agent_loop_backend": "synthetic_tq",
                "max_steps": 2,
                "enabled_tools": ["base_generator"],
            },
            "verl": {
                "enabled": True,
                "execute": False,
                "overrides": ["trainer.use_v1=true"],
            },
        },
        config_path="configs/agentflow/flow_grpo_verl_tiny.yaml",
    )

    command = result["verl_launch"]["command"]
    command_text = " ".join(command)
    assert result["canonical_recipe"] == "agentflow.flow_grpo.planner_tool"
    assert result["agentflow"]["runtime_recipe"] == "agentflow_planner_tool"
    assert "trajweave.backends.verl.main_ppo" in command
    assert "+trajweave.recipe=agentflow_planner_tool" in command
    assert "+trajweave.verl_extensions=[trajweave_agentflow_planner_grpo]" in command
    assert "++algorithm.extension_hooks_class=trajweave.backends.verl.extensions.hooks.AgentFlowPlannerGRPOHooks" in command
    assert "+agent.orchestra.agentflow.max_steps=2" in command
    assert "+agent.orchestra.agentflow.enabled_tools=[\"base_generator\"]" in command
    assert "+trajweave.credit_allocator=agentflow_planner_only_grpo" in command
    assert "+trajweave.agent_loop_backend=synthetic_tq" in command
    assert "trajweave.backends.verl.agent_loop.TrajWeaveAgentLoopManager" in command_text


def test_agentflow_verl_config_rejects_native_verl_tq_backend():
    with pytest.raises(ValueError, match="not verl_tq"):
        run_from_config(
            {
                "recipe": "agentflow.flow_grpo.planner_tool",
                "mode": "verl_train",
                "prepare": {"tiny_verl_assets": {"enabled": False}},
                "agentflow": {"agent_loop_backend": "verl_tq"},
                "verl": {"enabled": True, "execute": False},
            }
        )
