import pytest
from trajweave.core.specs import AgentSpec, PolicyGroupSpec, TeamSpec
from trajweave.core.trajectory import AgentTurn, MultiAgentTrajectory
from trajweave.credit.doctor_mas import DoctorMASCreditAssigner
from trajweave.recipes.doctor_mas import run_smoke


def test_doctor_mas_recipe_collects_solver_verifier_turns():
    summary, result = run_smoke(backend="rule", rollouts_per_task=1, max_turns=2)

    assert summary.trajectories == 2
    assert summary.success_rate == 1.0
    assert {sample.agent_name for sample in result.samples} == {"solver", "verifier"}
    assert all(sample.metadata["credit"] == "doctor_mas_agent_wise_grpo" for sample in result.samples)


def test_doctor_mas_agent_wise_advantage_groups_do_not_mix_agents():
    team = TeamSpec(
        name="t",
        agents=(
            AgentSpec(name="solver", role="solver", policy_group="pg"),
            AgentSpec(name="verifier", role="verifier", policy_group="pg"),
        ),
        policy_groups=(PolicyGroupSpec(name="pg"),),
        orchestra="solver_verifier",
        reward="math",
        credit="doctor_mas",
    )
    trajectories = []
    for episode_idx, reward in enumerate([1.0, 0.0]):
        traj = MultiAgentTrajectory(
            episode_id=f"e{episode_idx}",
            task_id="task",
            rollout_group="same_prompt",
            team_name="t",
            global_reward=reward,
        )
        for turn_id, agent in enumerate(team.agents):
            traj.add_turn(
                AgentTurn(
                    episode_id=traj.episode_id,
                    task_id=traj.task_id,
                    turn_id=turn_id,
                    agent_name=agent.name,
                    role=agent.role,
                    policy_group=agent.policy_group,
                    observation="1+1",
                    prompt="prompt",
                    action_text="Final answer: 2",
                    action_token_ids=[1, 2],
                )
            )
        trajectories.append(traj)

    samples = DoctorMASCreditAssigner().assign(trajectories, team)
    solver_advantages = [sample.advantage for sample in samples if sample.agent_name == "solver"]
    verifier_advantages = [sample.advantage for sample in samples if sample.agent_name == "verifier"]

    assert solver_advantages == verifier_advantages
    assert solver_advantages[0] > 0
    assert solver_advantages[1] < 0
    assert {sample.metadata["advantage_group"] for sample in samples} == {
        "same_prompt:solver",
        "same_prompt:verifier",
    }


def test_verl_dataproto_adapter_preserves_agent_metadata():
    pytest.importorskip("torch")
    pytest.importorskip("numpy")
    pytest.importorskip("ray")
    pytest.importorskip("tensordict")
    pytest.importorskip("omegaconf")
    from trajweave.backends.verl import VerlDataProtoAdapter

    _, result = run_smoke(backend="rule", rollouts_per_task=1, max_turns=2)
    data = VerlDataProtoAdapter().build(result.samples)

    assert len(data) == len(result.samples)
    assert "agent_name" in data.non_tensor_batch
    assert "agent_id" in data.non_tensor_batch
    assert set(data.non_tensor_batch["agent_id"]) == {"Solver Agent", "Verifier Agent"}
    assert "token_level_rewards" in data.batch
    assert data.batch["token_level_rewards"].shape == data.batch["responses"].shape
