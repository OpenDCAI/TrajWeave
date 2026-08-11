from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from trajweave.backends.policy import PolicyRequest, PolicyResponse, StableByteTokenizer
from trajweave.core.trajectory import MultiAgentTrajectory
from trajweave.credit.comas import (
    CoMASInteractionCreditAssigner,
    CoMASInteractionRewards,
    allocate_comas_interaction_rewards,
    parse_comas_score,
)
from trajweave.envs.math import MathTask
from trajweave.orchestration.comas import CoMASPeerReviewOrchestra
from trajweave.recipes.comas.peer_review_math import default_comas_team, run_comas_math_smoke
from trajweave.runner import run_from_config


@dataclass
class _RecordingBackend:
    tokenizer: StableByteTokenizer = StableByteTokenizer()
    requests: list[PolicyRequest] = field(default_factory=list)

    def generate(self, request: PolicyRequest) -> PolicyResponse:
        self.requests.append(request)
        stage = request.metadata["comas_stage"]
        discussion_id = int(request.metadata["discussion_id"])
        if stage == "solver":
            text = "\\boxed{2}" if discussion_id == 0 else "\\boxed{3}"
        elif stage == "evaluator":
            text = "No fatal issue." if discussion_id == 0 else "The result is incorrect."
        else:
            text = "<score>3</score>" if discussion_id == 0 else "<score>1</score>"
        ids = self.tokenizer.encode(text)
        return PolicyResponse(text=text, token_ids=ids, logprobs=[0.0] * len(ids))


def _trajectory(*, num_rounds: int = 1) -> tuple[MultiAgentTrajectory, _RecordingBackend]:
    backend = _RecordingBackend()
    task = MathTask(task_id="comas-task", question="What is 1 + 1?", answer=2)
    team = default_comas_team(num_rounds=num_rounds)
    trajectory = CoMASPeerReviewOrchestra(
        num_rounds=num_rounds,
        num_references=2,
        assignment_seed=7,
    ).run(
        episode_id="episode-0",
        rollout_group="comas-task",
        task=task,
        team=team,
        observation=task.question,
        policy_backend=backend,
    )
    return trajectory, backend


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("reason\n<score>1</score>", 1),
        ("<SCORE> 2 </SCORE>", 2),
        ("<score><b>3</b></score>", 3),
        ("score: 3", None),
        ("<score>4</score>", None),
        ("<score>two</score>", None),
    ],
)
def test_comas_score_parser_matches_source_contract(text: str, expected: int | None):
    assert parse_comas_score(text) == expected


def test_comas_interaction_reward_truth_table_matches_source():
    assert allocate_comas_interaction_rewards(1) == CoMASInteractionRewards(0.0, 1.0, 0.0, 0.0, True)
    assert allocate_comas_interaction_rewards(2) == CoMASInteractionRewards(0.5, 0.5, 0.0, 0.5, True)
    assert allocate_comas_interaction_rewards(3) == CoMASInteractionRewards(1.0, 0.0, 0.0, 1.0, True)
    assert allocate_comas_interaction_rewards(None) == CoMASInteractionRewards(0.0, 0.0, -1.0, None, False)


def test_comas_protocol_runs_solver_evaluator_scorer_and_never_passes_ground_truth_to_policy():
    trajectory, backend = _trajectory()

    assert len(trajectory.turns) == 6
    assert [turn.role for turn in trajectory.turns] == [
        "solver",
        "solver",
        "evaluator",
        "evaluator",
        "scorer",
        "scorer",
    ]
    assert {turn.agent_name for turn in trajectory.turns if turn.role == "evaluator"} == {"agent_0", "agent_1"}
    assert {turn.agent_name for turn in trajectory.turns if turn.role == "scorer"} == {"agent_0", "agent_1"}
    assert all("answer" not in request.metadata for request in backend.requests)
    assert all("ground_truth" not in request.metadata for request in backend.requests)
    assert trajectory.metadata["comas_final_solutions"] == ["\\boxed{2}", "\\boxed{3}"]


def test_comas_second_round_receives_sampled_solution_and_evaluation_references():
    trajectory, _backend = _trajectory(num_rounds=2)

    second_round_solvers = [
        turn for turn in trajectory.turns if turn.role == "solver" and turn.metadata["round_id"] == 1
    ]
    assert len(second_round_solvers) == 2
    assert all("=== Discussion 1 ===" in turn.prompt for turn in second_round_solvers)
    assert all("Solution:" in turn.prompt and "Evaluation:" in turn.prompt for turn in second_round_solvers)
    assert all(len(turn.metadata["reference_interaction_ids"]) == 2 for turn in second_round_solvers)


def test_comas_credit_uses_only_interaction_score_and_not_global_accuracy():
    trajectory, _backend = _trajectory()
    team = default_comas_team(num_rounds=1)
    assigner = CoMASInteractionCreditAssigner()

    trajectory.global_reward = 0.0
    rewards_at_zero = [sample.reward for sample in assigner.assign([trajectory], team)]
    trajectory.global_reward = 1.0
    rewards_at_one = [sample.reward for sample in assigner.assign([trajectory], team)]

    assert rewards_at_zero == rewards_at_one == [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    assert trajectory.metadata["comas_training_uses_ground_truth"] is False


def test_comas_invalid_score_penalizes_only_scorer():
    trajectory, _backend = _trajectory()
    scorer = next(turn for turn in trajectory.turns if turn.role == "scorer")
    scorer.action_text = "I cannot decide."

    samples = CoMASInteractionCreditAssigner().assign([trajectory], default_comas_team(num_rounds=1))
    interaction_id = scorer.metadata["interaction_id"]
    interaction = [sample for sample in samples if sample.metadata["interaction_id"] == interaction_id]

    assert {sample.role: sample.reward for sample in interaction} == {
        "solver": 0.0,
        "evaluator": 0.0,
        "scorer": -1.0,
    }
    assert all(sample.metadata["score_valid"] is False for sample in interaction)


def test_comas_smoke_emits_all_roles_for_both_independent_policies():
    summary, result = run_comas_math_smoke(num_rounds=2, num_references=2)

    assert summary.trajectories == 2
    assert summary.samples == 24
    assert {sample.policy_group for sample in result.samples} == {"policy_0", "policy_1"}
    assert {sample.role for sample in result.samples} == {"solver", "evaluator", "scorer"}
    assert {sample.metadata["credit"] for sample in result.samples} == {"comas_interaction_reward"}
    assert all(trajectory.metadata["comas_final_accuracy"] == 1.0 for trajectory in result.trajectories)


def test_comas_verl_config_selects_generic_multi_actor_trainer_and_namespaced_hook():
    result = run_from_config(
        {
            "recipe": "comas.peer_review_math",
            "mode": "verl_plan",
            "comas": {
                "agent_count": 2,
                "agent_ids": ["agent-a", "agent-b"],
                "model_ids": ["policy-a", "policy-b"],
                "shared_agents": False,
                "multi_actor_training": True,
                "agent_loop_backend": "synthetic_tq",
                "num_rounds": 2,
                "num_references": 2,
                "worker_groups": {
                    "policy-a": {
                        "model_path": "/models/qwen-a",
                        "tokenizer_path": "/models/tokenizer",
                        "gpus": 1,
                    },
                    "policy-b": {
                        "model_path": "/models/qwen-b",
                        "tokenizer_path": "/models/tokenizer",
                        "gpus": 1,
                    },
                },
            },
            "prepare": {"tiny_verl_assets": {"enabled": False}},
            "verl": {
                "enabled": True,
                "execute": False,
                "overrides": ["trainer.use_v1=true", "trainer.v1.trainer_mode=sync"],
            },
        }
    )

    command = result["verl_launch"]["command"]
    assert "trainer.v1.trainer_mode=sync" not in command
    assert "trainer.v1.trainer_mode=trajweave_multi_actor_sync" in command
    assert "+trajweave.recipe=comas_peer_review_math" in command
    assert "+trajweave.credit_allocator=comas_interaction_reward" in command
    assert "+trajweave.verl_extensions=[trajweave_comas_interaction_reinforce]" in command
    assert any("CoMASInteractionREINFORCEHooks" in item for item in command)
    assert '+agent.model_ids=["policy-a","policy-b"]' in command
    assert result["comas"]["ground_truth_used_for_training"] is False
    assert result["comas"]["multi_actor_validation"]["status"] == "passed"


def test_comas_independent_agents_cannot_silently_fall_back_to_single_actor():
    with pytest.raises(ValueError, match="multi_actor_training=true"):
        run_from_config(
            {
                "recipe": "comas.peer_review_math",
                "mode": "smoke",
                "comas": {
                    "agent_count": 2,
                    "model_ids": ["policy-a", "policy-b"],
                    "shared_agents": False,
                    "multi_actor_training": False,
                },
            }
        )
