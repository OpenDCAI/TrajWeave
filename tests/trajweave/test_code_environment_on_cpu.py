from __future__ import annotations

import pytest

from trajweave.envs.code import CodeExecutionEnvironment, CodeTask
from trajweave.verifiers import CallableVerifierAdapter, VerifierResult


def test_code_environment_exposes_initial_observation_and_deterministic_rewards():
    task = CodeTask(task_id="code-1", prompt="Implement add().", node_rewards=(0.0, 1.0))
    environment = CodeExecutionEnvironment()

    assert environment.initial_observation(task) == "Implement add()."
    assert environment.verify_candidate(task, "bad", node_id=0).score == 0.0
    result = environment.verify_candidate(task, "good", node_id=1, parent_idx=0)
    assert result.score == 1.0
    assert result.success is True
    assert result.metadata["verifier_name"] == "deterministic_code_verifier"


def test_code_environment_delegates_to_call_based_verifier():
    seen = []

    def verify(request):
        seen.append(request)
        return VerifierResult(score=0.75, success=True, feedback="partial tests passed")

    task = CodeTask(
        task_id="code-2",
        prompt="Solve it.",
        reward_model={"test_cases": {"inputs": [], "outputs": []}},
        extra_info={"difficulty": "easy"},
    )
    result = CodeExecutionEnvironment(CallableVerifierAdapter(verify)).verify_candidate(
        task, "candidate", node_id=3, parent_idx=1
    )

    assert result.score == 0.75
    assert seen[0].task_id == "code-2"
    assert seen[0].node_id == 3
    assert seen[0].parent_idx == 1
    assert seen[0].metadata["extra_info"]["difficulty"] == "easy"


def test_code_environment_builds_task_from_prompt_record():
    task = CodeExecutionEnvironment.from_record(
        {
            "uid": "row-1",
            "raw_prompt": [
                {"role": "system", "content": "Write Python."},
                {"role": "user", "content": "Return 42."},
            ],
            "reward_model": {"ground_truth": "return 42"},
            "extra_info": {"split": "train"},
        }
    )

    assert task.task_id == "row-1"
    assert task.prompt == "Write Python.\nReturn 42."
    assert task.reward_model == {"ground_truth": "return 42"}
    assert task.extra_info == {"split": "train"}


def test_code_environment_rejects_missing_deterministic_node_reward():
    task = CodeTask(task_id="code-3", prompt="x", node_rewards=(0.0,))

    with pytest.raises(IndexError, match="has no reward for node 1"):
        CodeExecutionEnvironment().verify_candidate(task, "candidate", node_id=1)
