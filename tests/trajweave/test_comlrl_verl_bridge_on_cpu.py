from __future__ import annotations

import pytest

from trajweave.backends.verl.emitters.comlrl import CoMLRLEmitterMixin
from trajweave.backends.verl.workflow_runtime import build_synthetic_comlrl_workflow_outputs


def _worker(algorithm: str, *, normalize: bool = False, max_turns: int = 1):
    config = {
        "trajweave": {
            "comlrl": {
                "algorithm": algorithm,
                "joint_mode": "aligned",
                "max_turns": max_turns,
                "normalize_advantages": normalize,
            }
        },
        "agent": {
            "agent_ids": ["alice", "bob"],
            "model_ids": ["actor-a", "actor-b"],
        },
    }

    class Worker(CoMLRLEmitterMixin):
        def __init__(self):
            self.config = config

        @staticmethod
        def _encode_prompt_text(text):
            return [ord(char) for char in text] or [1]

        @staticmethod
        def _encode_text(text):
            return [ord(char) for char in text] or [1]

        @staticmethod
        def _local_policy_version():
            return 0

    return Worker()


def _prompt(num_candidates: int) -> dict:
    return {
        "uid": "math-bridge",
        "raw_prompt": [{"role": "user", "content": "What is 1 + 1?"}],
        "reward_model": {"ground_truth": "2"},
        "__comlrl_num_candidates__": num_candidates,
    }


@pytest.mark.parametrize(
    ("algorithm", "expected"),
    [
        ("magrpo", [0.5, -0.5]),
        ("mareinforce", [1.0, 0.0]),
        ("marloo", [1.0, -1.0]),
        ("maremax", [0.0, -1.0]),
    ],
)
def test_synthetic_reinforce_family_emits_precomputed_joint_training_rows(algorithm, expected):
    outputs = build_synthetic_comlrl_workflow_outputs(
        _worker(algorithm),
        prompt=_prompt(2),
        session_id=0,
    )

    assert len(outputs) == 4
    assert [output.extra_fields["worker_group"] for output in outputs] == [
        "actor-a",
        "actor-a",
        "actor-b",
        "actor-b",
    ]
    assert [output.extra_fields["joint_advantage"] for output in outputs[:2]] == expected
    assert [output.extra_fields["joint_advantage"] for output in outputs[2:]] == expected
    assert [output.extra_fields["effective_projected_joint_return"] for output in outputs[:2]] == [1.0, 0.0]
    assert all(len(output.extra_fields["joint_action_ids"]) == 1 for output in outputs)
    assert all(len(output.extra_fields["joint_transition_ids"]) == 1 for output in outputs)
    assert all(output.response_mask for output in outputs)


@pytest.mark.parametrize("algorithm", ["iac", "maac"])
def test_synthetic_actor_critic_emits_one_linear_joint_transition_per_agent(algorithm):
    outputs = build_synthetic_comlrl_workflow_outputs(
        _worker(algorithm, max_turns=2),
        prompt=_prompt(1),
        session_id=0,
    )

    assert len(outputs) == 2
    assert {output.extra_fields["worker_group"] for output in outputs} == {"actor-a", "actor-b"}
    assert {tuple(output.extra_fields["joint_action_ids"]) for output in outputs} == {("math-bridge_0:root:joint:0",)}
    assert len({tuple(output.extra_fields["joint_transition_ids"]) for output in outputs}) == 1
    assert [output.extra_fields["turn_id"] for output in outputs] == [0, 0]
    assert all(output.extra_fields["joint_reward"] == 1.0 for output in outputs)
    assert all(output.extra_fields["joint_done"] is True for output in outputs)


@pytest.mark.parametrize("algorithm", ["iac", "maac"])
def test_actor_critic_rejects_multiple_candidates_before_tree_build(algorithm):
    with pytest.raises(ValueError, match="num_candidates=1"):
        build_synthetic_comlrl_workflow_outputs(
            _worker(algorithm),
            prompt=_prompt(2),
            session_id=0,
        )


def test_mareinforce_requires_two_candidates_like_the_upstream_family():
    with pytest.raises(ValueError, match="num_candidates>=2"):
        build_synthetic_comlrl_workflow_outputs(
            _worker("mareinforce"),
            prompt=_prompt(1),
            session_id=0,
        )
