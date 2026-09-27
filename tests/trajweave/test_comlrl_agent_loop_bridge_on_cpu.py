from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from trajweave.backends.policy import PolicyResponse, StableByteTokenizer
from trajweave.backends.verl.emitters.comlrl import CoMLRLEmitterMixin
from trajweave.backends.verl.workflow_runtime import _build_comlrl_outputs, build_hf_workflow_outputs
from trajweave.envs.math import MathTask
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput


class _Worker(CoMLRLEmitterMixin):
    def __init__(
        self,
        *,
        algorithm: str = "magrpo",
        joint_mode: str = "aligned",
        max_turns: int = 1,
        normalize: bool = False,
        responses: list[str] | None = None,
    ):
        actor_critic = None
        if algorithm in {"iac", "maac"}:
            actor_critic = {
                "topology": "independent" if algorithm == "iac" else "centralized",
                "critic_type": "v",
            }
        comlrl = {
            "algorithm": algorithm,
            "joint_mode": joint_mode,
            "max_turns": max_turns,
            "normalize_advantages": normalize,
        }
        if actor_critic is not None:
            comlrl["actor_critic"] = actor_critic
        self.config = {
            "trajweave": {"comlrl": comlrl},
            "agent": {
                "agent_ids": ["alice", "bob"],
                "model_ids": ["actor-a", "actor-b"],
            },
        }
        self.tokenizer = StableByteTokenizer()
        self.responses = list(responses or [])
        self.model_config = SimpleNamespace(local_path="/models/default")

    def _encode_prompt(self, value):
        return self.tokenizer.encode(str(value))

    def _encode_prompt_text(self, value: str):
        return self.tokenizer.encode(value)

    def _encode_text(self, value: str):
        return self.tokenizer.encode(value)

    def _decode_response_ids(self, values: list[int]):
        return self.tokenizer.decode(values)

    def _generate_local_response_ids(self, prompt_ids: list[int], **kwargs):
        del prompt_ids, kwargs
        if not self.responses:
            raise AssertionError("HF fake backend exhausted its responses")
        return self.tokenizer.encode(self.responses.pop(0))

    @staticmethod
    def _local_policy_version():
        return 7

    @staticmethod
    def _worker_group_model_path(group_id: str):
        return f"/models/{group_id}"


def _prompt(num_candidates: int = 2) -> dict:
    return {
        "uid": "joint-math-1",
        "raw_prompt": [{"role": "user", "content": "What is 1 + 1?"}],
        "reward_model": {"ground_truth": "2"},
        "__comlrl_num_candidates__": num_candidates,
    }


@pytest.mark.parametrize(
    ("joint_mode", "expected_component_count"),
    [("aligned", 1), ("cross", 2)],
)
def test_synthetic_full_tree_emits_one_row_per_completion_and_preserves_edges(
    joint_mode: str,
    expected_component_count: int,
):
    outputs = _Worker(joint_mode=joint_mode)._build_comlrl_joint_math_outputs(_prompt())

    assert len(outputs) == 4
    assert {output.extra_fields["worker_group"] for output in outputs} == {"actor-a", "actor-b"}
    assert len({output.extra_fields["completion_id"] for output in outputs}) == 4
    assert all(len(output.extra_fields["joint_action_ids"]) == expected_component_count for output in outputs)
    assert all(
        len(output.extra_fields["joint_action_ids"])
        == len(output.extra_fields["joint_transition_ids"])
        == len(output.extra_fields["joint_return_components"])
        for output in outputs
    )
    assert all(output.reward_score == output.extra_fields["effective_projected_joint_return"] for output in outputs)
    assert all(output.extra_fields["prompt_text"] for output in outputs)
    assert all(output.extra_fields["response_text"] for output in outputs)
    assert all(output.extra_fields["observation_text"] for output in outputs)
    assert any(output.extra_fields["joint_truncated"] for output in outputs)
    if joint_mode == "cross":
        assert any(len(output.extra_fields["joint_action_ids"]) > 1 for output in outputs)


@pytest.mark.parametrize(
    ("algorithm", "expected"),
    [
        ("magrpo", [0.5, -0.5, 0.5, -0.5]),
        ("mareinforce", [1.0, 0.0, 1.0, 0.0]),
        ("marloo", [1.0, -1.0, 1.0, -1.0]),
        ("maremax", [0.0, -1.0, 0.0, -1.0]),
    ],
)
def test_reinforce_family_selects_v141_advantage_mode(algorithm: str, expected: list[float]):
    outputs = _Worker(algorithm=algorithm)._build_comlrl_joint_math_outputs(_prompt())
    assert [output.extra_fields["joint_advantage"] for output in outputs] == pytest.approx(expected)


def test_madpo_agent_loop_builds_active_complete_preference_rows():
    outputs = _Worker(algorithm="madpo")._build_comlrl_joint_math_outputs(_prompt())

    assert len(outputs) == 4
    assert [(output.extra_fields["worker_group"], output.extra_fields["preference_side"]) for output in outputs] == [
        ("actor-a", "chosen"),
        ("actor-a", "rejected"),
        ("actor-b", "chosen"),
        ("actor-b", "rejected"),
    ]
    assert len({output.extra_fields["preference_pair_id"] for output in outputs}) == 1
    assert all(output.extra_fields["preference_loss_mask"] == 1.0 for output in outputs)
    assert all(output.extra_fields["chosen_reward"] > output.extra_fields["rejected_reward"] for output in outputs)
    assert all(output.reward_score in {0.0, 1.0} for output in outputs)
    assert all(output.extra_fields["joint_sampling_mode"] == "aligned" for output in outputs)


def test_madpo_agent_loop_enforces_v141_sampling_contract():
    with pytest.raises(ValueError, match="num_candidates>=2"):
        _Worker(algorithm="madpo")._build_comlrl_joint_math_outputs(_prompt(num_candidates=1))
    with pytest.raises(ValueError, match="joint_mode=aligned"):
        _Worker(algorithm="madpo", joint_mode="cross")._build_comlrl_joint_math_outputs(_prompt())
    with pytest.raises(ValueError, match="max_turns=1"):
        _Worker(algorithm="madpo", max_turns=2)._build_comlrl_joint_math_outputs(_prompt())


def test_iterative_preference_context_emits_policy_comparator_pairs():
    worker = _Worker(algorithm="madpo")
    worker._trajweave_comlrl_iterative_context = {
        "phase": "preference",
        "iteration": 2,
        "pair_selection": "comparator_reward",
        "pairs_per_sample": 2,
        "comparator": {
            "policy": "current",
            "generation_mode": "decentralized",
            "num_candidates": 2,
        },
    }

    outputs = worker._build_comlrl_joint_math_outputs(_prompt(), validate=False)

    assert len(outputs) == 4
    assert {output.extra_fields["preference_side"] for output in outputs} == {"chosen", "rejected"}
    assert len({output.extra_fields["preference_pair_id"] for output in outputs}) == 1
    assert all(output.extra_fields["preference_loss_mask"] == 1.0 for output in outputs)
    assert all(output.extra_fields["chosen_reward"] > output.extra_fields["rejected_reward"] for output in outputs)
    assert {output.extra_fields["worker_group"] for output in outputs} == {"actor-a", "actor-b"}
    assert all(output.extra_fields["raw_candidate_rewards"] for output in outputs)
    assert all(output.extra_fields["policy_provenance"] == {"policy": "current"} for output in outputs)
    assert all(output.extra_fields["comparator_provenance"]["policy"] == "current" for output in outputs)
    assert {output.extra_fields["winner_source"] for output in outputs} <= {"current", "comparator"}


def test_marlhf_online_reward_routes_through_frozen_scorer_but_validation_keeps_task_reward(monkeypatch):
    from trajweave.envs.math import SolverVerifierMathEnvironment

    task_evaluator_calls = []
    original_evaluate = SolverVerifierMathEnvironment.evaluate

    def recording_evaluate(self, task, action):
        task_evaluator_calls.append((task.task_id, action))
        return original_evaluate(self, task, action)

    monkeypatch.setattr(SolverVerifierMathEnvironment, "evaluate", recording_evaluate)

    class FakeScorer:
        def __init__(self):
            self.calls = []

        def score_many(self, prompts, responses):
            self.calls.append((prompts, responses))
            return [3.0, 1.0]

    scorer = FakeScorer()
    worker = _Worker(algorithm="magrpo")
    worker._comlrl_marlhf_reward_scorer = lambda _team: scorer

    train_outputs = worker._build_comlrl_joint_math_outputs(_prompt(), validate=False)
    assert task_evaluator_calls == []
    validation_outputs = worker._build_comlrl_joint_math_outputs(_prompt(), validate=True)

    assert len(task_evaluator_calls) == 2
    assert [output.reward_score for output in train_outputs] == pytest.approx([3.0, 1.0, 3.0, 1.0])
    assert [output.extra_fields["joint_advantage"] for output in train_outputs] == pytest.approx([1.0, -1.0, 1.0, -1.0])
    assert [output.reward_score for output in validation_outputs] == pytest.approx([1.0, 1.0])
    assert validation_outputs[-1].extra_fields["comlrl_validation_summary"] is True
    assert validation_outputs[-1].extra_fields["workflow_evaluation_reward"] == pytest.approx(1.0)
    assert len(scorer.calls) == 1
    prompts, responses = scorer.calls[0]
    assert list(prompts[0]) == ["alice", "bob"]
    assert list(responses[0]) == ["alice", "bob"]


def test_validation_uses_only_candidate_zero_even_when_only_a_later_candidate_is_correct():
    class CandidateIndexedBackend:
        def __init__(self):
            self.calls = []

        def generate(self, request):
            candidate_index = int(request.metadata["candidate_index"])
            self.calls.append((request.agent.name, candidate_index))
            text = f"Final answer: {2 if candidate_index == 1 else 999}"
            token_ids = StableByteTokenizer().encode(text)
            return PolicyResponse(text=text, token_ids=token_ids, logprobs=[0.0] * len(token_ids))

    backend = CandidateIndexedBackend()
    outputs = _build_comlrl_outputs(
        _Worker(algorithm="magrpo"),
        task=MathTask(task_id="validation-k1", question="What is 1 + 1?", answer=2),
        prompt=_prompt(num_candidates=2),
        session_id=0,
        policy_backend=backend,
        validate=True,
    )

    assert backend.calls == [("alice", 0), ("bob", 0)]
    assert len(outputs) == 2
    assert outputs[-1].extra_fields["agent_id"] == "alice"
    assert outputs[-1].extra_fields["worker_group"] == "actor-a"
    assert outputs[-1].reward_score == pytest.approx(0.0)
    assert outputs[-1].extra_fields["workflow_evaluation_reward"] == pytest.approx(0.0)


def test_hf_fake_backend_builds_the_same_joint_bridge_with_worker_routing():
    worker = _Worker(
        responses=[
            "Final answer: 2",
            "Final answer: 3",
            "Final answer: 2",
            "Final answer: 4",
        ]
    )
    outputs = build_hf_workflow_outputs(
        worker,
        recipe="comlrl_joint_math",
        prompt=_prompt(),
        session_id=0,
    )

    assert len(outputs) == 4
    assert {output.extra_fields["worker_group"] for output in outputs} == {"actor-a", "actor-b"}
    assert {output.extra_fields["worker_group_model_path"] for output in outputs} == {
        "/models/actor-a",
        "/models/actor-b",
    }
    assert all(output.extra_fields["rollout_source"] == "hf_local_tq" for output in outputs)


def test_default_early_stop_does_not_collapse_wrong_two_turn_rollout():
    worker = _Worker(
        max_turns=2,
        responses=["Final answer: 999"] * 12,
    )

    outputs = build_hf_workflow_outputs(
        worker,
        recipe="comlrl_joint_math",
        prompt=_prompt(),
        session_id=0,
    )

    assert {output.extra_fields["turn_id"] for output in outputs} == {0, 1}
    assert {output.num_turns for output in outputs} == {2}
    assert not any(output.extra_fields["joint_stop_reason"] == "early_stop" for output in outputs)


def test_standard_madpo_all_tied_rollout_emits_inactive_marker_instead_of_failing():
    worker = _Worker(
        algorithm="madpo",
        responses=["Final answer: 999"] * 4,
    )

    outputs = build_hf_workflow_outputs(
        worker,
        recipe="comlrl_joint_math",
        prompt=_prompt(),
        session_id=0,
    )

    assert len(outputs) == 1
    assert outputs[0].extra_fields["preference_loss_mask"] == 0.0
    assert outputs[0].extra_fields["joint_stop_reason"] == "all_tied"
    assert outputs[0].num_turns == 1


@pytest.mark.parametrize("algorithm", ["magrpo", "marloo", "maremax"])
def test_group_relative_reinforce_algorithms_reject_singleton_candidate_groups(algorithm: str):
    with pytest.raises(ValueError, match="num_candidates>=2"):
        _Worker(algorithm=algorithm)._build_comlrl_joint_math_outputs(_prompt(num_candidates=1))


@pytest.mark.parametrize("algorithm", ["iac", "maac"])
def test_actor_critic_rollout_is_linear_and_shares_joint_transition_per_turn(algorithm: str):
    outputs = _Worker(algorithm=algorithm)._build_comlrl_joint_math_outputs(_prompt(num_candidates=1))

    assert len(outputs) == 2
    assert len({tuple(output.extra_fields["joint_action_ids"]) for output in outputs}) == 1
    assert len({tuple(output.extra_fields["joint_transition_ids"]) for output in outputs}) == 1
    assert len({output.extra_fields["turn_id"] for output in outputs}) == 1
    assert len({output.extra_fields["joint_reward"] for output in outputs}) == 1
    assert len({output.extra_fields["joint_done"] for output in outputs}) == 1
    assert len({output.extra_fields["joint_truncated"] for output in outputs}) == 1
    assert {output.extra_fields["worker_group"] for output in outputs} == {"actor-a", "actor-b"}


@pytest.mark.parametrize("algorithm", ["iac", "maac"])
def test_actor_critic_rejects_multiple_candidates_and_cross_mode(algorithm: str):
    with pytest.raises(ValueError, match="num_candidates=1"):
        _Worker(algorithm=algorithm)._build_comlrl_joint_math_outputs(_prompt(num_candidates=2))
    with pytest.raises(ValueError, match="joint_mode=aligned"):
        _Worker(algorithm=algorithm, joint_mode="cross")._build_comlrl_joint_math_outputs(_prompt(num_candidates=1))


def test_agent_loop_uses_rollout_n_once_as_joint_candidates(monkeypatch):
    from trajweave.backends.verl import agent_loop as module

    actor_class = module.TrajWeaveSyntheticAgentLoopWorkerTQ.__ray_actor_class__
    statuses = []

    async def fake_kv_put(**kwargs):
        statuses.append(kwargs["tag"]["status"])

    monkeypatch.setattr(module.tq, "async_kv_put", fake_kv_put)

    class FakeWorker(CoMLRLEmitterMixin):
        _run_prompt = actor_class._run_prompt

        def __init__(self):
            self.config = SimpleNamespace(
                trajweave=SimpleNamespace(
                    recipe="comlrl_joint_math",
                    agent_loop_backend="synthetic_tq",
                    comlrl={"algorithm": "magrpo", "joint_mode": "aligned"},
                ),
                agent={"agent_ids": ["alice", "bob"], "model_ids": ["actor-a", "actor-b"]},
                actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(n=3, val_kwargs=SimpleNamespace(n=2))),
            )
            self.build_calls = []
            self.put_calls = []

        def _build_comlrl_joint_math_outputs(self, prompt, *, session_id=0, validate=False):
            self.build_calls.append((prompt["__comlrl_num_candidates__"], session_id, validate))
            return [
                AgentLoopOutput(
                    prompt_ids=[1],
                    response_ids=[2],
                    response_mask=[1],
                    reward_score=1.0,
                    num_turns=1,
                    metrics=AgentLoopMetrics(
                        generate_sequences=1.0,
                        tool_calls=0.0,
                        compute_score=0.0,
                        num_preempted=-1,
                    ),
                    extra_fields={},
                )
            ]

        async def _put_outputs(self, outputs, validate, **kwargs):
            self.put_calls.append((len(outputs), validate, kwargs["session_id"]))

    worker = FakeWorker()
    asyncio.run(
        worker._run_prompt(
            {"uid": "prompt", "global_steps": 0},
            trajectory={"validate": False},
        )
    )

    assert worker.build_calls == [(3, 0, False)]
    assert worker.put_calls == [(1, False, 0)]
    assert statuses == ["running", "finished"]
