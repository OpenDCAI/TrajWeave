from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from trajweave.backends.policy import PolicyRequest, PolicyResponse
from trajweave.backends.verl.workflow_runtime import build_hf_workflow_outputs, build_synthetic_c3_workflow_outputs
from trajweave.envs.math.c3 import C3MathEnvironment, C3MathTask, parse_c3_math_answer
from trajweave.orchestration.c3 import C3PrefixTreeOrchestra
from trajweave.recipes.c3 import default_c3_team


@pytest.mark.parametrize("prediction,expected", [
    ("Final answer: 73 * 24 = 1752", (1.0, True)),
    (r"Final answer: \sqrt{3069504}", (1.0, True)),
    ("Final answer: 73 * 25 = 1825", (0.0, False)),
])
def test_math_verify_agrees_in_ray_style_background_thread(prediction, expected):
    pytest.importorskip("math_verify")
    environment = C3MathEnvironment()
    task = C3MathTask("threaded", "What is 73 * 24?", "1752")
    assert environment.evaluate(task, prediction) == expected
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(environment.evaluate, task, prediction).result(timeout=20) == expected


@pytest.mark.parametrize(
    ("ground_truth", "prediction"),
    [
        ("0.5", "Final answer: 5"),
        (r"\boxed{\frac{1}{2}}", "Final answer: 2"),
        ("1,234", "Final answer: 234"),
        ("4", "Final answer: 4/5"),
        ("4.0", "Final answer: 0"),
        ("x+1", "Final answer: 1"),
        ("4", "I cannot answer. Confidence score: 4"),
    ],
)
def test_c3_math_environment_rejects_integer_truncation_and_trailing_number_false_positives(ground_truth, prediction):
    reward, success = C3MathEnvironment().evaluate(
        C3MathTask(task_id="task", question="question", answer=ground_truth),
        prediction,
    )
    assert reward == 0.0
    assert success is False


@pytest.mark.parametrize(
    ("ground_truth", "prediction"),
    [
        ("1/2", "Final answer: 0.5"),
        ("0.5", r"The result is \boxed{\frac{1}{2}}."),
        (r"\boxed{\frac{1}{2}}", r"Final answer: \frac{1}{2}"),
        ("-5", "Final answer: -5"),
        ("-0.5", r"\boxed{-\frac{1}{2}}"),
        ("-0.5", "Final answer: -0 1/2"),
        ("1,234", r"\boxed{1234}"),
    ],
)
def test_c3_math_environment_accepts_equivalent_string_answers(ground_truth, prediction):
    reward, success = C3MathEnvironment(use_math_verify=False).evaluate(
        C3MathTask(task_id="task", question="question", answer=ground_truth),
        prediction,
    )
    assert reward == 1.0
    assert success is True


def test_c3_math_parser_uses_the_last_self_corrected_answer():
    response = "Final answer: 4\nCorrection: that was wrong.\nFinal answer: 5"
    assert parse_c3_math_answer(response) == ("5", "anchor")

    environment = C3MathEnvironment(use_math_verify=False)
    assert environment.evaluate(C3MathTask("correct", "question", "5"), response) == (1.0, True)
    assert environment.evaluate(C3MathTask("wrong", "question", "4"), response) == (0.0, False)


@pytest.mark.parametrize(
    "response",
    [
        "First attempt: \\boxed{4}.\nCorrection.\nFinal answer: 5",
        "Final answer: 4\nCorrection.\nThe result is \\boxed{5}.",
        "#### 4\nCorrection.\nFinal answer: 5",
    ],
)
def test_c3_math_parser_uses_the_last_marker_across_answer_formats(response):
    environment = C3MathEnvironment(use_math_verify=False)

    assert parse_c3_math_answer(response)[0] == "5"
    assert environment.evaluate(C3MathTask("correct", "question", "5"), response) == (1.0, True)
    assert environment.evaluate(C3MathTask("wrong", "question", "4"), response) == (0.0, False)


def test_c3_actor_prompt_prefers_boxed_answers_but_keeps_the_explicit_anchor():
    prompt = C3PrefixTreeOrchestra._build_prompt(
        role="actor",
        observation="What is one half?",
        role_outputs={"reasoner": "Divide one by two."},
    )
    assert r"\boxed{...}" in prompt
    assert "Final answer: ..." in prompt


@pytest.mark.parametrize(
    ("ground_truth", "prediction"),
    [
        ("4", "4\n```output\n4\n```"),
        ("4", "4\nValueError: unavailable tool"),
        ("<|im_start|>assistant\n4<|im_end|>", "<|assistant|>4</s>"),
    ],
)
def test_c3_math_environment_sanitizes_tool_noise_only_for_judging(ground_truth, prediction):
    task = C3MathTask(task_id="sanitize-reward", question="What is 2 + 2?", answer=ground_truth)

    assert C3MathEnvironment(use_math_verify=False).evaluate(task, prediction) == (1.0, True)


class _ContaminatedReasonerBackend:
    raw_reasoner_output = (
        "<|im_start|>assistant\n"
        "**Reasoner** Use multiplication, then report the product.\n"
        "```python\nprint(73 * 24)\n```\n"
        "```output\n1752\n```\n"
        "Traceback: tool execution failed\n"
        "ValueError: unavailable tool\n"
        "<|im_end|>"
    )

    def __init__(self):
        self.actor_requests: list[PolicyRequest] = []

    def generate(self, request: PolicyRequest) -> PolicyResponse:
        if request.agent.name == "reasoner":
            text = self.raw_reasoner_output
        else:
            self.actor_requests.append(request)
            text = "Final answer: 1752"
        return PolicyResponse(text=text, token_ids=[1], logprobs=[0.0])


def test_c3_sanitizes_only_actor_visible_context_and_preserves_raw_q_prefix():
    backend = _ContaminatedReasonerBackend()
    task = C3MathTask(task_id="sanitize", question="What is 73 * 24?", answer="1752")
    trajectory = C3PrefixTreeOrchestra(fanout=(2, 2)).run_tree(
        episode_id="sanitize",
        rollout_group="sanitize",
        task=task,
        team=default_c3_team(),
        observation=task.question,
        policy_backend=backend,
        environment=C3MathEnvironment(use_math_verify=False),
    )

    assert len(backend.actor_requests) == 4
    for request in backend.actor_requests:
        for contaminated in ("<|im_start|>", "```python", "```output", "Traceback", "ValueError"):
            assert contaminated not in request.prompt
            assert contaminated not in request.team_context
        assert "Use multiplication, then report the product." in request.prompt
        assert "Use multiplication, then report the product." in request.team_context

    reasoner_turns = [turn for turn in trajectory.turns if turn.agent_name == "reasoner"]
    actor_turns = [turn for turn in trajectory.turns if turn.agent_name == "actor"]
    assert all(turn.action_text == backend.raw_reasoner_output for turn in reasoner_turns)
    assert all(
        turn.metadata["c3_prefix_outputs"]["reasoner"] == backend.raw_reasoner_output for turn in trajectory.turns
    )
    assert all(backend.raw_reasoner_output in turn.metadata["c3_prefix_text"] for turn in actor_turns)


def test_c3_math_environment_has_a_deterministic_fallback_when_math_verify_is_unavailable(monkeypatch):
    import trajweave.envs.math.c3 as c3_math

    monkeypatch.setattr(c3_math, "_math_verify_equal", lambda *_args, **_kwargs: None)
    environment = C3MathEnvironment(use_math_verify=True)
    task = C3MathTask("task", "question", r"\sqrt{4}")

    assert environment.evaluate(task, r"Final answer: \sqrt{9}") == (0.0, False)
    assert environment.evaluate(task, r"Final answer: \sqrt{4}") == (1.0, True)


def test_c3_math_verify_receives_only_the_selected_final_answer(monkeypatch):
    import trajweave.envs.math.c3 as c3_math

    calls = []

    def recording_verify(predicted, expected):
        calls.append((predicted, expected))
        return False

    monkeypatch.setattr(c3_math, "_math_verify_equal", recording_verify)
    response = r"First attempt: \boxed{\sqrt{4}}. Correction. Final answer: \sqrt{9}"

    assert C3MathEnvironment().evaluate(C3MathTask("task", "question", r"\sqrt{4}"), response) == (0.0, False)
    assert calls == [(r"\sqrt{9}", r"\sqrt{4}")]


class _TreeBackend:
    def generate(self, request: PolicyRequest) -> PolicyResponse:
        depth = int(request.metadata["c3_depth"])
        branch = int(request.metadata["c3_branch_index"])
        if depth == 0:
            text = f"plan-{branch}"
        else:
            parent_branch = int(str(request.metadata["c3_parent_node_id"]).split(":")[-1])
            text = "Final answer: 0.5" if (parent_branch, branch) == (0, 0) else "Final answer: 2"
        return PolicyResponse(text=text, token_ids=[depth + 1, branch + 1], logprobs=[0.0, 0.0])


def test_c3_string_math_rewards_propagate_through_the_complete_prefix_tree():
    task = C3MathTask(task_id="fraction", question="What is one half?", answer=r"\boxed{\frac{1}{2}}")
    trajectory = C3PrefixTreeOrchestra(fanout=(2, 2)).run_tree(
        episode_id="episode",
        rollout_group="fraction",
        task=task,
        team=default_c3_team(),
        observation=task.question,
        policy_backend=_TreeBackend(),
        environment=C3MathEnvironment(use_math_verify=False),
    )

    leaves = [turn for turn in trajectory.turns if turn.metadata["c3_is_leaf"]]
    reasoners = [turn for turn in trajectory.turns if not turn.metadata["c3_is_leaf"]]
    assert [turn.reward for turn in leaves] == [1.0, 0.0, 0.0, 0.0]
    assert [turn.metadata["c3_subtree_return"] for turn in reasoners] == [0.5, 0.0]
    assert trajectory.global_reward == 1.0
    assert trajectory.success is True
    assert trajectory.final_answer == "Final answer: 0.5"


class _SyntheticWorker:
    config = SimpleNamespace()

    @staticmethod
    def _c3_model_ids():
        return ("reasoner", "actor")

    @staticmethod
    def _c3_fanout():
        return (2, 2)

    @staticmethod
    def _encode_text(text):
        return [ord(char) % 127 for char in text]

    _encode_prompt_text = _encode_text

    @staticmethod
    def _local_policy_version():
        return 0


class _HFWorker(_SyntheticWorker):
    def __init__(self, responses):
        self.responses = list(responses)
        self.model_config = SimpleNamespace(local_path="/models/default")

    @staticmethod
    def _encode_prompt_text(text):
        return [ord(char) % 127 for char in text]

    @staticmethod
    def _decode_response_ids(token_ids):
        return "".join(chr(token_id) for token_id in token_ids)

    def _generate_local_response_ids(self, _prompt_ids, **_kwargs):
        return [ord(char) for char in self.responses.pop(0)]

    @staticmethod
    def _worker_group_model_path(group_id):
        return f"/models/{group_id}"


def test_c3_synthetic_runtime_preserves_raw_fractional_ground_truth():
    outputs = build_synthetic_c3_workflow_outputs(
        _SyntheticWorker(),
        prompt={
            "uid": "fraction-runtime",
            "raw_prompt": [{"role": "user", "content": "What is 2 + 3?"}],
            "reward_model": {"ground_truth": "0.5"},
        },
        session_id=0,
    )

    assert len(outputs) == 6
    assert all(output.reward_score == 0.0 for output in outputs)
    assert all(output.extra_fields["workflow_evaluation_reward"] == 0.0 for output in outputs)
    assert all(output.extra_fields["workflow_success"] is False for output in outputs)


def test_c3_hf_runtime_preserves_raw_fractional_ground_truth():
    outputs = build_hf_workflow_outputs(
        _HFWorker(
            [
                "plan-0",
                "plan-1",
                "Final answer: 5",
                "Final answer: 5",
                "Final answer: 5",
                "Final answer: 5",
            ]
        ),
        recipe="c3_reasoner_actor_math",
        prompt={
            "uid": "fraction-hf-runtime",
            "raw_prompt": [{"role": "user", "content": "What is one half?"}],
            "reward_model": {"ground_truth": "0.5"},
        },
        session_id=0,
    )

    leaves = [output for output in outputs if output.extra_fields["c3_is_leaf"]]
    assert len(outputs) == 6
    assert len(leaves) == 4
    assert all(output.reward_score == 0.0 for output in outputs)
    assert all(output.extra_fields["workflow_success"] is False for output in outputs)
