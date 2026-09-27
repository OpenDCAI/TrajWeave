import pytest

from trajweave.backends.policy import PolicyRequest, PolicyResponse
from trajweave.backends.verl.emitters.marti_mars2 import MARTIMARS2EmitterMixin
from trajweave.core import AgentSpec, PolicyGroupSpec, SearchNode, TeamSpec, TreeTrajectory
from trajweave.credit import TreePathCreditAllocator, discounted_path_returns, parent_sibling_shaped_rewards
from trajweave.orchestration import TreeSearchController, TreeSearchProtocol
from trajweave.verifiers import CallableVerifierAdapter, CodeVerifierAdapter, VerifierRequest, VerifierResult


class RecordingBackend:
    def __init__(self):
        self.requests: list[PolicyRequest] = []

    def generate(self, request: PolicyRequest) -> PolicyResponse:
        self.requests.append(request)
        node_id = int(request.metadata["node_id"])
        return PolicyResponse(
            text=f"candidate-{node_id}",
            token_ids=[node_id + 1],
            logprobs=[-0.1],
            metadata={"node_id": node_id},
        )


def make_team() -> TeamSpec:
    return TeamSpec(
        name="single-mcts",
        agents=(AgentSpec(name="generator", role="generator", policy_group="shared"),),
        policy_groups=(PolicyGroupSpec(name="shared"),),
        orchestra="mcts_tree_search",
        reward="code_verifier",
        credit="tree_path",
    )


def make_node(node_id: int, reward: float, *, parent_idx: int | None, path: tuple[int, ...]) -> SearchNode:
    return SearchNode(
        tree_id="tree-0",
        prompt_id="prompt-0",
        node_id=node_id,
        parent_idx=parent_idx,
        agent_name="generator",
        role="generator",
        policy_group="shared",
        prompt="solve",
        action_text=f"candidate-{node_id}",
        reward=reward,
        path=path,
    )


def test_marti_multi_agent_node_binding_is_deterministic_and_round_robin():
    emitter = MARTIMARS2EmitterMixin()
    emitter.config = {
        "agent": {
            "orchestra": {
                "marti_mars2": {
                    "agent_ids": ["generator", "critic"],
                    "model_ids": ["policy_a", "policy_b"],
                }
            }
        }
    }

    assert [emitter._marti_agent_binding(node_id) for node_id in range(5)] == [
        ("generator", "policy_a"),
        ("critic", "policy_b"),
        ("generator", "policy_a"),
        ("critic", "policy_b"),
        ("generator", "policy_a"),
    ]


def test_marti_fixture_candidate_decodes_hydra_escaped_newlines():
    emitter = MARTIMARS2EmitterMixin()
    emitter.config = {
        "agent": {
            "orchestra": {
                "marti_mars2": {
                    "fixture_candidates": [r"def add_one(x):\n    return x + 1"],
                }
            }
        }
    }

    assert emitter._marti_fixture_candidate(0) == "def add_one(x):\n    return x + 1"


def test_callable_verifier_adapter_normalizes_mapping_results():
    adapter = CallableVerifierAdapter(
        lambda request: {"reward": 0.75, "success": True, "terminal": False, "feedback": "partial", "case": 3}
    )

    result = adapter.verify(VerifierRequest(task_id="task", prompt="p", candidate="c", node_id=0))

    assert result == VerifierResult(
        score=0.75,
        success=True,
        terminal=False,
        feedback="partial",
        metadata={"case": 3},
    )


def test_code_verifier_adapter_reports_explicit_ground_truth_fallback():
    adapter = CodeVerifierAdapter()
    request = VerifierRequest(
        task_id="code-add-one",
        prompt="Implement add_one(x).",
        candidate="def add_one(x):\n    return x + 1",
        node_id=0,
        metadata={"reward_model": {"ground_truth": "return x + 1"}},
    )

    result = adapter.verify(request)

    assert result.success is True
    assert result.score == pytest.approx(1.0)
    assert result.metadata["verification_mode"] == "ground_truth_match"


def test_code_verifier_adapter_records_failed_ground_truth_feedback():
    result = CodeVerifierAdapter().verify(
        VerifierRequest(
            task_id="code-add-one",
            prompt="Implement add_one(x).",
            candidate="def add_one(x):\n    return x - 1",
            node_id=1,
            metadata={"reward_model": {"ground_truth": "return x + 1"}},
        )
    )

    assert result.success is False
    assert result.metadata["failure_type"] == "wrong_answer"
    assert "did not match" in result.feedback


def test_code_verifier_adapter_runs_call_based_smoke_without_pyext():
    result = CodeVerifierAdapter().verify(
        VerifierRequest(
            task_id="code-add-one",
            prompt="Implement add_one(x).",
            candidate="def add_one(x):\n    return x + 1",
            node_id=0,
            metadata={
                "reward_model": {
                    "test_cases": {
                        "fn_name": "add_one",
                        "inputs": ["1", "2", "5"],
                        "outputs": ["2", "3", "6"],
                    }
                }
            },
        )
    )

    assert result.success is True
    assert result.score == pytest.approx(1.0)
    assert result.metadata["verification_mode"] in {"prime_code_subprocess", "local_subprocess_fallback"}


def test_code_verifier_adapter_extracts_fenced_python_for_call_based_smoke():
    result = CodeVerifierAdapter().verify(
        VerifierRequest(
            task_id="code-clamp",
            prompt="Implement clamp_zero_ten(x).",
            candidate=(
                "Here is the implementation:\n```python\n"
                "def clamp_zero_ten(x):\n"
                "    return min(10, max(0, x))\n"
                "```\nThis clamps the input."
            ),
            node_id=0,
            metadata={
                "reward_model": {
                    "test_cases": {
                        "fn_name": "clamp_zero_ten",
                        "inputs": ["-3", "7", "14"],
                        "outputs": ["0", "7", "10"],
                    }
                }
            },
        )
    )

    assert result.success is True
    assert result.score == pytest.approx(1.0)
    assert result.metadata["verification_mode"] in {"prime_code_subprocess", "local_subprocess_fallback"}


def test_code_verifier_adapter_keeps_wrong_fenced_code_at_zero_reward():
    result = CodeVerifierAdapter().verify(
        VerifierRequest(
            task_id="code-clamp",
            prompt="Implement clamp_zero_ten(x).",
            candidate="```python\ndef clamp_zero_ten(x):\n    return x\n```",
            node_id=1,
            metadata={
                "reward_model": {
                    "test_cases": {
                        "fn_name": "clamp_zero_ten",
                        "inputs": ["-3", "7", "14"],
                        "outputs": ["0", "7", "10"],
                    }
                }
            },
        )
    )

    assert result.success is False
    assert result.score < 1.0


def test_marti_emitter_known_correct_candidate_sets_terminal_stop():
    pytest.importorskip("hydra")
    from trajweave.backends.verl.emitters.marti_mars2 import MARTIMARS2EmitterMixin

    class Harness(MARTIMARS2EmitterMixin):
        config = {
            "agent": {
                "orchestra": {
                    "marti_mars2": {
                        "max_num_nodes": 3,
                        "initial_candidates": 2,
                        "stop_on_success": True,
                    }
                }
            }
        }

    prompt = {
        "uid": "fixture-tree",
        "reward_model": {
            "test_cases": {
                "fn_name": "add_one",
                "inputs": ["1", "2"],
                "outputs": ["2", "3"],
            }
        },
        "raw_prompt": "Implement add_one(x).",
    }
    result = Harness()._marti_verify(
        prompt,
        node_id=0,
        parent_idx=-1,
        candidate="def add_one(x):\n    return x + 1",
    )

    assert result.score == pytest.approx(1.0)
    assert result.success is True
    assert result.terminal is True
    assert len(result.metadata["candidate_sha256"]) == 64
    assert result.metadata["candidate_preview"].startswith("def add_one")
    assert result.metadata["verification_mode"] in {"prime_code_subprocess", "local_subprocess_fallback"}


def test_tree_search_protocol_runs_selection_expansion_refinement_and_aggregation():
    backend = RecordingBackend()
    verifier = CallableVerifierAdapter(
        lambda request: {
            "score": [0.2, 0.8, 1.0][request.node_id],
            "success": request.node_id == 2,
            "terminal": False,
            "feedback": f"feedback-{request.node_id}",
        },
        name="recording_verifier",
    )
    protocol = TreeSearchProtocol(max_num_nodes=3, initial_candidates=1)

    tree = protocol.run(
        tree_id="tree-0",
        prompt_id="prompt-0",
        task=type("Task", (), {"task_id": "task-0"})(),
        team=make_team(),
        observation="implement add_one",
        policy_backend=backend,
        verifier=verifier,
    )

    assert [node.parent_idx for node in tree.nodes] == [None, 0, 1]
    assert [node.path for node in tree.nodes] == [(0,), (0, 1), (0, 1, 2)]
    assert backend.requests[1].metadata["search_stage"] == "refine"
    assert "Previous candidate" in backend.requests[1].prompt
    assert tree.final_answer == "candidate-2"
    assert tree.global_reward == 1.0
    assert tree.success is True
    assert tree.metadata["best_node_id"] == 2


def test_tree_search_controller_persists_ucb_state_and_paths():
    controller = TreeSearchController(max_num_nodes=4, initial_candidates=2)
    assert controller.select_parent(0) == -1
    assert controller.select_parent(1) == -1
    controller.record(
        node_id=0,
        parent_idx=-1,
        path=(0,),
        reward=0.2,
        feedback="retry",
        success=False,
        terminal=False,
    )
    controller.record(
        node_id=1,
        parent_idx=-1,
        path=(1,),
        reward=0.8,
        feedback="retry",
        success=False,
        terminal=False,
    )
    assert controller.select_parent(2) in {0, 1}
    assert controller.path_for(2, 1) == (1, 2)
    controller.record(
        node_id=2,
        parent_idx=1,
        path=(1, 2),
        reward=1.0,
        feedback="passed",
        success=True,
        terminal=True,
    )
    assert controller.select_parent(3) in {0, 1}
    assert controller.select_parent(3) != 2


def test_tree_search_controller_stops_after_success_when_enabled():
    controller = TreeSearchController(max_num_nodes=4, initial_candidates=2, stop_on_success=True)
    controller.record(
        node_id=0,
        parent_idx=-1,
        path=(0,),
        reward=0.0,
        feedback="retry",
        success=False,
        terminal=False,
    )
    controller.record(
        node_id=1,
        parent_idx=-1,
        path=(1,),
        reward=1.0,
        feedback="passed",
        success=True,
        terminal=True,
    )
    assert controller.should_stop(success=True, pending_node=False) is True


def test_parent_sibling_and_path_credit_match_marti_shaping_semantics():
    nodes = [
        make_node(0, 0.4, parent_idx=None, path=(0,)),
        make_node(1, 0.2, parent_idx=None, path=(1,)),
        make_node(2, 1.0, parent_idx=0, path=(0, 2)),
        make_node(3, 0.0, parent_idx=0, path=(0, 3)),
    ]

    shaped = parent_sibling_shaped_rewards(nodes, gamma=0.5, sibling_mix=0.5)
    path_returns = discounted_path_returns(nodes, shaped, discount=0.5)

    assert shaped == pytest.approx([0.5, 0.1, 1.4, -0.35])
    assert path_returns == pytest.approx([0.5, 0.1, 1.65, -0.1])


def test_tree_credit_keeps_frozen_ancestors_in_path_context():
    frozen = AgentSpec(name="planner", role="planner", policy_group="shared", trainable=False)
    generator = AgentSpec(name="generator", role="generator", policy_group="shared", trainable=True)
    team = TeamSpec(
        name="mixed-tree",
        agents=(frozen, generator),
        policy_groups=(PolicyGroupSpec(name="shared"),),
        orchestra="mcts_tree_search",
        reward="code_verifier",
        credit="tree_path",
    )
    root = make_node(0, 1.0, parent_idx=None, path=(0,))
    root.agent_name = "planner"
    child = make_node(1, 0.0, parent_idx=0, path=(0, 1))
    tree = TreeTrajectory(tree_id="tree-0", prompt_id="prompt-0", task_id="task", rollout_group="tree-0")
    tree.add_node(root)
    tree.add_node(child)

    samples = TreePathCreditAllocator(normalize_advantage=False).assign_tree(tree, team)

    assert len(samples) == 1
    assert samples[0].agent_name == "generator"
    assert samples[0].reward == pytest.approx(0.0)


def test_prime_partial_credit_keeps_call_based_function_name(monkeypatch):
    import sys
    import types
    from trajweave.verifiers.code import _score_prime_call_cases

    calls = []

    def check(cases, code, **kwargs):
        assert cases['fn_name'] == 'identity'
        calls.append(cases)
        return [raw == expected for raw, expected in zip(cases['inputs'], cases['outputs'])], []

    monkeypatch.setitem(sys.modules, 'verl.utils.reward_score.prime_code.utils',
                        types.SimpleNamespace(check_correctness=check))
    cases = {'fn_name': 'identity', 'inputs': ['1', '2'], 'outputs': ['1', '3']}
    score, _ = _score_prime_call_cases('def identity(x): return x', cases, 5)
    assert score == pytest.approx(0.5)
    assert len(calls) == 3
    assert cases['inputs'] == ['1', '2']


def test_prime_failure_after_tenth_case_is_not_terminal_success(monkeypatch):
    import sys
    import types
    from trajweave.verifiers.code import _score_prime_call_cases

    def check(cases, code, **kwargs):
        return [raw == expected for raw, expected in zip(cases['inputs'], cases['outputs'])], []

    monkeypatch.setitem(sys.modules, 'verl.utils.reward_score.prime_code.utils',
                        types.SimpleNamespace(check_correctness=check))
    inputs = [str(i) for i in range(11)]
    cases = {'fn_name': 'identity', 'inputs': inputs, 'outputs': inputs[:10] + ['-1']}
    score, _ = _score_prime_call_cases('def identity(x): return x', cases, 5)
    assert score == pytest.approx(10 / 11)
    assert score < 1.0
