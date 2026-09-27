from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pytest

from trajweave.backends.policy import PolicyRequest, PolicyResponse, StableByteTokenizer
from trajweave.core.specs import AgentSpec, PolicyGroupSpec, TeamSpec
from trajweave.core.trajectory import MultiAgentTrajectory
from trajweave.envs.comlrl import JointMathEnvironment, JointStepResult
from trajweave.envs.math import MathTask
from trajweave.orchestration.comlrl import (
    FullJointTreeBuilder,
    canonical_joint_mode,
    compose_joint_action_indices,
)


@dataclass
class _DeterministicBackend:
    events: list[tuple[str, str, str, int]]
    tokenizer: StableByteTokenizer = field(default_factory=StableByteTokenizer)
    requests: list[PolicyRequest] = field(default_factory=list)

    def generate(self, request: PolicyRequest) -> PolicyResponse:
        self.requests.append(request)
        node_id = request.metadata["tree_node_id"]
        candidate_index = request.metadata["candidate_index"]
        self.events.append(("generate", node_id, request.agent.name, candidate_index))
        text = f"{node_id}|{request.agent.name}|candidate-{candidate_index}"
        token_ids = self.tokenizer.encode(text)
        return PolicyResponse(text=text, token_ids=token_ids, logprobs=[0.0] * len(token_ids))


@dataclass
class _RecordingJointEnvironment:
    events: list[tuple[str, str, str, int]]
    reward: float | Callable[[dict[str, str], int], float] = 0.0
    done: Callable[[dict[str, str], int], bool] = lambda _actions, _turn: False
    calls: list[dict] = field(default_factory=list)

    def step_joint(
        self,
        task: MathTask,
        observations_by_agent: dict[str, str],
        actions_by_agent: dict[str, str],
        histories_by_agent: dict[str, list[str]],
        turn_index: int,
    ) -> JointStepResult:
        del task
        node_id = next(iter(actions_by_agent.values())).split("|", maxsplit=1)[0]
        self.events.append(("transition", node_id, "", turn_index))
        call_index = len(self.calls)
        recorded = {
            "node_id": node_id,
            "observations": dict(observations_by_agent),
            "actions": dict(actions_by_agent),
            "histories": {name: list(history) for name, history in histories_by_agent.items()},
            "turn_index": turn_index,
        }
        self.calls.append(recorded)
        histories_by_agent["alice"].append("environment mutation")
        team_reward = self.reward(actions_by_agent, turn_index) if callable(self.reward) else self.reward
        is_done = self.done(actions_by_agent, turn_index)
        next_observations = {
            name: f"next-{call_index}-{name}-after-{actions_by_agent[name]}" for name in actions_by_agent
        }
        return JointStepResult(
            team_reward=float(team_reward),
            next_observations_by_agent=next_observations,
            done=is_done,
            metadata={"call_index": call_index},
        )


def _team(*, max_turns: int = 3) -> TeamSpec:
    agents = (
        AgentSpec(name="alice", role="solver", policy_group="alice-policy"),
        AgentSpec(name="bob", role="reviewer", policy_group="bob-policy"),
    )
    return TeamSpec(
        name="joint-team",
        agents=agents,
        policy_groups=(
            PolicyGroupSpec(name="alice-policy"),
            PolicyGroupSpec(name="bob-policy"),
        ),
        orchestra="comlrl_full_tree",
        reward="joint",
        credit="none",
        max_turns=max_turns,
    )


def _build(
    *,
    mode: str = "aligned",
    max_turns: int = 3,
    reward: float | Callable[[dict[str, str], int], float] = 0.0,
    done: Callable[[dict[str, str], int], bool] = lambda _actions, _turn: False,
    early_stop_threshold: float | None = None,
    max_joint_actions: int | None = None,
    max_tree_nodes: int | None = None,
):
    events: list[tuple[str, str, str, int]] = []
    backend = _DeterministicBackend(events)
    environment = _RecordingJointEnvironment(events, reward=reward, done=done)
    tree = FullJointTreeBuilder(
        num_candidates=2,
        joint_mode=mode,
        max_turns=max_turns,
        max_joint_actions=max_joint_actions,
        max_tree_nodes=max_tree_nodes,
        early_stop_threshold=early_stop_threshold,
    ).build(
        root_id="episode-0",
        task=MathTask(task_id="math-0", question="What is 1 + 1?", answer=2),
        team=_team(max_turns=max_turns),
        observations_by_agent={"alice": "alice-root", "bob": "bob-root"},
        policy_backend=backend,
        environment=environment,
    )
    return tree, backend, environment, events


def test_joint_action_indices_cover_aligned_cross_and_crossed_alias():
    assert canonical_joint_mode("align") == "aligned"
    assert canonical_joint_mode("aligned") == "aligned"
    assert canonical_joint_mode("cross") == "cross"
    assert canonical_joint_mode("crossed") == "cross"
    assert compose_joint_action_indices(("alice", "bob"), 2, "aligned") == [(0, 0), (1, 1)]
    assert compose_joint_action_indices(("alice", "bob"), 2, "crossed") == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    assert compose_joint_action_indices(("alice", "bob"), 10_000, "cross", limit=2) == [
        (0, 0),
        (0, 1),
    ]

    with pytest.raises(ValueError, match="unsupported"):
        canonical_joint_mode("diagonal")
    with pytest.raises(ValueError, match="unique"):
        compose_joint_action_indices(("alice", "alice"), 2, "aligned")
    with pytest.raises(ValueError, match="positive"):
        compose_joint_action_indices(("alice", "bob"), 0, "cross")


def test_aligned_n2_k2_depth3_builds_full_tree_with_parent_child_links():
    tree, _backend, _environment, _events = _build()

    assert len(tree.nodes) == 7
    assert len(tree.joint_actions) == 14
    assert len(tree.transitions) == 14
    assert len(tree.completions) == 28
    assert [sum(node.depth == depth for node in tree.nodes) for depth in range(3)] == [1, 2, 4]
    assert len({node.tree_node_id for node in tree.nodes}) == len(tree.nodes)
    assert len({action.joint_action_id for action in tree.joint_actions}) == len(tree.joint_actions)
    assert len({transition.joint_transition_id for transition in tree.transitions}) == len(tree.transitions)

    actions_by_id = {action.joint_action_id: action for action in tree.joint_actions}
    for node in tree.nodes[1:]:
        parent = actions_by_id[node.parent_joint_action_id]
        assert parent.child_tree_node_id == node.tree_node_id
        assert node.depth == next(
            source.depth + 1 for source in tree.nodes if source.tree_node_id == parent.tree_node_id
        )
    tree.validate()


def test_joint_tree_result_attaches_to_storage_trajectory():
    tree, *_ = _build(max_turns=1)
    trajectory = MultiAgentTrajectory(
        episode_id="episode-0",
        task_id="math-0",
        rollout_group="group-0",
        team_name="joint-test",
    )

    attached = tree.attach_to_trajectory(trajectory)

    assert attached is trajectory
    assert trajectory.joint_nodes == tree.nodes
    assert trajectory.joint_completions == tree.completions
    assert trajectory.joint_actions == tree.joint_actions
    assert trajectory.joint_transitions == tree.transitions
    assert trajectory.metadata["joint_tree"]["root_id"] == tree.root_id


def test_each_joint_action_has_independent_transition_observation_and_history():
    tree, _backend, environment, _events = _build(mode="cross", max_turns=2)
    root_actions = [action for action in tree.joint_actions if action.tree_node_id == tree.root_id]
    completions = {completion.completion_id: completion for completion in tree.completions}
    nodes = {node.tree_node_id: node for node in tree.nodes}
    transitions = {transition.joint_action_id: transition for transition in tree.transitions}

    assert len(root_actions) == 4
    assert all(call["histories"] == {"alice": [], "bob": []} for call in environment.calls[:4])
    assert len({tuple(call["observations"].values()) for call in environment.calls[:4]}) == 1
    for action in root_actions:
        child = nodes[action.child_tree_node_id]
        expected_history = {
            name: [completions[completion_id].text] for name, completion_id in action.completion_ids.items()
        }
        assert child.metadata["histories_by_agent"] == expected_history
        assert "environment mutation" not in child.metadata["histories_by_agent"]["alice"]
        assert (
            child.metadata["observations_by_agent"]
            == transitions[action.joint_action_id].metadata["next_observations_by_agent"]
        )
    assert (
        len(
            {
                tuple(nodes[action.child_tree_node_id].metadata["observations_by_agent"].values())
                for action in root_actions
            }
        )
        == 4
    )


def test_attach_to_trajectory_preserves_cross_sampling_mode_for_credit():
    tree, *_ = _build(mode="cross", max_turns=1)
    trajectory = MultiAgentTrajectory("episode-0", "math-0", "group-0", "joint-team")

    tree.attach_to_trajectory(trajectory)

    assert trajectory.metadata["joint_sampling_mode"] == "cross"
    assert trajectory.metadata["joint_tree"]["joint_mode"] == "cross"


def test_node_level_early_stop_is_strictly_greater_than_threshold():
    equal_tree, *_ = _build(max_turns=2, reward=0.5, early_stop_threshold=0.5)
    greater_tree, *_ = _build(max_turns=2, reward=0.5, early_stop_threshold=0.49)

    assert len(equal_tree.nodes) == 3
    assert equal_tree.nodes[0].metadata["stopped_early"] is False
    assert len(greater_tree.nodes) == 1
    assert greater_tree.nodes[0].metadata["stopped_early"] is True
    assert greater_tree.nodes[0].truncated is True
    assert greater_tree.nodes[0].stop_reason == "early_stop"
    assert all(action.truncated and action.stop_reason == "early_stop" for action in greater_tree.joint_actions)
    assert all(
        transition.truncated and transition.stop_reason == "early_stop" for transition in greater_tree.transitions
    )
    assert all(action.shared_return is None for action in equal_tree.joint_actions + greater_tree.joint_actions)


def test_max_turns_marks_non_terminal_leaf_actions_as_truncated():
    tree, *_ = _build(max_turns=1)

    assert len(tree.nodes) == 1
    assert tree.nodes[0].truncated is True
    assert tree.nodes[0].stop_reason == "max_turns"
    assert all(action.truncated and action.stop_reason == "max_turns" for action in tree.joint_actions)
    assert all(transition.truncated and transition.stop_reason == "max_turns" for transition in tree.transitions)
    tree.validate()


def test_done_joint_action_does_not_expand_but_its_sibling_does():
    def done(actions: dict[str, str], _turn: int) -> bool:
        return all("candidate-0" in action for action in actions.values())

    tree, *_ = _build(max_turns=2, done=done)
    root_actions = [action for action in tree.joint_actions if action.tree_node_id == tree.root_id]

    assert [action.done for action in root_actions] == [True, False]
    assert root_actions[0].child_tree_node_id is None
    assert root_actions[1].child_tree_node_id is not None
    assert len(tree.nodes) == 2


def test_tree_and_joint_action_limits_are_hard_caps():
    node_limited, *_ = _build(mode="cross", max_turns=3, max_tree_nodes=3)

    assert len(node_limited.nodes) == 3
    assert node_limited.metadata["max_tree_nodes_reached"] is True
    node_limited.validate()

    with pytest.raises(ValueError, match="complete root joint-action group"):
        _build(mode="cross", max_turns=3, max_joint_actions=3)


def test_max_joint_actions_terminalizes_incoming_edges_for_unexpanded_children():
    tree, *_ = _build(max_turns=2, max_joint_actions=2)

    assert len(tree.nodes) == 1
    assert len(tree.joint_actions) == 2
    assert tree.metadata["max_joint_actions_reached"] is True
    assert all(action.truncated and action.stop_reason == "max_joint_actions" for action in tree.joint_actions)
    assert all(action.child_tree_node_id is None for action in tree.joint_actions)
    assert all(
        transition.truncated
        and transition.stop_reason == "max_joint_actions"
        and transition.target_tree_node_id is None
        and transition.target_turn is None
        for transition in tree.transitions
    )
    tree.validate()


def test_every_agent_finishes_candidate_generation_before_any_node_transition():
    tree, _backend, _environment, events = _build(mode="cross", max_turns=2)

    for node in tree.nodes:
        node_events = [event for event in events if event[1] == node.tree_node_id]
        generate_positions = [index for index, event in enumerate(node_events) if event[0] == "generate"]
        transition_positions = [index for index, event in enumerate(node_events) if event[0] == "transition"]
        assert len(generate_positions) == 4
        assert transition_positions
        assert max(generate_positions) < min(transition_positions)
        generated_agents = [event[2] for event in node_events if event[0] == "generate"]
        assert generated_agents == ["alice", "alice", "bob", "bob"]


def test_joint_math_adapter_uses_existing_evaluator_and_emits_observation_for_every_agent():
    environment = JointMathEnvironment()
    task = MathTask(task_id="math-smoke", question="What is 1 + 1?", answer=2)

    result = environment.step_joint(
        task,
        {"alice": task.question, "bob": task.question},
        {"alice": "Final answer: 2", "bob": "Final answer: 3"},
        {"alice": [], "bob": []},
        0,
    )

    assert result.team_reward == 0.5
    assert result.done is False
    assert set(result.next_observations_by_agent) == {"alice", "bob"}
    assert result.metadata["agent_correct"] == {"alice": True, "bob": False}
