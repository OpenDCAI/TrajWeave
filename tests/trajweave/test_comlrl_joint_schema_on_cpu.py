from dataclasses import replace

import pytest

from trajweave.core.joint_trajectory import (
    JointAction,
    JointCompletion,
    JointTransition,
    JointTreeNode,
    validate_joint_trajectory,
)
from trajweave.core.preference import JointPreferencePair


def joint_tree_records():
    nodes = [
        JointTreeNode(tree_node_id="node-root", root_id="node-root", depth=0),
        JointTreeNode(
            tree_node_id="node-child",
            root_id="node-root",
            depth=1,
            parent_joint_action_id="action-expand",
        ),
    ]
    completions = [
        JointCompletion("completion-a0", "node-root", "agent-a", 0, "a0", [10], [-0.1]),
        JointCompletion("completion-a1", "node-root", "agent-a", 1, "a1", [11], [-0.2]),
        JointCompletion("completion-b0", "node-root", "agent-b", 0, "b0", [20], [-0.3]),
        JointCompletion("completion-b1", "node-root", "agent-b", 1, "b1", [21], [-0.4]),
    ]
    actions = [
        JointAction(
            joint_action_id="action-expand",
            tree_node_id="node-root",
            completion_ids={"agent-a": "completion-a0", "agent-b": "completion-b0"},
            candidate_indices={"agent-a": 0, "agent-b": 0},
            shared_reward=1.0,
            shared_return=1.5,
            child_tree_node_id="node-child",
            joint_transition_id="transition-expand",
        ),
        JointAction(
            joint_action_id="action-terminal",
            tree_node_id="node-root",
            completion_ids={"agent-a": "completion-a0", "agent-b": "completion-b1"},
            candidate_indices={"agent-a": 0, "agent-b": 1},
            shared_reward=0.25,
            shared_return=0.25,
            done=True,
            joint_transition_id="transition-terminal",
        ),
    ]
    transitions = [
        JointTransition(
            joint_transition_id="transition-expand",
            joint_action_id="action-expand",
            source_tree_node_id="node-root",
            source_turn=0,
            target_tree_node_id="node-child",
            target_turn=1,
        ),
        JointTransition(
            joint_transition_id="transition-terminal",
            joint_action_id="action-terminal",
            source_tree_node_id="node-root",
            source_turn=0,
            done=True,
        ),
    ]
    return nodes, completions, actions, transitions


def test_joint_tree_validation_allows_cross_action_completion_reuse():
    nodes, completions, actions, transitions = joint_tree_records()

    validate_joint_trajectory(nodes, completions, actions, transitions, {"agent-a", "agent-b"})

    assert actions[0].completion_ids["agent-a"] == actions[1].completion_ids["agent-a"]


def test_joint_tree_validation_rejects_bad_references():
    nodes, completions, actions, transitions = joint_tree_records()
    actions[1] = replace(
        actions[1],
        completion_ids={"agent-a": "missing-completion", "agent-b": "completion-b1"},
    )

    with pytest.raises(ValueError, match="unknown completion"):
        validate_joint_trajectory(nodes, completions, actions, transitions, {"agent-a", "agent-b"})


def test_joint_tree_validation_requires_exactly_one_transition_per_action():
    nodes, completions, actions, transitions = joint_tree_records()

    with pytest.raises(ValueError, match="exactly one transition"):
        validate_joint_trajectory(nodes, completions, actions, transitions[:-1], {"agent-a", "agent-b"})


def test_joint_tree_validation_rejects_inconsistent_transition_turns():
    nodes, completions, actions, transitions = joint_tree_records()
    transitions[0] = replace(transitions[0], source_turn=2, target_turn=3)

    with pytest.raises(ValueError, match="source node or turn"):
        validate_joint_trajectory(nodes, completions, actions, transitions, {"agent-a", "agent-b"})


def test_joint_tree_validation_rejects_terminal_action_with_child():
    nodes, completions, actions, transitions = joint_tree_records()
    actions[0] = replace(actions[0], done=True)

    with pytest.raises(ValueError, match="cannot have a child"):
        validate_joint_trajectory(nodes, completions, actions, transitions, {"agent-a", "agent-b"})


def test_joint_preference_pair_requires_matching_agent_keys():
    with pytest.raises(ValueError, match="identical agent keys"):
        JointPreferencePair(
            preference_pair_id="pair-mismatched",
            episode_id="episode-1",
            tree_node_id="node-root",
            chosen_joint_action_id="action-chosen",
            rejected_joint_action_id="action-rejected",
            prompts_by_agent={"agent-a": "prompt a", "agent-b": "prompt b"},
            chosen_by_agent={"agent-a": "chosen a"},
            rejected_by_agent={"agent-a": "rejected a", "agent-b": "rejected b"},
            chosen_reward=1.0,
            rejected_reward=0.0,
            candidate_mean=0.5,
        )


def test_joint_preference_pair_rejects_reward_ties():
    with pytest.raises(ValueError, match="strictly greater"):
        JointPreferencePair(
            preference_pair_id="pair-tied",
            episode_id="episode-1",
            tree_node_id="node-root",
            chosen_joint_action_id="action-chosen",
            rejected_joint_action_id="action-rejected",
            prompts_by_agent={"agent-a": "prompt"},
            chosen_by_agent={"agent-a": "chosen"},
            rejected_by_agent={"agent-a": "rejected"},
            chosen_reward=0.5,
            rejected_reward=0.5,
            candidate_mean=0.5,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_joint_preference_pair_rejects_non_finite_rewards(value):
    with pytest.raises(ValueError, match="finite"):
        JointPreferencePair(
            preference_pair_id="pair-non-finite",
            episode_id="episode-1",
            tree_node_id="node-root",
            chosen_joint_action_id="action-chosen",
            rejected_joint_action_id="action-rejected",
            prompts_by_agent={"agent-a": "prompt"},
            chosen_by_agent={"agent-a": "chosen"},
            rejected_by_agent={"agent-a": "rejected"},
            chosen_reward=value,
            rejected_reward=0.0,
            candidate_mean=0.5,
        )
