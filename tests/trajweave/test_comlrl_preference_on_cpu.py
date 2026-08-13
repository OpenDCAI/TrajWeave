from __future__ import annotations

import pytest

from trajweave.core.joint_trajectory import JointAction, JointCompletion, JointTransition, JointTreeNode
from trajweave.core.specs import AgentSpec, PolicyGroupSpec, TeamSpec
from trajweave.core.trajectory import MultiAgentTrajectory
from trajweave.credit.comlrl.preference import (
    build_joint_preference_pairs,
    preference_pairs_to_training_samples,
)


def _team() -> TeamSpec:
    return TeamSpec(
        name="preference-team",
        agents=(
            AgentSpec(name="alice", role="solver", policy_group="actor-a", trainable=True),
            AgentSpec(name="bob", role="solver", policy_group="actor-b", trainable=True),
        ),
        policy_groups=(
            PolicyGroupSpec(name="actor-a", backend="local", trainable=True),
            PolicyGroupSpec(name="actor-b", backend="local", trainable=True),
        ),
        orchestra="joint",
        reward="joint",
        credit="preference",
    )


def _trajectory(rewards=(0.1, 0.8, 0.4), *, mode="aligned") -> MultiAgentTrajectory:
    completions = []
    actions = []
    transitions = []
    for candidate_index, reward in enumerate(rewards):
        completion_ids = {}
        candidate_indices = {}
        for agent_index, agent_name in enumerate(("alice", "bob")):
            completion = JointCompletion(
                completion_id=f"completion-{agent_name}-{candidate_index}",
                tree_node_id="root",
                agent_name=agent_name,
                candidate_index=candidate_index,
                text=f"{agent_name}-answer-{candidate_index}",
                token_ids=[agent_index + 1, candidate_index + 10],
                logprobs=[-0.1, -0.2],
                metadata={"prompt": f"prompt-{agent_name}"},
            )
            completions.append(completion)
            completion_ids[agent_name] = completion.completion_id
            candidate_indices[agent_name] = candidate_index
        action_id = f"action-{candidate_index}"
        transition_id = f"transition-{candidate_index}"
        actions.append(
            JointAction(
                joint_action_id=action_id,
                tree_node_id="root",
                completion_ids=completion_ids,
                candidate_indices=candidate_indices,
                joint_transition_id=transition_id,
                shared_reward=reward,
                done=True,
            )
        )
        transitions.append(
            JointTransition(
                joint_transition_id=transition_id,
                joint_action_id=action_id,
                source_tree_node_id="root",
                source_turn=0,
                done=True,
            )
        )
    return MultiAgentTrajectory(
        episode_id="episode-pref",
        task_id="task-pref",
        rollout_group="group-pref",
        team_name="preference-team",
        joint_nodes=[JointTreeNode("root", "root", 0)],
        joint_completions=completions,
        joint_actions=actions,
        joint_transitions=transitions,
        metadata={"joint_sampling_mode": mode},
    )


def test_reward_gap_pairing_and_candidate_mean_match_v141():
    pairs = build_joint_preference_pairs(_trajectory(), pair_selection="reward_gap", pairs_per_sample=16)

    assert [(pair.metadata["winner_candidate_index"], pair.metadata["loser_candidate_index"]) for pair in pairs] == [
        (1, 0),
        (1, 2),
        (2, 0),
    ]
    assert [pair.metadata["reward_gap"] for pair in pairs] == pytest.approx([0.7, 0.4, 0.3])
    assert all(pair.candidate_mean == pytest.approx(1.3 / 3.0) for pair in pairs)
    assert pairs[0].chosen_joint_action_id == "action-1"
    assert pairs[0].rejected_joint_action_id == "action-0"
    assert pairs[0].chosen_by_agent == {"alice": "alice-answer-1", "bob": "bob-answer-1"}


def test_pair_selection_limit_random_reproducibility_ties_and_mode_constraints():
    trajectory = _trajectory()
    assert len(build_joint_preference_pairs(trajectory, pairs_per_sample=2)) == 2
    all_pairs = build_joint_preference_pairs(trajectory, pair_selection="all", pairs_per_sample=1)
    assert len(all_pairs) == 3
    random_a = build_joint_preference_pairs(trajectory, pair_selection="random", random_seed=7)
    random_b = build_joint_preference_pairs(trajectory, pair_selection="random", random_seed=7)
    assert [pair.preference_pair_id for pair in random_a] == [pair.preference_pair_id for pair in random_b]
    assert build_joint_preference_pairs(_trajectory((0.5, 0.5))) == []
    with pytest.raises(ValueError, match="aligned"):
        build_joint_preference_pairs(_trajectory(mode="cross"))


def test_preference_pairs_expand_to_complete_chosen_rejected_actor_rows():
    trajectory = _trajectory()
    [pair] = build_joint_preference_pairs(trajectory, pairs_per_sample=1)

    rows = preference_pairs_to_training_samples(trajectory, [pair], _team())

    assert len(rows) == 4
    assert [(row.agent_name, row.metadata["preference_side"]) for row in rows] == [
        ("alice", "chosen"),
        ("alice", "rejected"),
        ("bob", "chosen"),
        ("bob", "rejected"),
    ]
    assert {row.metadata["preference_pair_id"] for row in rows} == {pair.preference_pair_id}
    assert [row.metadata["candidate_mean"] for row in rows] == pytest.approx([1.3 / 3.0] * 4)
    assert all(row.metadata["preference_loss_mask"] == 1.0 for row in rows)
    assert all(len(row.joint_action_ids) == len(row.joint_transition_ids) == 1 for row in rows)
    assert rows[0].completion_id == "completion-alice-1"
    assert rows[1].completion_id == "completion-alice-0"
