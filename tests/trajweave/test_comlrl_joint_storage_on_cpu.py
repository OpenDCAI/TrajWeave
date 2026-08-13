import json
from dataclasses import replace
from pathlib import Path

import pytest

from trajweave.core.joint_trajectory import JointAction, JointCompletion, JointTransition, JointTreeNode
from trajweave.core.preference import JointPreferencePair
from trajweave.core.trajectory import AgentTurn, MultiAgentTrajectory, TrainingSample
from trajweave.storage import TrajectoryStore


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def without_storage_context(row: dict) -> dict:
    context_fields = {"run_id", "episode_id", "task_id", "rollout_group"}
    return {key: value for key, value in row.items() if key not in context_fields}


def without_run_id(row: dict) -> dict:
    return {key: value for key, value in row.items() if key != "run_id"}


def test_joint_records_and_preferences_round_trip_through_jsonl(tmp_path):
    node = JointTreeNode("node-root", "node-root", 0, metadata={"artifact": tmp_path / "artifact.txt"})
    completions = [
        JointCompletion("completion-a", "node-root", "agent-a", 0, "answer a", [1, 2], [-0.1, -0.2]),
        JointCompletion("completion-b", "node-root", "agent-b", 0, "answer b", [3], [-0.3]),
        JointCompletion("completion-other-a", "node-root", "agent-a", 1, "other a", [4], [-0.4]),
        JointCompletion("completion-other-b", "node-root", "agent-b", 1, "other b", [5], [-0.5]),
    ]
    action = JointAction(
        joint_action_id="action-terminal",
        tree_node_id="node-root",
        completion_ids={"agent-a": "completion-a", "agent-b": "completion-b"},
        candidate_indices={"agent-a": 0, "agent-b": 0},
        shared_reward=1.0,
        shared_return=1.0,
        done=True,
        joint_transition_id="transition-terminal",
    )
    transition = JointTransition(
        joint_transition_id="transition-terminal",
        joint_action_id="action-terminal",
        source_tree_node_id="node-root",
        source_turn=0,
        done=True,
    )
    rejected_action = JointAction(
        joint_action_id="action-rejected",
        tree_node_id="node-root",
        completion_ids={"agent-a": "completion-other-a", "agent-b": "completion-other-b"},
        candidate_indices={"agent-a": 1, "agent-b": 1},
        shared_reward=0.0,
        shared_return=0.0,
        done=True,
        joint_transition_id="transition-rejected",
    )
    rejected_transition = JointTransition(
        joint_transition_id="transition-rejected",
        joint_action_id="action-rejected",
        source_tree_node_id="node-root",
        source_turn=0,
        done=True,
    )
    turn = AgentTurn(
        "episode-1",
        "task-1",
        0,
        "agent-a",
        "solver",
        "shared",
        "observation",
        "prompt",
        "answer a",
        completion_id="completion-a",
        tree_node_id="node-root",
        joint_action_ids=["action-terminal"],
        joint_transition_ids=["transition-terminal"],
    )
    trajectory = MultiAgentTrajectory(
        episode_id="episode-1",
        task_id="task-1",
        rollout_group="group-1",
        team_name="joint-team",
        turns=[turn],
        joint_nodes=[node],
        joint_completions=completions,
        joint_actions=[action, rejected_action],
        joint_transitions=[transition, rejected_transition],
    )
    second_trajectory = MultiAgentTrajectory(
        episode_id="episode-2",
        task_id="task-2",
        rollout_group="group-2",
        team_name="joint-team",
        joint_nodes=[JointTreeNode("node-root", "node-root", 0)],
    )
    sample = TrainingSample(
        sample_id="sample-1",
        episode_id="episode-1",
        task_id="task-1",
        rollout_group="group-1",
        turn_id=0,
        agent_name="agent-a",
        role="solver",
        policy_group="shared",
        prompt="prompt",
        response="answer a",
        response_token_ids=[1, 2],
        response_logprobs=[-0.1, -0.2],
        reward=1.0,
        completion_id="completion-a",
        tree_node_id="node-root",
        joint_action_ids=["action-terminal"],
        joint_transition_ids=["transition-terminal"],
    )
    pair = JointPreferencePair(
        preference_pair_id="pair-1",
        episode_id="episode-1",
        tree_node_id="node-root",
        chosen_joint_action_id="action-terminal",
        rejected_joint_action_id="action-rejected",
        prompts_by_agent={"agent-a": "prompt a", "agent-b": "prompt b"},
        chosen_by_agent={"agent-a": "answer a", "agent-b": "answer b"},
        rejected_by_agent={"agent-a": "other a", "agent-b": "other b"},
        chosen_reward=1.0,
        rejected_reward=0.0,
        candidate_mean=0.5,
        metadata={"candidate_rewards": (1.0, 0.0)},
    )

    store = TrajectoryStore(tmp_path, "run-joint")
    store.write_trajectories([trajectory, second_trajectory])
    store.write_samples([sample])
    store.write_preference_pairs([pair])
    with pytest.raises(ValueError, match="unique within a run"):
        store.write_preference_pairs([pair])
    with pytest.raises(ValueError, match="unknown joint actions"):
        store.write_preference_pairs(
            [
                replace(
                    pair,
                    preference_pair_id="pair-dangling",
                    rejected_joint_action_id="missing-action",
                )
            ]
        )
    with pytest.raises(ValueError, match="belong to tree_node_id"):
        store.write_preference_pairs([replace(pair, preference_pair_id="pair-wrong-node", tree_node_id="other-node")])

    trajectory_dir = tmp_path / "trajectories"
    node_rows = read_jsonl(trajectory_dir / "joint_nodes.jsonl")
    [node_row] = [row for row in node_rows if row["episode_id"] == "episode-1"]
    completion_rows = read_jsonl(trajectory_dir / "joint_completions.jsonl")
    action_row = read_jsonl(trajectory_dir / "joint_actions.jsonl")[0]
    transition_row = read_jsonl(trajectory_dir / "joint_transitions.jsonl")[0]
    preference_row = read_jsonl(tmp_path / "preferences" / "pairs.jsonl")[0]

    assert {row["run_id"] for row in [*node_rows, *completion_rows, action_row, transition_row, preference_row]} == {
        "run-joint"
    }
    joint_rows = [node_row, *completion_rows, action_row, transition_row]
    assert {(row["episode_id"], row["task_id"], row["rollout_group"]) for row in joint_rows} == {
        ("episode-1", "task-1", "group-1")
    }
    assert {(row["tree_node_id"], row["episode_id"]) for row in node_rows} == {
        ("node-root", "episode-1"),
        ("node-root", "episode-2"),
    }
    assert node_row["metadata"]["artifact"] == str(tmp_path / "artifact.txt")
    assert JointTreeNode(**without_storage_context(node_row)).tree_node_id == node.tree_node_id
    assert [JointCompletion(**without_storage_context(row)) for row in completion_rows] == completions
    assert JointAction(**without_storage_context(action_row)) == action
    assert JointTransition(**without_storage_context(transition_row)) == transition
    assert JointPreferencePair(**without_run_id(preference_row)).chosen_by_agent == pair.chosen_by_agent
    assert preference_row["metadata"]["candidate_rewards"] == [1.0, 0.0]

    assert read_jsonl(trajectory_dir / "trajectories.jsonl")[0]["episode_id"] == "episode-1"
    assert read_jsonl(trajectory_dir / "turns.jsonl")[0]["completion_id"] == "completion-a"
    assert read_jsonl(trajectory_dir / "samples.jsonl")[0]["joint_action_ids"] == ["action-terminal"]
