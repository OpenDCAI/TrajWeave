from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from trajweave.core.preference import JointPreferencePair
from trajweave.core.trajectory import MultiAgentTrajectory, TrainingSample
from trajweave.storage.jsonl import JsonlWriter
from trajweave.storage.serialization import json_safe


class TrajectoryStore:
    def __init__(self, run_dir: str | Path, run_id: str) -> None:
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.trajectory_dir = self.run_dir / "trajectories"
        self.online_turn_dir = self.trajectory_dir / "online_turns"
        self.preference_dir = self.run_dir / "preferences"
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        self.online_turn_dir.mkdir(parents=True, exist_ok=True)
        self.preference_dir.mkdir(parents=True, exist_ok=True)
        self._trajectories = JsonlWriter(self.trajectory_dir / "trajectories.jsonl")
        self._turns = JsonlWriter(self.trajectory_dir / "turns.jsonl")
        self._samples = JsonlWriter(self.trajectory_dir / "samples.jsonl")
        self._joint_nodes = JsonlWriter(self.trajectory_dir / "joint_nodes.jsonl")
        self._joint_completions = JsonlWriter(self.trajectory_dir / "joint_completions.jsonl")
        self._joint_actions = JsonlWriter(self.trajectory_dir / "joint_actions.jsonl")
        self._joint_transitions = JsonlWriter(self.trajectory_dir / "joint_transitions.jsonl")
        self._preference_pairs = JsonlWriter(self.preference_dir / "pairs.jsonl")
        self._joint_action_nodes = _existing_action_nodes(self._joint_actions.path)
        self._joint_action_keys = set(self._joint_action_nodes)
        self._preference_pair_ids = {
            key[0] for key in _existing_keys(self._preference_pairs.path, "preference_pair_id")
        }

    def write_trajectories(self, trajectories: list[MultiAgentTrajectory]) -> None:
        for trajectory in trajectories:
            action_keys = {(trajectory.episode_id, action.joint_action_id) for action in trajectory.joint_actions}
            if len(action_keys) != len(trajectory.joint_actions):
                raise ValueError("joint_action_id must be unique within an episode")
            if action_keys & self._joint_action_keys:
                raise ValueError("joint_action_id already exists for this episode")
            self._trajectories.write(
                {
                    "run_id": self.run_id,
                    "episode_id": trajectory.episode_id,
                    "task_id": trajectory.task_id,
                    "rollout_group": trajectory.rollout_group,
                    "team_name": trajectory.team_name,
                    "final_answer": trajectory.final_answer,
                    "global_reward": trajectory.global_reward,
                    "success": trajectory.success,
                    "metadata": trajectory.metadata,
                }
            )
            for turn in trajectory.turns:
                self._turns.write({"run_id": self.run_id, **json_safe(turn)})
            trajectory_context = {
                "run_id": self.run_id,
                "episode_id": trajectory.episode_id,
                "task_id": trajectory.task_id,
                "rollout_group": trajectory.rollout_group,
            }
            self._joint_nodes.write_many([{**trajectory_context, **json_safe(node)} for node in trajectory.joint_nodes])
            self._joint_completions.write_many(
                [{**trajectory_context, **json_safe(completion)} for completion in trajectory.joint_completions]
            )
            self._joint_actions.write_many(
                [{**trajectory_context, **json_safe(action)} for action in trajectory.joint_actions]
            )
            self._joint_transitions.write_many(
                [{**trajectory_context, **json_safe(transition)} for transition in trajectory.joint_transitions]
            )
            self._joint_action_keys.update(action_keys)
            self._joint_action_nodes.update(
                {
                    (trajectory.episode_id, action.joint_action_id): action.tree_node_id
                    for action in trajectory.joint_actions
                }
            )

    def write_samples(self, samples: list[TrainingSample]) -> None:
        self._samples.write_many([{"run_id": self.run_id, **json_safe(sample)} for sample in samples])

    def write_preference_pairs(self, pairs: list[JointPreferencePair]) -> None:
        pair_ids = [pair.preference_pair_id for pair in pairs]
        if len(pair_ids) != len(set(pair_ids)) or set(pair_ids) & self._preference_pair_ids:
            raise ValueError("preference_pair_id must be unique within a run")
        referenced_actions = {
            (pair.episode_id, action_id)
            for pair in pairs
            for action_id in (pair.chosen_joint_action_id, pair.rejected_joint_action_id)
        }
        missing_actions = referenced_actions - self._joint_action_keys
        if missing_actions:
            raise ValueError(f"preference pairs reference unknown joint actions: {sorted(missing_actions)}")
        mismatched_nodes = [
            (pair.preference_pair_id, action_id)
            for pair in pairs
            for action_id in (pair.chosen_joint_action_id, pair.rejected_joint_action_id)
            if self._joint_action_nodes[(pair.episode_id, action_id)] != pair.tree_node_id
        ]
        if mismatched_nodes:
            raise ValueError(f"preference actions must belong to tree_node_id: {mismatched_nodes}")
        self._preference_pairs.write_many([{"run_id": self.run_id, **json_safe(pair)} for pair in pairs])
        self._preference_pair_ids.update(pair_ids)

    def online_turn_shard_path(self, worker_name: str) -> Path:
        safe_worker_name = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in worker_name)
        return self.online_turn_dir / f"{safe_worker_name}.jsonl"

    def write_online_turn(self, worker_name: str, row: dict[str, Any]) -> None:
        JsonlWriter(self.online_turn_shard_path(worker_name)).write({"run_id": self.run_id, **row})


def _existing_keys(path: Path, *fields: str) -> set[tuple[str, ...]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, ...]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        keys.add(tuple(str(row[field]) for field in fields))
    return keys


def _existing_action_nodes(path: Path) -> dict[tuple[str, str], str]:
    if not path.exists():
        return {}
    action_nodes: dict[tuple[str, str], str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        action_nodes[(str(row["episode_id"]), str(row["joint_action_id"]))] = str(row["tree_node_id"])
    return action_nodes
