from __future__ import annotations

from pathlib import Path
from typing import Any

from trajweave.core.trajectory import MultiAgentTrajectory, TrainingSample
from trajweave.storage.jsonl import JsonlWriter
from trajweave.storage.serialization import json_safe


class TrajectoryStore:
    def __init__(self, run_dir: str | Path, run_id: str) -> None:
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.trajectory_dir = self.run_dir / "trajectories"
        self.online_turn_dir = self.trajectory_dir / "online_turns"
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        self.online_turn_dir.mkdir(parents=True, exist_ok=True)
        self._trajectories = JsonlWriter(self.trajectory_dir / "trajectories.jsonl")
        self._turns = JsonlWriter(self.trajectory_dir / "turns.jsonl")
        self._samples = JsonlWriter(self.trajectory_dir / "samples.jsonl")

    def write_trajectories(self, trajectories: list[MultiAgentTrajectory]) -> None:
        for trajectory in trajectories:
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

    def write_samples(self, samples: list[TrainingSample]) -> None:
        self._samples.write_many([{"run_id": self.run_id, **json_safe(sample)} for sample in samples])

    def online_turn_shard_path(self, worker_name: str) -> Path:
        safe_worker_name = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in worker_name)
        return self.online_turn_dir / f"{safe_worker_name}.jsonl"

    def write_online_turn(self, worker_name: str, row: dict[str, Any]) -> None:
        JsonlWriter(self.online_turn_shard_path(worker_name)).write({"run_id": self.run_id, **row})
