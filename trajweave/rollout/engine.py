from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from trajweave.backends.policy import PolicyBackend
from trajweave.core.specs import TeamSpec
from trajweave.core.trajectory import MultiAgentTrajectory, TrainingSample
from trajweave.credit.base import CreditAssigner
from trajweave.envs.base import Environment, evaluate_trajectory
from trajweave.orchestration.base import Orchestra


@dataclass
class RolloutResult:
    trajectories: list[MultiAgentTrajectory]
    samples: list[TrainingSample]

    @property
    def success_rate(self) -> float:
        if not self.trajectories:
            return 0.0
        return sum(1 for item in self.trajectories if item.success) / len(self.trajectories)


@dataclass
class RolloutEngine:
    team: TeamSpec
    orchestra: Orchestra
    environment: Environment
    policy_backend: PolicyBackend
    credit_assigner: CreditAssigner

    def run(self, tasks: list[Any], rollouts_per_task: int = 1) -> RolloutResult:
        trajectories: list[MultiAgentTrajectory] = []
        for task in tasks:
            observation = self.environment.initial_observation(task)
            run_tree = getattr(self.orchestra, "run_tree", None)
            if run_tree is not None:
                episode_id = str(uuid4())
                trajectory = run_tree(
                    episode_id=episode_id,
                    rollout_group=task.task_id,
                    task=task,
                    team=self.team,
                    observation=observation,
                    policy_backend=self.policy_backend,
                    environment=self.environment,
                    branch_factor=rollouts_per_task,
                )
                reward, success = evaluate_trajectory(self.environment, task, trajectory)
                trajectory.global_reward = reward
                trajectory.success = success
                trajectory.metadata["rollout_idx"] = 0
                trajectories.append(trajectory)
                continue
            for rollout_idx in range(rollouts_per_task):
                episode_id = str(uuid4())
                trajectory = self.orchestra.run(
                    episode_id=episode_id,
                    rollout_group=task.task_id,
                    task=task,
                    team=self.team,
                    observation=observation,
                    policy_backend=self.policy_backend,
                    environment=self.environment,
                )
                reward, success = evaluate_trajectory(self.environment, task, trajectory)
                trajectory.global_reward = reward
                trajectory.success = success
                trajectory.metadata["rollout_idx"] = rollout_idx
                trajectories.append(trajectory)
        samples = self.credit_assigner.assign(trajectories, self.team)
        return RolloutResult(trajectories=trajectories, samples=samples)
