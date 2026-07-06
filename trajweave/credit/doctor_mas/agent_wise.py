from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import sqrt

from trajweave.core.specs import TeamSpec
from trajweave.core.trajectory import MultiAgentTrajectory, TrainingSample
from trajweave.credit.global_broadcast import GlobalBroadcastCreditAssigner


@dataclass
class DoctorMASCreditAssigner(GlobalBroadcastCreditAssigner):
    name: str = "doctor_mas_agent_wise_grpo"
    epsilon: float = 1e-6
    normalize_by_std: bool = True

    def assign(self, trajectories: list[MultiAgentTrajectory], team: TeamSpec) -> list[TrainingSample]:
        samples = super().assign(trajectories, team)
        groups: dict[str, list[TrainingSample]] = defaultdict(list)
        for sample in samples:
            groups[f"{sample.rollout_group}:{sample.agent_name}"].append(sample)

        for group_samples in groups.values():
            rewards = [sample.reward for sample in group_samples]
            mean = sum(rewards) / len(rewards)
            if len(rewards) > 1:
                variance = sum((reward - mean) ** 2 for reward in rewards) / (len(rewards) - 1)
                std = sqrt(variance)
            else:
                std = 1.0
            if std < self.epsilon:
                std = 1.0
            for sample in group_samples:
                advantage = sample.reward - mean
                if self.normalize_by_std:
                    advantage = advantage / (std + self.epsilon)
                sample.advantage = advantage
                sample.metadata["advantage_group"] = f"{sample.rollout_group}:{sample.agent_name}"
                sample.metadata["reward_mean"] = mean
                sample.metadata["reward_std"] = std
        return samples
