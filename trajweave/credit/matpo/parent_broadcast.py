from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import sqrt

from trajweave.core.specs import TeamSpec
from trajweave.core.trajectory import MultiAgentTrajectory, TrainingSample
from trajweave.credit.global_broadcast import GlobalBroadcastCreditAssigner


@dataclass
class MATPOParentBroadcastCreditAssigner(GlobalBroadcastCreditAssigner):
    name: str = "matpo_parent_broadcast_grpo"
    epsilon: float = 1e-6
    normalize_by_std: bool = True

    def assign(self, trajectories: list[MultiAgentTrajectory], team: TeamSpec) -> list[TrainingSample]:
        samples = super().assign(trajectories, team)
        main_samples = [sample for sample in samples if not bool(sample.metadata.get("is_from_subagent_tool", False))]
        groups: dict[str, list[TrainingSample]] = defaultdict(list)
        for sample in main_samples:
            groups[sample.rollout_group].append(sample)

        parent_advantages: dict[str, float] = {}
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
                sample.metadata["advantage_group"] = sample.rollout_group
                sample.metadata["reward_mean"] = mean
                sample.metadata["reward_std"] = std
                reqs_id = str(sample.metadata.get("reqs_id", sample.sample_id))
                parent_advantages[reqs_id] = advantage

        for sample in samples:
            if not bool(sample.metadata.get("is_from_subagent_tool", False)):
                continue
            parent_id = str(sample.metadata.get("parent_reqs_id", ""))
            sample.advantage = parent_advantages.get(parent_id, 0.0)
            sample.metadata["advantage_group"] = f"parent:{parent_id}"
            sample.metadata["parent_advantage_broadcast"] = True
        return samples
