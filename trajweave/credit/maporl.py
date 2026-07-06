from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import sqrt

from trajweave.core.specs import TeamSpec
from trajweave.core.trajectory import MultiAgentTrajectory, TrainingSample
from trajweave.credit.global_broadcast import GlobalBroadcastCreditAssigner


def shape_maporl_reward(
    global_reward: float,
    *,
    agent_answer: object | None,
    consensus_answer: object | None,
    consensus_reached: bool,
    correct_turn_bonus: float = 0.25,
    consensus_bonus: float = 0.25,
) -> tuple[float, dict[str, float | str]]:
    shaped_reward = float(global_reward)
    task_success = float(global_reward) > 0
    if task_success and _same_answer(agent_answer, consensus_answer):
        shaped_reward += correct_turn_bonus
    if task_success and consensus_reached:
        shaped_reward += consensus_bonus
    return shaped_reward, {
        "score_rule": "final_task_success",
        "bonus_rule": "correct_turn_and_consensus",
        "raw_global_reward": float(global_reward),
        "shaped_reward": shaped_reward,
    }


@dataclass
class MAPoRLScoreBonusCreditAssigner(GlobalBroadcastCreditAssigner):
    name: str = "maporl_score_bonus"
    correct_turn_bonus: float = 0.25
    consensus_bonus: float = 0.25
    epsilon: float = 1e-6
    normalize_by_agent: bool = True
    baseline_scope: str = "policy_group"

    def assign(self, trajectories: list[MultiAgentTrajectory], team: TeamSpec) -> list[TrainingSample]:
        samples = super().assign(trajectories, team)
        trajectory_by_id = {trajectory.episode_id: trajectory for trajectory in trajectories}
        for sample in samples:
            trajectory = trajectory_by_id[sample.episode_id]
            shaped_reward, reward_metadata = shape_maporl_reward(
                sample.reward,
                agent_answer=sample.metadata.get("agent_answer"),
                consensus_answer=trajectory.metadata.get("consensus_answer"),
                consensus_reached=bool(sample.metadata.get("consensus_reached")),
                correct_turn_bonus=self.correct_turn_bonus,
                consensus_bonus=self.consensus_bonus,
            )
            sample.reward = shaped_reward
            sample.metadata["credit"] = self.name
            sample.metadata.update(reward_metadata)

        if self.normalize_by_agent:
            self._normalize_advantage(samples)
        return samples

    def _normalize_advantage(self, samples: list[TrainingSample]) -> None:
        groups: dict[str, list[TrainingSample]] = defaultdict(list)
        for sample in samples:
            if self.baseline_scope == "agent":
                group_key = f"{sample.rollout_group}:{sample.agent_name}"
            elif self.baseline_scope == "policy_group":
                group_key = f"{sample.rollout_group}:{sample.policy_group}"
            else:
                raise ValueError(f"Unsupported MAPoRL baseline_scope: {self.baseline_scope}")
            groups[group_key].append(sample)

        for group_key, group_samples in groups.items():
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
                sample.advantage = (sample.reward - mean) / (std + self.epsilon)
                sample.metadata["advantage_group"] = group_key
                sample.metadata["reward_mean"] = mean
                sample.metadata["reward_std"] = std


def _same_answer(left: object | None, right: object | None) -> bool:
    if left is None or right is None:
        return False
    return str(left).strip() == str(right).strip()
