from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import sqrt

from trajweave.core.specs import TeamSpec
from trajweave.core.trajectory import MultiAgentTrajectory, TrainingSample
from trajweave.credit.global_broadcast import GlobalBroadcastCreditAssigner


def apply_mixed_reward(
    trajectory: MultiAgentTrajectory,
    team: TeamSpec,
    *,
    alpha: float = 1.0,
    verifier_local_reward_scale: float = 1.0,
    solver_name: str = "solver",
    verifier_name: str = "verifier",
) -> None:
    """Apply ``alpha * global_reward + role_local_reward`` to every AT-GRPO sibling."""

    trainable = {agent.name for agent in team.trainable_agents()}
    final_correct = (
        bool(trajectory.success) if trajectory.success is not None else float(trajectory.global_reward or 0.0) > 0
    )
    global_reward = float(trajectory.global_reward or 0.0)
    for turn in trajectory.trainable_turns(trainable):
        if turn.agent_name not in {solver_name, verifier_name}:
            continue

        if turn.local_score is not None:
            local_score = float(turn.local_score)
        elif turn.agent_name == solver_name:
            local_score = float(bool(turn.metadata.get("local_correct", False)))
        else:
            model_approved = bool(turn.metadata.get("model_approved", turn.metadata.get("approved", False)))
            local_solver_correct = bool(turn.metadata.get("local_solver_correct", final_correct))
            local_score = 1.0 if model_approved == local_solver_correct else -1.0

        role_local_reward = (
            local_score * verifier_local_reward_scale if turn.agent_name == verifier_name else local_score
        )
        turn.reward = alpha * global_reward + role_local_reward
        turn.metadata["mixed_reward"] = {
            "alpha": alpha,
            "global_reward": global_reward,
            "local_score": local_score,
            "role_local_reward": role_local_reward,
        }


@dataclass
class ATGRPOCreditAssigner(GlobalBroadcastCreditAssigner):
    """Agent- and Turn-wise GRPO credit assignment.

    Mirrors PettingLLMs' AT-GRPO: rather than normalizing rewards only within
    an agent's role across rollouts (as :class:`DoctorMASCreditAssigner`
    does), samples are grouped jointly by ``(rollout_group, turn_id,
    agent_name)`` so the baseline is computed within trajectories that share
    both the same turn index and the same agent role.
    """

    name: str = "atgrpo_agent_turn_wise_grpo"
    epsilon: float = 1e-6
    normalize_by_std: bool = True
    mixed_reward_enabled: bool = False
    alpha: float = 1.0
    verifier_local_reward_scale: float = 1.0
    verifier_name: str = "verifier"

    def assign(self, trajectories: list[MultiAgentTrajectory], team: TeamSpec) -> list[TrainingSample]:
        if self.mixed_reward_enabled:
            for trajectory in trajectories:
                apply_mixed_reward(
                    trajectory,
                    team,
                    alpha=self.alpha,
                    verifier_local_reward_scale=self.verifier_local_reward_scale,
                    verifier_name=self.verifier_name,
                )
        samples = super().assign(trajectories, team)
        groups: dict[str, list[TrainingSample]] = defaultdict(list)
        for sample in samples:
            group_id = sample.observation_group_id or sample.metadata.get("observation_group_id")
            if group_id is None:
                group_id = f"legacy:{sample.rollout_group}:{sample.turn_id}:{sample.agent_name}"
            groups[str(group_id)].append(sample)

        max_group_size = max((len(group_samples) for group_samples in groups.values()), default=0)
        for group_id, group_samples in groups.items():
            rewards = [sample.reward for sample in group_samples]
            if len(rewards) > 1:
                mean = sum(rewards) / len(rewards)
                variance = sum((reward - mean) ** 2 for reward in rewards) / (len(rewards) - 1)
                std = sqrt(variance)
            elif max_group_size == 1:
                mean = 0.0
                std = 1.0
            else:
                mean = rewards[0]
                std = 0.0
            for sample in group_samples:
                advantage = sample.reward - mean
                if self.normalize_by_std:
                    advantage = advantage / (std + self.epsilon)
                sample.advantage = advantage
                sample.metadata["advantage_group"] = group_id
                sample.metadata["reward_mean"] = mean
                sample.metadata["reward_std"] = std
        return samples
