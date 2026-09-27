from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import isfinite, sqrt

from trajweave.core.specs import TeamSpec
from trajweave.core.trajectory import MultiAgentTrajectory, TrainingSample
from trajweave.credit.global_broadcast import GlobalBroadcastCreditAssigner

DEFAULT_EXPLORER_FORMAT_BONUS = 0.1
DEFAULT_ADAPTER_FORMAT_BONUS = 0.1


def apply_mrlx_trajectory_rewards(
    trajectory: MultiAgentTrajectory,
    *,
    explorer_agent: str,
    adapter_agent: str,
    explorer_format_bonus: float = DEFAULT_EXPLORER_FORMAT_BONUS,
    adapter_format_bonus: float = DEFAULT_ADAPTER_FORMAT_BONUS,
) -> dict[str, float]:
    """Apply MrlX's end-to-end explorer and inherited adapter rewards."""

    explorer_bonus = _unit_score(explorer_format_bonus, field="explorer_format_bonus")
    adapter_bonus = _unit_score(adapter_format_bonus, field="adapter_format_bonus")
    outcome_reward = _unit_score(float(trajectory.global_reward or 0.0), field="outcome_reward")
    explorer_turns = [turn for turn in trajectory.turns if turn.agent_name == explorer_agent]
    adapter_turns = [turn for turn in trajectory.turns if turn.agent_name == adapter_agent]
    if not explorer_turns:
        raise ValueError("MrlX trajectories must contain explorer turns.")

    explorer_format = all(bool(turn.metadata.get("mrlx_format_valid", False)) for turn in explorer_turns)
    adapter_format = bool(adapter_turns) and all(
        bool(turn.metadata.get("mrlx_format_valid", False)) for turn in adapter_turns
    )
    explorer_succeeded = explorer_format and outcome_reward > 0.0
    explorer_reward = outcome_reward if explorer_succeeded else explorer_bonus if explorer_format else 0.0
    if not adapter_format:
        adapter_reward = 0.0
    elif explorer_succeeded:
        adapter_reward = outcome_reward
    else:
        adapter_reward = adapter_bonus

    rewards = {explorer_agent: explorer_reward}
    if adapter_turns:
        rewards[adapter_agent] = adapter_reward
    for turn in trajectory.turns:
        if turn.agent_name in rewards:
            turn.reward = rewards[turn.agent_name]
            turn.metadata.update(
                {
                    "mrlx_outcome_reward": outcome_reward,
                    "mrlx_role_reward": rewards[turn.agent_name],
                    "mrlx_explorer_format_valid": explorer_format,
                    "mrlx_adapter_format_valid": adapter_format,
                    "mrlx_explorer_format_bonus": explorer_bonus,
                    "mrlx_adapter_format_bonus": adapter_bonus,
                }
            )
    trajectory.metadata.update(
        {
            "mrlx_outcome_reward": outcome_reward,
            "mrlx_explorer_reward": explorer_reward,
            "mrlx_adapter_reward": adapter_reward,
            "mrlx_explorer_format_valid": explorer_format,
            "mrlx_adapter_format_valid": adapter_format,
            "mrlx_adapter_present": bool(adapter_turns),
        }
    )
    return rewards


@dataclass
class MrlXMGRPOCreditAssigner(GlobalBroadcastCreditAssigner):
    name: str = "mrlx_mgrpo"
    explorer_agent: str = "main_explorer"
    adapter_agent: str = "sub_adapter"
    explorer_format_bonus: float = DEFAULT_EXPLORER_FORMAT_BONUS
    adapter_format_bonus: float = DEFAULT_ADAPTER_FORMAT_BONUS
    epsilon: float = 1e-6
    normalize_by_std: bool = True

    def assign(self, trajectories: list[MultiAgentTrajectory], team: TeamSpec) -> list[TrainingSample]:
        role_rewards: dict[tuple[str, str], float] = {}
        grouped: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)
        for trajectory in trajectories:
            rewards = apply_mrlx_trajectory_rewards(
                trajectory,
                explorer_agent=self.explorer_agent,
                adapter_agent=self.adapter_agent,
                explorer_format_bonus=self.explorer_format_bonus,
                adapter_format_bonus=self.adapter_format_bonus,
            )
            for agent_name, reward in rewards.items():
                role_rewards[(trajectory.episode_id, agent_name)] = reward
                grouped[(trajectory.rollout_group, agent_name)].append((trajectory.episode_id, reward))

        samples = super().assign(trajectories, team)
        advantages: dict[tuple[str, str], tuple[float, float, float]] = {}
        for (_rollout_group, agent_name), group_rows in grouped.items():
            rewards = [reward for _, reward in group_rows]
            mean = sum(rewards) / len(rewards)
            if len(rewards) > 1:
                variance = sum((reward - mean) ** 2 for reward in rewards) / (len(rewards) - 1)
                std = sqrt(variance)
            else:
                std = 1.0
            if std < self.epsilon:
                std = 1.0
            for (episode_id, reward) in group_rows:
                advantage = reward - mean
                if self.normalize_by_std:
                    advantage /= std + self.epsilon
                advantages[(episode_id, agent_name)] = (advantage, mean, std)

        for sample in samples:
            key = (sample.episode_id, sample.agent_name)
            sample.reward = role_rewards[key]
            advantage, mean, std = advantages[key]
            sample.advantage = advantage
            sample.metadata.update(
                {
                    "credit": self.name,
                    "advantage_group": f"{sample.rollout_group}:{sample.agent_name}",
                    "reward_mean": mean,
                    "reward_std": std,
                    "mrlx_role_wise_grpo": True,
                }
            )
        return samples


def _unit_score(value: float, *, field: str) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"MrlX {field} must be a finite number between 0 and 1.") from exc
    if not isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"MrlX {field} must be a finite number between 0 and 1.")
    return score
