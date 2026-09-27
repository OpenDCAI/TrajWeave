from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import isfinite, sqrt

from trajweave.core.specs import TeamSpec
from trajweave.core.trajectory import MultiAgentTrajectory, TrainingSample
from trajweave.credit.global_broadcast import GlobalBroadcastCreditAssigner


def apply_wideseek_trajectory_reward(
    trajectory: MultiAgentTrajectory,
    *,
    format_reward: float = 0.1,
    search_reward: float = 0.05,
    length_limit: int = 3000,
    max_length_limit: int = 5000,
    length_penalty: float = 0.1,
) -> float:
    """Compute one WideSeek-R1 outcome and broadcast it to all agents."""

    format_reward = _nonnegative(format_reward, "format_reward")
    search_reward = _nonnegative(search_reward, "search_reward")
    length_penalty = _nonnegative(length_penalty, "length_penalty")
    if length_limit < 1 or max_length_limit <= length_limit:
        raise ValueError("WideSeek-R1 requires 0 < length_limit < max_length_limit.")

    # RLinf's WideSeek-R1 credit assignment gates the trajectory on the final
    # answer format.  Intermediate tool calls are intentionally best-effort:
    # a model can emit extra prose around a search/access call and still
    # produce a valid final answer.  Requiring every intermediate turn to be
    # perfectly formatted makes realistic HF rollouts receive zero reward and
    # therefore zero policy gradient, even though the workflow reached a
    # verifiable answer.
    final_turns = [turn for turn in trajectory.turns if turn.metadata.get("wideseek_turn_role") == "lead_final"]
    final_turn = final_turns[-1] if final_turns else (trajectory.turns[-1] if trajectory.turns else None)
    format_valid = bool(final_turn is not None and final_turn.metadata.get("wideseek_format_valid", False))
    all_formats_valid = bool(trajectory.turns) and all(
        bool(turn.metadata.get("wideseek_format_valid", False)) for turn in trajectory.turns
    )
    used_access = any(turn.metadata.get("tool_name") == "access" for turn in trajectory.turns)
    outcome = float(trajectory.global_reward or 0.0)
    max_tokens = max((len(turn.action_token_ids) for turn in trajectory.turns), default=0)
    overflow = max(0.0, min(1.0, (max_tokens - length_limit) / (max_length_limit - length_limit)))
    reward = 0.0
    if format_valid:
        reward = outcome + format_reward + (search_reward if used_access else 0.0) - overflow * length_penalty
        reward = max(0.0, reward)

    for turn in trajectory.turns:
        turn.reward = reward
        turn.metadata.update(
            {
                "wideseek_outcome_reward": outcome,
                "wideseek_trajectory_reward": reward,
                "wideseek_final_format_valid": format_valid,
                "wideseek_all_formats_valid": all_formats_valid,
                "wideseek_used_access": used_access,
                "wideseek_length_penalty": overflow * length_penalty,
            }
        )
    trajectory.metadata.update(
        {
            "wideseek_outcome_reward": outcome,
            "wideseek_trajectory_reward": reward,
            "wideseek_final_format_valid": format_valid,
            "wideseek_all_formats_valid": all_formats_valid,
            "wideseek_used_access": used_access,
        }
    )
    return reward


@dataclass
class WideSeekR1CreditAssigner(GlobalBroadcastCreditAssigner):
    """Trajectory-level GRPO advantage shared by lead and all subagents."""

    name: str = "wideseek_r1_multi_agent_grpo"
    format_reward: float = 0.1
    search_reward: float = 0.05
    length_limit: int = 3000
    max_length_limit: int = 5000
    length_penalty: float = 0.1
    epsilon: float = 1e-6
    normalize_by_std: bool = True

    def assign(self, trajectories: list[MultiAgentTrajectory], team: TeamSpec) -> list[TrainingSample]:
        grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
        rewards: dict[str, float] = {}
        for trajectory in trajectories:
            reward = apply_wideseek_trajectory_reward(
                trajectory,
                format_reward=self.format_reward,
                search_reward=self.search_reward,
                length_limit=self.length_limit,
                max_length_limit=self.max_length_limit,
                length_penalty=self.length_penalty,
            )
            rewards[trajectory.episode_id] = reward
            grouped[trajectory.rollout_group].append((trajectory.episode_id, reward))

        stats: dict[str, tuple[float, float, float]] = {}
        for rows in grouped.values():
            values = [reward for _, reward in rows]
            mean = sum(values) / len(values)
            std = sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1)) if len(values) > 1 else 1.0
            if std < self.epsilon:
                std = 1.0
            for episode_id, reward in rows:
                advantage = reward - mean
                if self.normalize_by_std:
                    advantage /= std + self.epsilon
                stats[episode_id] = (advantage, mean, std)

        samples = super().assign(trajectories, team)
        for sample in samples:
            advantage, mean, std = stats[sample.episode_id]
            sample.reward = rewards[sample.episode_id]
            sample.advantage = advantage
            sample.metadata.update(
                {
                    "credit": self.name,
                    "advantage_group": sample.rollout_group,
                    "reward_mean": mean,
                    "reward_std": std,
                    "wideseek_shared_trajectory_advantage": True,
                    "wideseek_dual_level_reweighting": True,
                }
            )
        return samples


def _nonnegative(value: float, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"WideSeek-R1 {field} must be a finite non-negative number.") from exc
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"WideSeek-R1 {field} must be a finite non-negative number.")
    return result
