from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch

from trajweave.core.specs import TeamSpec
from trajweave.core.trajectory import MultiAgentTrajectory, TrainingSample


@dataclass(frozen=True)
class MARSHALAdvantageResult:
    advantages: torch.Tensor
    agent_normalized_advantages: torch.Tensor
    turn_returns: torch.Tensor
    normalized_rewards: torch.Tensor
    raw_rewards: torch.Tensor


def compute_marshal_scalar_advantages(
    *,
    turn_rewards: torch.Tensor,
    episode_ids: list[Any],
    player_ids: list[Any],
    player_turn_ids: list[int],
    gamma: float = 1.0,
    reward_normalization: str = "mean",
    advantage_normalization: str = "mean",
    whiten_rewards: bool = True,
    whiten_advantages: bool = True,
    active_mask: list[bool] | None = None,
    epsilon: float = 1e-6,
) -> MARSHALAdvantageResult:
    """Compute MARSHAL turn returns and normalize advantages per logical player."""

    if turn_rewards.ndim != 1:
        raise ValueError("turn_rewards must be a 1D tensor.")
    row_count = turn_rewards.shape[0]
    fields = (episode_ids, player_ids, player_turn_ids)
    if any(len(field) != row_count for field in fields):
        raise ValueError("All MARSHAL row fields must have the same length.")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("MARSHAL gamma must be in [0, 1].")
    valid_modes = {"identity", "mean", "mean_std"}
    if reward_normalization not in valid_modes:
        raise ValueError(f"MARSHAL reward_normalization must be one of {sorted(valid_modes)}.")
    if advantage_normalization not in {"mean", "mean_std"}:
        raise ValueError("MARSHAL advantage_normalization must be mean or mean_std.")
    active = [True] * row_count if active_mask is None else [bool(value) for value in active_mask]
    if len(active) != row_count:
        raise ValueError("MARSHAL active_mask must align with turn_rewards.")

    raw_rewards = turn_rewards.float().clone()
    normalized_rewards = _normalize_by_player(
        raw_rewards,
        player_ids=player_ids,
        active_mask=active,
        mode=reward_normalization,
        epsilon=epsilon,
    )
    if whiten_rewards:
        normalized_rewards = _whiten_active(normalized_rewards, active_mask=active, epsilon=epsilon)

    turn_returns = torch.zeros_like(normalized_rewards)
    rows_by_subtrajectory: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row, (episode_id, player_id, is_active) in enumerate(zip(episode_ids, player_ids, active, strict=True)):
        if is_active:
            rows_by_subtrajectory[(str(episode_id), str(player_id))].append(row)
    with torch.no_grad():
        for rows in rows_by_subtrajectory.values():
            ordered = sorted(rows, key=lambda row: int(player_turn_ids[row]))
            turn_numbers = [int(player_turn_ids[row]) for row in ordered]
            if len(turn_numbers) != len(set(turn_numbers)):
                raise ValueError("MARSHAL player_turn_ids must be unique within each player trajectory.")
            cumulative = normalized_rewards.new_tensor(0.0)
            for row in reversed(ordered):
                cumulative = normalized_rewards[row] + gamma * cumulative
                turn_returns[row] = cumulative

    agent_normalized = _normalize_unique_returns_by_player(
        turn_returns,
        player_ids=player_ids,
        active_mask=active,
        mode=advantage_normalization,
        epsilon=epsilon,
    )
    advantages = (
        _whiten_active(agent_normalized, active_mask=active, epsilon=epsilon)
        if whiten_advantages
        else agent_normalized.clone()
    )
    inactive = torch.tensor([not value for value in active], dtype=torch.bool, device=turn_rewards.device)
    normalized_rewards[inactive] = 0.0
    turn_returns[inactive] = 0.0
    agent_normalized[inactive] = 0.0
    advantages[inactive] = 0.0
    return MARSHALAdvantageResult(
        advantages=advantages,
        agent_normalized_advantages=agent_normalized,
        turn_returns=turn_returns,
        normalized_rewards=normalized_rewards,
        raw_rewards=raw_rewards,
    )


@dataclass
class MARSHALCreditAssigner:
    name: str = "marshal_turn_level_reinforce"
    gamma: float = 1.0
    reward_normalization: str = "mean"
    advantage_normalization: str = "mean"
    whiten_rewards: bool = True
    whiten_advantages: bool = True

    def assign(self, trajectories: list[MultiAgentTrajectory], team: TeamSpec) -> list[TrainingSample]:
        trainable_agents = {agent.name for agent in team.trainable_agents()}
        rows = [
            (trajectory, turn)
            for trajectory in trajectories
            for turn in trajectory.trainable_turns(trainable_agents)
        ]
        if not rows:
            return []
        result = compute_marshal_scalar_advantages(
            turn_rewards=torch.tensor([float(turn.step_reward or 0.0) for _, turn in rows]),
            episode_ids=[trajectory.episode_id for trajectory, _ in rows],
            player_ids=[turn.metadata["marshal_player_id"] for _, turn in rows],
            player_turn_ids=[int(turn.metadata["marshal_player_turn"]) for _, turn in rows],
            gamma=self.gamma,
            reward_normalization=self.reward_normalization,
            advantage_normalization=self.advantage_normalization,
            whiten_rewards=self.whiten_rewards,
            whiten_advantages=self.whiten_advantages,
        )
        samples: list[TrainingSample] = []
        for row, (trajectory, turn) in enumerate(rows):
            advantage = float(result.advantages[row])
            turn.advantage = advantage
            metadata = {
                **turn.metadata,
                "credit": self.name,
                "marshal_raw_turn_reward": float(result.raw_rewards[row]),
                "marshal_normalized_turn_reward": float(result.normalized_rewards[row]),
                "marshal_turn_return": float(result.turn_returns[row]),
                "marshal_agent_normalized_advantage": float(result.agent_normalized_advantages[row]),
            }
            samples.append(
                TrainingSample(
                    sample_id=f"{trajectory.episode_id}:p{turn.metadata['marshal_player_id']}:{turn.turn_id}",
                    episode_id=trajectory.episode_id,
                    task_id=trajectory.task_id,
                    rollout_group=trajectory.rollout_group,
                    turn_id=turn.turn_id,
                    agent_name=turn.agent_name,
                    role=turn.role,
                    policy_group=turn.policy_group,
                    prompt=turn.prompt,
                    response=turn.action_text,
                    response_token_ids=turn.action_token_ids,
                    response_logprobs=turn.action_logprobs,
                    reward=float(turn.step_reward or 0.0),
                    advantage=advantage,
                    metadata=metadata,
                )
            )
        return samples


def _normalize_by_player(
    values: torch.Tensor,
    *,
    player_ids: list[Any],
    active_mask: list[bool],
    mode: str,
    epsilon: float,
) -> torch.Tensor:
    output = torch.zeros_like(values, dtype=torch.float32)
    rows_by_player: dict[str, list[int]] = defaultdict(list)
    for row, (player_id, is_active) in enumerate(zip(player_ids, active_mask, strict=True)):
        if is_active:
            rows_by_player[str(player_id)].append(row)
    with torch.no_grad():
        for rows in rows_by_player.values():
            player_values = values[rows].float()
            if mode == "identity":
                normalized = player_values
            else:
                normalized = player_values - player_values.mean()
                if mode == "mean_std":
                    std = player_values.std(unbiased=True) if len(rows) > 1 else player_values.new_tensor(0.0)
                    normalized = (
                        normalized / (std + epsilon)
                        if float(std.abs()) > epsilon
                        else torch.zeros_like(normalized)
                    )
            output[rows] = normalized
    return output


def _normalize_unique_returns_by_player(
    values: torch.Tensor,
    *,
    player_ids: list[Any],
    active_mask: list[bool],
    mode: str,
    epsilon: float,
) -> torch.Tensor:
    output = torch.zeros_like(values, dtype=torch.float32)
    rows_by_player: dict[str, list[int]] = defaultdict(list)
    for row, (player_id, is_active) in enumerate(zip(player_ids, active_mask, strict=True)):
        if is_active:
            rows_by_player[str(player_id)].append(row)
    with torch.no_grad():
        for rows in rows_by_player.values():
            player_values = values[rows].float()
            unique_values = torch.unique(player_values)
            if unique_values.numel() <= 1:
                continue
            centered_unique = unique_values - unique_values.mean()
            if mode == "mean_std":
                std = unique_values.std(unbiased=True)
                centered_unique = centered_unique / (std + epsilon)
            for original, normalized in zip(unique_values, centered_unique, strict=True):
                row_mask = player_values == original
                output[torch.tensor(rows, device=values.device)[row_mask]] = normalized
    return output


def _whiten_active(values: torch.Tensor, *, active_mask: list[bool], epsilon: float) -> torch.Tensor:
    output = torch.zeros_like(values, dtype=torch.float32)
    rows = [row for row, is_active in enumerate(active_mask) if is_active]
    if not rows:
        return output
    active_values = values[rows].float()
    centered = active_values - active_values.mean()
    if len(rows) == 1:
        output[rows] = 0.0
        return output
    std = active_values.std(unbiased=True)
    output[rows] = centered / (std + epsilon) if float(std.abs()) > epsilon else torch.zeros_like(centered)
    return output
