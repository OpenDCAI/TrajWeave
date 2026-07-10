from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from trajweave.core.specs import TeamSpec
from trajweave.core.trajectory import MultiAgentTrajectory, TrainingSample
from trajweave.credit.common import StepGroupBuilder, compute_discounted_step_returns


@dataclass(frozen=True)
class GiGPOAdvantageResult:
    advantages: torch.Tensor
    episode_advantages: torch.Tensor
    step_advantages: torch.Tensor
    step_returns: torch.Tensor
    step_group_ids: np.ndarray
    step_group_sizes: dict[str, int]


def compute_gigpo_scalar_advantages(
    *,
    episode_rewards: torch.Tensor,
    step_rewards: torch.Tensor,
    rollout_groups: list[Any],
    trajectory_ids: list[Any],
    turn_ids: list[int],
    anchor_observations: list[Any],
    gamma: float = 0.95,
    step_advantage_weight: float = 1.0,
    mode: str = "mean_std_norm",
    enable_similarity: bool = False,
    similarity_threshold: float = 0.95,
    active_mask: list[bool] | None = None,
    epsilon: float = 1e-6,
) -> GiGPOAdvantageResult:
    """实现 GiGPO 的 episode + step 两级相对优势。"""

    if episode_rewards.ndim != 1 or step_rewards.ndim != 1:
        raise ValueError("episode_rewards and step_rewards must be 1D tensors.")
    row_count = episode_rewards.shape[0]
    sequences = (step_rewards, rollout_groups, trajectory_ids, turn_ids, anchor_observations)
    if any(len(sequence) != row_count for sequence in sequences):
        raise ValueError("All GiGPO row fields must have the same length.")
    if step_advantage_weight < 0.0:
        raise ValueError("step_advantage_weight must be non-negative.")
    if mode not in {"mean_norm", "mean_std_norm"}:
        raise ValueError("mode must be 'mean_norm' or 'mean_std_norm'.")
    active = [True] * row_count if active_mask is None else [bool(value) for value in active_mask]
    if len(active) != row_count:
        raise ValueError("active_mask must align with episode_rewards.")

    step_returns = compute_discounted_step_returns(
        step_rewards=step_rewards,
        trajectory_ids=trajectory_ids,
        turn_ids=turn_ids,
        gamma=gamma,
        active_mask=active,
    )
    step_groups = StepGroupBuilder(
        enable_similarity=enable_similarity,
        similarity_threshold=similarity_threshold,
    ).build(
        anchor_observations=anchor_observations,
        rollout_groups=rollout_groups,
        active_mask=active,
    )
    episode_advantages = _normalize_grouped_scores(
        episode_rewards.float(),
        group_ids=[str(value) for value in rollout_groups],
        active_mask=active,
        mode=mode,
        preserve_singleton_score=True,
        epsilon=epsilon,
    )
    step_advantages = _normalize_grouped_scores(
        step_returns,
        group_ids=[str(value) for value in step_groups.group_ids],
        active_mask=active,
        mode=mode,
        preserve_singleton_score=False,
        epsilon=epsilon,
    )
    advantages = episode_advantages + step_advantage_weight * step_advantages
    return GiGPOAdvantageResult(
        advantages=advantages,
        episode_advantages=episode_advantages,
        step_advantages=step_advantages,
        step_returns=step_returns,
        step_group_ids=step_groups.group_ids,
        step_group_sizes=step_groups.group_sizes,
    )


@dataclass
class GiGPOCreditAssigner:
    name: str = "gigpo_hierarchical_grpo"
    gamma: float = 0.95
    step_advantage_weight: float = 1.0
    mode: str = "mean_std_norm"
    enable_similarity: bool = False
    similarity_threshold: float = 0.95

    def assign(self, trajectories: list[MultiAgentTrajectory], team: TeamSpec) -> list[TrainingSample]:
        trainable_agents = {agent.name for agent in team.trainable_agents()}
        rows: list[tuple[MultiAgentTrajectory, Any]] = []
        for trajectory in trajectories:
            turns = trajectory.trainable_turns(trainable_agents)
            for index, turn in enumerate(turns):
                turn.reward = float(trajectory.global_reward or 0.0)
                turn.step_reward = float(trajectory.global_reward or 0.0) if index == len(turns) - 1 else 0.0
                rows.append((trajectory, turn))
        if not rows:
            return []

        device = torch.device("cpu")
        result = compute_gigpo_scalar_advantages(
            episode_rewards=torch.tensor([float(traj.global_reward or 0.0) for traj, _ in rows], device=device),
            step_rewards=torch.tensor([float(turn.step_reward or 0.0) for _, turn in rows], device=device),
            rollout_groups=[traj.rollout_group for traj, _ in rows],
            trajectory_ids=[traj.episode_id for traj, _ in rows],
            turn_ids=[turn.turn_id for _, turn in rows],
            anchor_observations=[_anchor_observation(turn) for _, turn in rows],
            gamma=self.gamma,
            step_advantage_weight=self.step_advantage_weight,
            mode=self.mode,
            enable_similarity=self.enable_similarity,
            similarity_threshold=self.similarity_threshold,
        )

        samples: list[TrainingSample] = []
        for row, (trajectory, turn) in enumerate(rows):
            advantage = float(result.advantages[row])
            turn.advantage = advantage
            metadata = {
                "credit": self.name,
                "anchor_observation": _anchor_observation(turn),
                "next_observation": turn.next_observation,
                "step_reward": float(turn.step_reward or 0.0),
                "step_return": float(result.step_returns[row]),
                "episode_advantage": float(result.episode_advantages[row]),
                "step_advantage": float(result.step_advantages[row]),
                "step_group_uid": str(result.step_group_ids[row]),
                "step_group_size": result.step_group_sizes.get(str(result.step_group_ids[row]), 0),
                **turn.metadata,
            }
            samples.append(
                TrainingSample(
                    sample_id=f"{trajectory.episode_id}:{turn.turn_id}:{turn.agent_name}",
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
                    reward=float(trajectory.global_reward or 0.0),
                    advantage=advantage,
                    metadata=metadata,
                )
            )
        return samples


def _normalize_grouped_scores(
    scores: torch.Tensor,
    *,
    group_ids: list[str],
    active_mask: list[bool],
    mode: str,
    preserve_singleton_score: bool,
    epsilon: float,
) -> torch.Tensor:
    normalized = torch.zeros_like(scores, dtype=torch.float32)
    rows_by_group: dict[str, list[int]] = defaultdict(list)
    for row, (group_id, is_active) in enumerate(zip(group_ids, active_mask, strict=True)):
        if is_active:
            rows_by_group[group_id].append(row)

    with torch.no_grad():
        for rows in rows_by_group.values():
            values = scores[rows].float()
            if len(rows) == 1:
                mean = values.new_tensor(0.0) if preserve_singleton_score else values[0]
                std = values.new_tensor(1.0)
            else:
                mean = values.mean()
                std = values.std(unbiased=True)
            group_values = values - mean
            if mode == "mean_std_norm":
                group_values = group_values / (std + epsilon)
            normalized[rows] = group_values
    return normalized


def _anchor_observation(turn: Any) -> Any:
    if turn.anchor_observation is not None:
        return turn.anchor_observation
    return {"agent_id": turn.agent_name, "observation": turn.observation}
