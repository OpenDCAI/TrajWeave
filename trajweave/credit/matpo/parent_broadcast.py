from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from math import isclose, isfinite, sqrt

from trajweave.core.specs import TeamSpec
from trajweave.core.trajectory import MultiAgentTrajectory, TrainingSample
from trajweave.credit.global_broadcast import GlobalBroadcastCreditAssigner

DEFAULT_ACCURACY_REWARD_WEIGHT = 0.9
DEFAULT_TOOL_FORMAT_REWARD_WEIGHT = 0.1


def combined_matpo_reward(
    *,
    accuracy: float,
    planner_format: float,
    worker_formats: Iterable[float],
    accuracy_reward_weight: float = DEFAULT_ACCURACY_REWARD_WEIGHT,
    tool_format_reward_weight: float = DEFAULT_TOOL_FORMAT_REWARD_WEIGHT,
) -> float:
    """Return the strict MATPO accuracy/format reward shared by local and VERL paths."""

    accuracy_score = _unit_score(accuracy, field="accuracy")
    planner_format_score = _unit_score(planner_format, field="planner_format")
    worker_format_scores = tuple(_unit_score(value, field="worker_format") for value in worker_formats)
    accuracy_weight = _unit_score(accuracy_reward_weight, field="accuracy_reward_weight")
    tool_format_weight = _unit_score(tool_format_reward_weight, field="tool_format_reward_weight")
    if not isclose(accuracy_weight + tool_format_weight, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("MATPO accuracy_reward_weight and tool_format_reward_weight must sum to 1.0.")

    mean_worker_format = sum(worker_format_scores) / len(worker_format_scores) if worker_format_scores else 0.0
    return accuracy_weight * accuracy_score + tool_format_weight * 0.5 * (planner_format_score + mean_worker_format)


def apply_matpo_trajectory_reward(
    trajectory: MultiAgentTrajectory,
    *,
    accuracy_reward_weight: float = DEFAULT_ACCURACY_REWARD_WEIGHT,
    tool_format_reward_weight: float = DEFAULT_TOOL_FORMAT_REWARD_WEIGHT,
) -> float:
    """Compute and record the canonical MATPO rollout reward exactly once."""

    planner_format, worker_formats = _trajectory_format_scores(trajectory)
    accuracy = float(trajectory.global_reward or 0.0)
    reward = combined_matpo_reward(
        accuracy=accuracy,
        planner_format=planner_format,
        worker_formats=worker_formats,
        accuracy_reward_weight=accuracy_reward_weight,
        tool_format_reward_weight=tool_format_reward_weight,
    )
    trajectory.metadata.update(
        {
            "matpo_accuracy_reward": accuracy,
            "matpo_planner_format": planner_format,
            "matpo_worker_formats": worker_formats,
            "matpo_combined_reward": reward,
        }
    )
    return reward


@dataclass
class MATPOParentBroadcastCreditAssigner(GlobalBroadcastCreditAssigner):
    name: str = "matpo_parent_broadcast_grpo"
    epsilon: float = 1e-6
    normalize_by_std: bool = True
    accuracy_reward_weight: float = DEFAULT_ACCURACY_REWARD_WEIGHT
    tool_format_reward_weight: float = DEFAULT_TOOL_FORMAT_REWARD_WEIGHT

    def assign(self, trajectories: list[MultiAgentTrajectory], team: TeamSpec) -> list[TrainingSample]:
        trajectory_rewards: dict[str, float] = {}
        for trajectory in trajectories:
            combined_reward = apply_matpo_trajectory_reward(
                trajectory,
                accuracy_reward_weight=self.accuracy_reward_weight,
                tool_format_reward_weight=self.tool_format_reward_weight,
            )
            trajectory_rewards[trajectory.episode_id] = combined_reward
            for turn in trajectory.turns:
                turn.reward = combined_reward

        samples = super().assign(trajectories, team)
        _validate_training_sample_links(samples)
        grouped_trajectories: dict[str, list[MultiAgentTrajectory]] = defaultdict(list)
        for trajectory in trajectories:
            grouped_trajectories[trajectory.rollout_group].append(trajectory)

        trajectory_advantages: dict[str, tuple[float, float, float]] = {}
        for group_trajectories in grouped_trajectories.values():
            rewards = [trajectory_rewards[trajectory.episode_id] for trajectory in group_trajectories]
            mean = sum(rewards) / len(rewards)
            if len(rewards) > 1:
                variance = sum((reward - mean) ** 2 for reward in rewards) / (len(rewards) - 1)
                std = sqrt(variance)
            else:
                std = 1.0
            if std < self.epsilon:
                std = 1.0
            for trajectory, reward in zip(group_trajectories, rewards, strict=True):
                advantage = reward - mean
                if self.normalize_by_std:
                    advantage /= std + self.epsilon
                trajectory_advantages[trajectory.episode_id] = (advantage, mean, std)

        for sample in samples:
            advantage, mean, std = trajectory_advantages[sample.episode_id]
            sample.reward = trajectory_rewards[sample.episode_id]
            sample.advantage = advantage
            sample.metadata.update(
                {
                    "advantage_group": sample.rollout_group,
                    "reward_mean": mean,
                    "reward_std": std,
                    "trajectory_advantage_broadcast": True,
                }
            )
            if bool(sample.metadata.get("is_from_subagent_tool", False)):
                sample.metadata["parent_advantage_broadcast"] = True
        return samples


def _validate_training_sample_links(samples: list[TrainingSample]) -> None:
    req_to_sample: dict[str, TrainingSample] = {}
    for sample in samples:
        reqs_id = str(sample.metadata.get("reqs_id", ""))
        if not reqs_id:
            raise ValueError(f"MATPO sample {sample.sample_id!r} is missing reqs_id.")
        if reqs_id in req_to_sample:
            raise ValueError(f"MATPO duplicate reqs_id: {reqs_id!r}.")
        req_to_sample[reqs_id] = sample

    for sample in samples:
        if not bool(sample.metadata.get("is_from_subagent_tool", False)):
            continue
        parent_id = str(sample.metadata.get("parent_reqs_id", ""))
        if not parent_id:
            raise ValueError(f"MATPO child sample {sample.sample_id!r} is missing parent_reqs_id.")
        parent = req_to_sample.get(parent_id)
        if parent is None:
            raise ValueError(f"MATPO child sample {sample.sample_id!r} has unknown parent {parent_id!r}.")
        if bool(parent.metadata.get("is_from_subagent_tool", False)):
            raise ValueError(f"MATPO child sample {sample.sample_id!r} cannot reference another child.")


def _trajectory_format_scores(trajectory: MultiAgentTrajectory) -> tuple[float, tuple[float, ...]]:
    planner_events = [
        turn
        for turn in trajectory.turns
        if not bool(turn.metadata.get("is_from_subagent_tool", False))
        and turn.metadata.get("matpo_turn_role") in {"delegate", "invalid_planner_call", "invalid_tool"}
    ]
    planner_format = float(
        bool(planner_events) and all(bool(turn.metadata.get("matpo_tool_format_valid")) for turn in planner_events)
    )
    worker_formats = tuple(
        float(bool(turn.metadata.get("matpo_tool_format_valid")))
        for turn in trajectory.turns
        if turn.metadata.get("matpo_turn_role") in {"worker_call", "invalid_worker_call"}
    )
    return planner_format, worker_formats


def _unit_score(value: float, *, field: str) -> float:
    if isinstance(value, bool):
        score = float(value)
    else:
        try:
            score = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"MATPO {field} must be a finite number between 0 and 1.") from exc
    if not isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"MATPO {field} must be a finite number between 0 and 1.")
    return score
