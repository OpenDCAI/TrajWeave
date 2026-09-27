from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Any, Callable, Sequence

from trajweave.core.specs import TeamSpec
from trajweave.core.trajectory import AgentTurn, MultiAgentTrajectory, TrainingSample


@dataclass(frozen=True)
class C3CreditResult:
    advantages: list[float]
    baselines: list[float]
    q_values: list[float | None]
    group_sizes: list[int]


def compute_c3_scalar_credit(
    *,
    subtree_returns: Sequence[float],
    group_ids: Sequence[Any],
    q_values: Sequence[float] | None = None,
    variant: str = "reward_only",
    baseline_mode: str = "loo",
    value_assisted_alpha: float = 1.0,
    normalize: bool = False,
    epsilon: float = 1e-8,
) -> C3CreditResult:
    """Compute C3 Rule-B sibling credit from frozen-context prefix groups."""

    returns = [float(value) for value in subtree_returns]
    groups = [str(value) for value in group_ids]
    if len(returns) != len(groups):
        raise ValueError("subtree_returns and group_ids must have the same length.")
    if any(not isfinite(value) for value in returns):
        raise ValueError("C3 subtree returns must be finite.")
    variant_name = str(variant).strip().lower()
    if variant_name not in {"reward_only", "value_only", "value_assisted"}:
        raise ValueError("C3 variant must be reward_only, value_only, or value_assisted.")
    baseline_name = str(baseline_mode).strip().lower()
    if baseline_name not in {"loo", "full_mean"}:
        raise ValueError("C3 baseline_mode must be loo or full_mean.")
    alpha = float(value_assisted_alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("C3 value_assisted_alpha must be in [0, 1].")
    q = None if q_values is None else [float(value) for value in q_values]
    if variant_name != "reward_only":
        if q is None or len(q) != len(returns):
            raise ValueError(f"C3 {variant_name} requires one q_value per row.")
        if any(not isfinite(value) for value in q):
            raise ValueError("C3 q_values must be finite.")

    rows_by_group: dict[str, list[int]] = defaultdict(list)
    for row, group_id in enumerate(groups):
        if not group_id:
            raise ValueError(f"C3 group id is empty at row {row}.")
        rows_by_group[group_id].append(row)
    advantages = [0.0] * len(returns)
    baselines = [0.0] * len(returns)
    group_sizes = [0] * len(returns)

    for group_id, rows in rows_by_group.items():
        if baseline_name == "loo" and len(rows) < 2:
            raise ValueError(f"C3 LOO group {group_id!r} requires at least two sibling alternatives.")
        reward_group = [returns[row] for row in rows]
        q_group = [q[row] for row in rows] if q is not None else None
        reward_baseline = _baselines(reward_group, baseline_name)
        q_baseline = _baselines(q_group, baseline_name) if q_group is not None else None
        group_advantages = []
        group_baselines = []
        for offset, row in enumerate(rows):
            if variant_name == "reward_only":
                baseline = reward_baseline[offset]
                advantage = returns[row] - baseline
            elif variant_name == "value_only":
                assert q is not None and q_baseline is not None
                baseline = q_baseline[offset]
                advantage = q[row] - baseline
            else:
                assert q_baseline is not None
                baseline = (1.0 - alpha) * reward_baseline[offset] + alpha * q_baseline[offset]
                advantage = returns[row] - baseline
            group_advantages.append(advantage)
            group_baselines.append(baseline)

        if normalize and len(group_advantages) > 1:
            mean = sum(group_advantages) / len(group_advantages)
            variance = sum((value - mean) ** 2 for value in group_advantages) / len(group_advantages)
            std = sqrt(variance + float(epsilon))
            group_advantages = [(value - mean) / std for value in group_advantages]
        for offset, row in enumerate(rows):
            advantages[row] = float(group_advantages[offset])
            baselines[row] = float(group_baselines[offset])
            group_sizes[row] = len(rows)

    return C3CreditResult(
        advantages=advantages,
        baselines=baselines,
        q_values=[None] * len(returns) if q is None else [float(value) for value in q],
        group_sizes=group_sizes,
    )


@dataclass
class C3CreditAssigner:
    name: str = "c3_contextual_counterfactual"
    variant: str = "reward_only"
    baseline_mode: str = "loo"
    value_assisted_alpha: float = 1.0
    normalize: bool = True
    q_scorer: Callable[[Sequence[str]], Sequence[float]] | None = None

    def assign(self, trajectories: list[MultiAgentTrajectory], team: TeamSpec) -> list[TrainingSample]:
        trainable = {agent.name for agent in team.trainable_agents()}
        rows: list[tuple[MultiAgentTrajectory, AgentTurn]] = [
            (trajectory, turn)
            for trajectory in trajectories
            for turn in trajectory.turns
            if turn.agent_name in trainable
        ]
        if not rows:
            return []
        prefix_texts = [str(turn.metadata.get("c3_prefix_text", "")) for _, turn in rows]
        q_values = None
        if self.variant != "reward_only":
            if self.q_scorer is None:
                raise ValueError(f"C3 {self.variant} requires q_scorer.")
            q_values = [float(value) for value in self.q_scorer(prefix_texts)]
        result = compute_c3_scalar_credit(
            subtree_returns=[_required_subtree_return(turn) for _, turn in rows],
            group_ids=[_required_group_id(turn) for _, turn in rows],
            q_values=q_values,
            variant=self.variant,
            baseline_mode=self.baseline_mode,
            value_assisted_alpha=self.value_assisted_alpha,
            normalize=self.normalize,
        )
        samples = []
        for row, (trajectory, turn) in enumerate(rows):
            advantage = result.advantages[row]
            turn.advantage = advantage
            metadata = {
                **turn.metadata,
                "credit": self.name,
                "c3_credit_variant": self.variant,
                "c3_baseline_mode": self.baseline_mode,
                "c3_baseline": result.baselines[row],
                "c3_q_value": result.q_values[row],
                "c3_group_size": result.group_sizes[row],
                "c3_advantage": advantage,
            }
            samples.append(
                TrainingSample(
                    sample_id=f"{trajectory.episode_id}:{turn.node_id}",
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
                    reward=_required_subtree_return(turn),
                    advantage=advantage,
                    root_id=turn.root_id,
                    node_id=turn.node_id,
                    parent_node_id=turn.parent_node_id,
                    observation_group_id=turn.observation_group_id,
                    branch_index=turn.branch_index,
                    selected_for_expansion=turn.selected_for_expansion,
                    local_score=turn.local_score,
                    metadata=metadata,
                )
            )
        return samples


def _baselines(values: Sequence[float] | None, mode: str) -> list[float]:
    if values is None:
        raise ValueError("baseline values are required.")
    total = sum(values)
    if mode == "full_mean":
        return [total / len(values)] * len(values)
    return [(total - value) / (len(values) - 1) for value in values]


def _required_subtree_return(turn: AgentTurn) -> float:
    value = turn.metadata.get("c3_subtree_return")
    if value is None:
        raise ValueError(f"C3 turn {turn.node_id!r} is missing c3_subtree_return.")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"C3 turn {turn.node_id!r} has a non-finite subtree return.")
    return result


def _required_group_id(turn: AgentTurn) -> str:
    value = str(turn.metadata.get("c3_group_id", turn.observation_group_id or ""))
    if not value:
        raise ValueError(f"C3 turn {turn.node_id!r} is missing c3_group_id.")
    return value
