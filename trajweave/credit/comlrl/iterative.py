from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class IndexedPolicyComparison:
    candidate_index: int
    winner_source: str
    loser_source: str
    current_reward: float
    comparator_reward: float

    @property
    def reward_gap(self) -> float:
        return abs(self.current_reward - self.comparator_reward)


def compare_policy_candidates_by_index(
    current_rewards: Sequence[float],
    comparator_rewards: Sequence[float],
) -> list[IndexedPolicyComparison]:
    """Compare current/comparator candidates at the same index, matching CoMLRL v1.4.1."""
    current = _finite_rewards(current_rewards, "current_rewards")
    comparator = _finite_rewards(comparator_rewards, "comparator_rewards")
    comparisons: list[IndexedPolicyComparison] = []
    for candidate_index in range(min(len(current), len(comparator))):
        current_reward = current[candidate_index]
        comparator_reward = comparator[candidate_index]
        if current_reward == comparator_reward:
            continue
        comparisons.append(
            IndexedPolicyComparison(
                candidate_index=candidate_index,
                winner_source="current" if current_reward > comparator_reward else "comparator",
                loser_source="comparator" if current_reward > comparator_reward else "current",
                current_reward=current_reward,
                comparator_reward=comparator_reward,
            )
        )
    return comparisons


def select_policy_comparisons(
    comparisons: Sequence[IndexedPolicyComparison],
    *,
    mode: str = "comparator_reward",
    limit: int | None = 4,
    seed: int = 0,
) -> list[IndexedPolicyComparison]:
    selection = str(mode).strip().lower()
    if selection not in {"reward_gap", "all", "random", "comparator_reward"}:
        raise ValueError("pair selection must be one of: reward_gap, all, random, comparator_reward")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("pair selection limit must be a positive integer or None")
    if selection == "all" and limit is not None:
        raise ValueError("pair selection limit must be None when mode='all'")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    selected = list(comparisons)
    if any(not isinstance(item, IndexedPolicyComparison) for item in selected):
        raise TypeError("comparisons must contain IndexedPolicyComparison values")
    if selection == "random":
        random.Random(seed).shuffle(selected)
    elif selection == "all":
        selected.sort(key=lambda item: item.candidate_index)
    elif selection == "comparator_reward":
        selected.sort(key=lambda item: (item.comparator_reward, item.reward_gap), reverse=True)
    else:
        selected.sort(key=lambda item: item.reward_gap, reverse=True)
    return selected if selection == "all" or limit is None else selected[:limit]


def _finite_rewards(values: Sequence[float], name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of finite numbers")
    output: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError(f"{name} must contain numbers")
        normalized = float(value)
        if not isfinite(normalized):
            raise FloatingPointError(f"{name} must contain only finite values")
        output.append(normalized)
    return tuple(output)


__all__ = [
    "IndexedPolicyComparison",
    "compare_policy_candidates_by_index",
    "select_policy_comparisons",
]
