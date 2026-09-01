from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from trajweave.core.tree import SearchNode


@dataclass(frozen=True)
class TreeGroupBuilder:
    require_contiguous: bool = True

    def group_indices(self, nodes: list[SearchNode]) -> list[list[int]]:
        groups: list[list[int]] = []
        tree_to_group: dict[str, int] = {}
        closed_trees: set[str] = set()
        current_tree_id: str | None = None

        for index, node in enumerate(nodes):
            tree_id = node.tree_id
            if tree_id in tree_to_group:
                group_index = tree_to_group[tree_id]
                if self.require_contiguous and current_tree_id != tree_id:
                    raise ValueError(
                        f"Tree {tree_id!r} is not contiguous in rollout order; explicit tree grouping is required."
                    )
                groups[group_index].append(index)
            else:
                if self.require_contiguous and tree_id in closed_trees:
                    raise ValueError(f"Tree {tree_id!r} reappeared after another tree.")
                tree_to_group[tree_id] = len(groups)
                groups.append([index])

            if current_tree_id is not None and current_tree_id != tree_id:
                closed_trees.add(current_tree_id)
            current_tree_id = tree_id

        return groups

    def fixed_size_indices(self, sample_count: int, group_size: int) -> list[list[int]]:
        if group_size <= 0:
            raise ValueError("group_size must be positive.")
        if sample_count % group_size != 0:
            raise ValueError(f"sample_count={sample_count} is not divisible by group_size={group_size}.")
        return [list(range(start, start + group_size)) for start in range(0, sample_count, group_size)]

    def fixed_size_matches_tree_groups(self, nodes: list[SearchNode], group_size: int) -> bool:
        return self.fixed_size_indices(len(nodes), group_size) == self.group_indices(nodes)


def group_normalized_advantages(rewards: list[float], groups: list[list[int]], *, eps: float = 1e-9) -> list[float]:
    advantages = [0.0 for _ in rewards]
    for group in groups:
        if not group:
            raise ValueError("advantage group must not be empty.")
        values = [float(rewards[index]) for index in group]
        mean = sum(values) / len(values)
        std = _sample_std(values, mean)
        for index in group:
            if std <= eps:
                advantages[index] = 0.0
            else:
                advantages[index] = (float(rewards[index]) - mean) / (std + eps)
    return advantages


def rewards_from_nodes(nodes: list[SearchNode]) -> list[float]:
    return [node.effective_reward for node in nodes]


def importance_correction_weights(
    *,
    old_logprobs: list[list[float]],
    rollout_logprobs: list[list[float]],
    action_mask: list[list[int | bool | float]],
    level: Literal["token", "sequence", "geometric"] = "token",
    mode: Literal["truncate", "mask"] = "truncate",
    upper_threshold: float = 2.0,
    lower_threshold: float | None = None,
) -> list[list[float]]:
    if upper_threshold <= 0:
        raise ValueError("upper_threshold must be positive.")
    if lower_threshold is None:
        lower_threshold = 1.0 / upper_threshold
    if lower_threshold <= 0:
        raise ValueError("lower_threshold must be positive.")
    _validate_same_shape(old_logprobs, rollout_logprobs, action_mask)

    rows: list[list[float]] = []
    for old_row, rollout_row, mask_row in zip(old_logprobs, rollout_logprobs, action_mask, strict=True):
        active = [bool(value) for value in mask_row]
        log_ratios = [float(old) - float(rollout) for old, rollout in zip(old_row, rollout_row, strict=True)]
        if level == "token":
            raw = [math.exp(value) for value in log_ratios]
        elif level == "sequence":
            raw_value = math.exp(sum(value for value, is_active in zip(log_ratios, active, strict=True) if is_active))
            raw = [raw_value for _ in log_ratios]
        elif level == "geometric":
            active_values = [value for value, is_active in zip(log_ratios, active, strict=True) if is_active]
            raw_value = math.exp(sum(active_values) / len(active_values)) if active_values else 1.0
            raw = [raw_value for _ in log_ratios]
        else:
            raise ValueError(f"Unsupported importance correction level: {level!r}.")

        if mode == "truncate":
            row = [
                min(value, upper_threshold) if is_active else 0.0 for value, is_active in zip(raw, active, strict=True)
            ]
        elif mode == "mask":
            row = [
                value if is_active and lower_threshold <= value <= upper_threshold else 0.0
                for value, is_active in zip(raw, active, strict=True)
            ]
        else:
            raise ValueError(f"Unsupported importance correction mode: {mode!r}.")
        rows.append(row)
    return rows


def _sample_std(values: list[float], mean: float) -> float:
    if len(values) <= 1:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _validate_same_shape(
    old_logprobs: list[list[float]],
    rollout_logprobs: list[list[float]],
    action_mask: list[list[int | bool | float]],
) -> None:
    if len(old_logprobs) != len(rollout_logprobs) or len(old_logprobs) != len(action_mask):
        raise ValueError("old_logprobs, rollout_logprobs, and action_mask must have the same batch size.")
    for row, (old_row, rollout_row, mask_row) in enumerate(
        zip(old_logprobs, rollout_logprobs, action_mask, strict=True)
    ):
        if len(old_row) != len(rollout_row) or len(old_row) != len(mask_row):
            raise ValueError(f"importance correction inputs have mismatched lengths at row {row}.")
