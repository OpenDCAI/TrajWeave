from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch


def compute_discounted_step_returns(
    *,
    step_rewards: torch.Tensor,
    trajectory_ids: list[Any],
    turn_ids: list[int],
    gamma: float,
    active_mask: list[bool] | None = None,
) -> torch.Tensor:
    """按 trajectory 和 turn 顺序计算折扣 step return。"""

    if step_rewards.ndim != 1:
        raise ValueError("step_rewards must be a 1D tensor.")
    row_count = step_rewards.shape[0]
    if len(trajectory_ids) != row_count or len(turn_ids) != row_count:
        raise ValueError("trajectory_ids and turn_ids must align with step_rewards.")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1].")
    active = [True] * row_count if active_mask is None else [bool(value) for value in active_mask]
    if len(active) != row_count:
        raise ValueError("active_mask must align with step_rewards.")

    rows_by_trajectory: dict[str, list[int]] = defaultdict(list)
    for row, (trajectory_id, is_active) in enumerate(zip(trajectory_ids, active, strict=True)):
        if is_active:
            rows_by_trajectory[str(trajectory_id)].append(row)

    returns = torch.zeros_like(step_rewards, dtype=torch.float32)
    for trajectory_id, rows in rows_by_trajectory.items():
        ordered_rows = sorted(rows, key=lambda row: int(turn_ids[row]))
        ordered_turns = [int(turn_ids[row]) for row in ordered_rows]
        if len(set(ordered_turns)) != len(ordered_turns):
            raise ValueError(f"trajectory {trajectory_id!r} contains duplicate turn_ids: {ordered_turns}.")
        running_return = step_rewards.new_tensor(0.0, dtype=torch.float32)
        for row in reversed(ordered_rows):
            running_return = step_rewards[row].float() + gamma * running_return
            returns[row] = running_return
    return returns
