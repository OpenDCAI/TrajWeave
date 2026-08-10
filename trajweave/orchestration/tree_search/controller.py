from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TreeSearchController:
    """Stateful worker-side controller for incremental rollout sessions."""

    max_num_nodes: int = 2
    initial_candidates: int = 2
    exploration_constant: float = 1.0
    stop_on_success: bool = False
    records: dict[int, dict[str, Any]] = field(default_factory=dict)
    visits: dict[int, int] = field(default_factory=dict)
    value_sums: dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_num_nodes <= 0:
            raise ValueError("TreeSearchController.max_num_nodes must be positive.")
        self.initial_candidates = max(1, min(int(self.initial_candidates), self.max_num_nodes))

    def select_parent(self, node_id: int) -> int:
        if node_id < self.initial_candidates:
            return -1
        candidates = [
            (candidate_id, record)
            for candidate_id, record in self.records.items()
            if not bool(record.get("terminal", False))
        ]
        if not candidates:
            return -1
        total_visits = max(1, sum(self.visits.values()))

        def score(item: tuple[int, dict[str, Any]]) -> tuple[float, int]:
            candidate_id, _record = item
            count = self.visits.get(candidate_id, 0)
            mean = self.value_sums.get(candidate_id, 0.0) / max(count, 1)
            exploration = self.exploration_constant * math.sqrt(
                math.log(total_visits + 1) / (count + 1)
            )
            return mean + exploration, -candidate_id

        return max(candidates, key=score)[0]

    def path_for(self, node_id: int, parent_idx: int) -> tuple[int, ...]:
        if parent_idx < 0:
            return (node_id,)
        parent = self.records.get(parent_idx)
        if not parent:
            return (node_id,)
        return (*tuple(parent.get("path", (parent_idx,))), node_id)

    def record(
        self,
        *,
        node_id: int,
        parent_idx: int,
        path: tuple[int, ...],
        reward: float,
        feedback: str,
        success: bool,
        terminal: bool,
        response_ids: list[int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.records[node_id] = {
            "parent_idx": parent_idx,
            "path": tuple(path),
            "reward": float(reward),
            "feedback": feedback,
            "success": bool(success),
            "terminal": bool(terminal),
            "response_ids": list(response_ids or []),
            "metadata": dict(metadata or {}),
        }
        current: int | None = node_id
        while current is not None and current >= 0:
            self.visits[current] = self.visits.get(current, 0) + 1
            self.value_sums[current] = self.value_sums.get(current, 0.0) + float(reward)
            parent = self.records.get(current, {}).get("parent_idx", -1)
            current = int(parent) if parent is not None and int(parent) >= 0 else None

    def should_stop(self, *, success: bool, pending_node: bool = True) -> bool:
        observed = len(self.records) + (1 if pending_node else 0)
        if observed >= self.max_num_nodes:
            return True
        return bool(self.stop_on_success and success and observed >= self.initial_candidates)


__all__ = ["TreeSearchController"]
