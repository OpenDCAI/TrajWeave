from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

import numpy as np


@dataclass(frozen=True)
class StepGroupResult:
    group_ids: np.ndarray
    group_sizes: dict[str, int]

    @property
    def valid_group_count(self) -> int:
        return len(self.group_sizes)

    @property
    def average_group_size(self) -> float:
        if not self.group_sizes:
            return 0.0
        return sum(self.group_sizes.values()) / len(self.group_sizes)


@dataclass(frozen=True)
class StepGroupBuilder:
    """在同一任务组内，按 anchor observation 构造 step-level 对比组。"""

    enable_similarity: bool = False
    similarity_threshold: float = 0.95

    def __post_init__(self) -> None:
        if self.enable_similarity and not 0.0 < self.similarity_threshold < 1.0:
            raise ValueError("similarity_threshold must be in (0, 1) when similarity grouping is enabled.")

    def build(
        self,
        *,
        anchor_observations: list[Any] | np.ndarray,
        rollout_groups: list[Any] | np.ndarray,
        active_mask: list[bool] | np.ndarray | None = None,
    ) -> StepGroupResult:
        anchors = list(anchor_observations)
        groups = list(rollout_groups)
        if len(anchors) != len(groups):
            raise ValueError("anchor_observations and rollout_groups must have the same length.")
        active = [True] * len(anchors) if active_mask is None else [bool(value) for value in active_mask]
        if len(active) != len(anchors):
            raise ValueError("active_mask must have the same length as anchor_observations.")

        group_ids = np.empty(len(anchors), dtype=object)
        group_sizes: dict[str, int] = {}
        rows_by_rollout: dict[str, list[int]] = {}
        for row, (rollout_group, is_active) in enumerate(zip(groups, active, strict=True)):
            if not is_active:
                group_ids[row] = f"__inactive__:{row}"
                continue
            rows_by_rollout.setdefault(_canonical_text(rollout_group), []).append(row)

        for rollout_group, rows in rows_by_rollout.items():
            clusters: list[dict[str, Any]] = []
            for row in rows:
                anchor = _canonical_text(anchors[row])
                cluster_index = self._matching_cluster(anchor, clusters)
                if cluster_index is None:
                    clusters.append({"representative": anchor, "rows": [row]})
                else:
                    clusters[cluster_index]["rows"].append(row)

            for cluster_index, cluster in enumerate(clusters):
                group_id = _stable_group_id(rollout_group, cluster_index, cluster["representative"])
                cluster_rows = cluster["rows"]
                group_sizes[group_id] = len(cluster_rows)
                for row in cluster_rows:
                    group_ids[row] = group_id

        return StepGroupResult(group_ids=group_ids, group_sizes=group_sizes)

    def _matching_cluster(self, anchor: str, clusters: list[dict[str, Any]]) -> int | None:
        for index, cluster in enumerate(clusters):
            representative = str(cluster["representative"])
            if anchor == representative:
                return index
            similarity = SequenceMatcher(None, anchor, representative).ratio()
            if self.enable_similarity and similarity >= self.similarity_threshold:
                return index
        return None


def _canonical_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    except TypeError:
        return str(value)


def _stable_group_id(rollout_group: str, cluster_index: int, anchor: str) -> str:
    digest = hashlib.sha256(f"{rollout_group}\0{anchor}".encode()).hexdigest()[:16]
    return f"step:{cluster_index}:{digest}"
