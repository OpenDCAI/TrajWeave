"""Pinned, intentionally small LiveCodeBench evaluation manifest."""
from __future__ import annotations

from dataclasses import dataclass

from trajweave.recipes.marti_mars2.eval import SearchEvalCase


LIVE_CODEBENCH_REVISION = "livecodebench-v5"
DEFAULT_LIVE_CODEBENCH_IDS: tuple[str, ...] = tuple(f"lcb-{index:04d}" for index in range(12))


@dataclass(frozen=True)
class LiveCodeBenchSubset:
    revision: str = LIVE_CODEBENCH_REVISION
    task_ids: tuple[str, ...] = DEFAULT_LIVE_CODEBENCH_IDS

    def cases(self, prompts: dict[str, str] | None = None) -> list[SearchEvalCase]:
        prompts = prompts or {}
        return [SearchEvalCase(task_id=task_id, prompt=prompts.get(task_id, task_id)) for task_id in self.task_ids]


def fixed_livecodebench_subset(*, revision: str = LIVE_CODEBENCH_REVISION, task_ids: tuple[str, ...] | None = None) -> LiveCodeBenchSubset:
    if revision != LIVE_CODEBENCH_REVISION:
        raise ValueError(f"Unsupported LiveCodeBench revision {revision!r}; expected {LIVE_CODEBENCH_REVISION!r}")
    ids = DEFAULT_LIVE_CODEBENCH_IDS if task_ids is None else tuple(str(item) for item in task_ids)
    if len(ids) != 12:
        raise ValueError("The pinned smoke subset must contain exactly 12 tasks")
    return LiveCodeBenchSubset(revision=revision, task_ids=ids)


__all__ = ["DEFAULT_LIVE_CODEBENCH_IDS", "LIVE_CODEBENCH_REVISION", "LiveCodeBenchSubset", "fixed_livecodebench_subset"]
