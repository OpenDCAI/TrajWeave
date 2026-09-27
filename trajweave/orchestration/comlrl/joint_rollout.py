from __future__ import annotations

from itertools import islice, product
from typing import Sequence


def canonical_joint_mode(mode: str) -> str:
    if not isinstance(mode, str):
        raise TypeError("joint mode must be a string")
    normalized = mode.lower()
    if normalized in {"align", "aligned"}:
        return "aligned"
    if normalized in {"cross", "crossed"}:
        return "cross"
    raise ValueError(f"unsupported joint mode: {mode!r}")


def compose_joint_action_indices(
    agent_names: Sequence[str],
    num_candidates: int,
    mode: str,
    *,
    limit: int | None = None,
) -> list[tuple[int, ...]]:
    names = _validate_agent_names(agent_names)
    if isinstance(num_candidates, bool) or not isinstance(num_candidates, int):
        raise TypeError("num_candidates must be an integer")
    if num_candidates < 1:
        raise ValueError("num_candidates must be positive")

    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int)):
        raise TypeError("limit must be an integer or None")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")

    canonical_mode = canonical_joint_mode(mode)
    if canonical_mode == "aligned":
        combinations = (tuple(candidate_index for _ in names) for candidate_index in range(num_candidates))
    else:
        combinations = product(range(num_candidates), repeat=len(names))
    return list(combinations if limit is None else islice(combinations, limit))


def _validate_agent_names(agent_names: Sequence[str]) -> tuple[str, ...]:
    if isinstance(agent_names, (str, bytes)):
        raise TypeError("agent_names must be a sequence of names, not a string")
    names = tuple(agent_names)
    if not names:
        raise ValueError("agent_names must not be empty")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("agent_names must contain non-empty strings")
    if len(names) != len(set(names)):
        raise ValueError("agent_names must be unique")
    return names
