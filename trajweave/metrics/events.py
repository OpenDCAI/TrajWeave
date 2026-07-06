from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MetricEvent:
    run_id: str
    name: str
    value: float | int | str | bool | None
    step: int | None = None
    source: str = "trajweave"
    split: str | None = None
    unit: str = "scalar"
    tags: dict[str, Any] = field(default_factory=dict)
