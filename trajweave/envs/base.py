from __future__ import annotations

from typing import Any, Protocol


class Environment(Protocol):
    name: str

    def initial_observation(self, task: Any) -> str:
        ...

    def evaluate(self, task: Any, final_answer: str) -> tuple[float, bool]:
        ...
