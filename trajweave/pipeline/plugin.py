from __future__ import annotations

from typing import Protocol

from trajweave.pipeline.context import RunContext


class RecipePlugin(Protocol):
    name: str

    def supports(self, context: RunContext) -> bool:
        ...

    def run(self, context: RunContext) -> dict:
        ...
