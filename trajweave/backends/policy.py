from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from trajweave.core.specs import AgentSpec


@dataclass(frozen=True)
class PolicyRequest:
    agent: AgentSpec
    task_id: str
    observation: str
    prompt: str
    team_context: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class PolicyResponse:
    text: str
    token_ids: list[int]
    logprobs: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class PolicyBackend(Protocol):
    def generate(self, request: PolicyRequest) -> PolicyResponse:
        ...


class StableByteTokenizer:
    pad_token_id = 0

    def encode(self, text: str) -> list[int]:
        values = list(text.encode("utf-8"))
        return [value + 1 for value in values] or [1]
