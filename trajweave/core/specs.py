from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentSpec:
    name: str
    role: str
    policy_group: str
    trainable: bool = True
    model_path: str | None = None
    prompt_template: str | None = None
    tools: tuple[str, ...] = ()
    generation_config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyGroupSpec:
    name: str
    model_path: str | None = None
    trainable: bool = True
    backend: str = "local"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TeamSpec:
    name: str
    agents: tuple[AgentSpec, ...]
    policy_groups: tuple[PolicyGroupSpec, ...]
    orchestra: str
    reward: str
    credit: str
    max_turns: int = 3
    metadata: dict[str, Any] = field(default_factory=dict)

    def agent(self, name: str) -> AgentSpec:
        for agent in self.agents:
            if agent.name == name:
                return agent
        raise KeyError(f"Unknown agent: {name}")

    def trainable_agents(self) -> tuple[AgentSpec, ...]:
        return tuple(agent for agent in self.agents if agent.trainable)
