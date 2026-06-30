from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from trajweave.backends.policy import PolicyBackend
    from trajweave.core.specs import TeamSpec
    from trajweave.core.trajectory import MultiAgentTrajectory


@dataclass
class TeamContext:
    entries: list[tuple[str, str]] = field(default_factory=list)

    def append(self, agent_name: str, text: str) -> None:
        self.entries.append((agent_name, text))

    def render(self) -> str:
        return "\n".join(f'The output of "{agent_name}": {text}' for agent_name, text in self.entries)


class Orchestra(Protocol):
    def run(
        self,
        *,
        episode_id: str,
        rollout_group: str,
        task: Any,
        team: TeamSpec,
        observation: str,
        policy_backend: PolicyBackend,
        environment: Any | None = None,
    ) -> MultiAgentTrajectory:
        ...
