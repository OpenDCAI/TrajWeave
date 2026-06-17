from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TeamContext:
    entries: list[tuple[str, str]] = field(default_factory=list)

    def append(self, agent_name: str, text: str) -> None:
        self.entries.append((agent_name, text))

    def render(self) -> str:
        return "\n".join(f'The output of "{agent_name}": {text}' for agent_name, text in self.entries)
