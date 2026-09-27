from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    source: str
    unit: str = "scalar"
    description: str = ""


class MetricRegistry:
    def __init__(self) -> None:
        self._definitions: dict[str, MetricDefinition] = {}

    def register(self, definition: MetricDefinition) -> None:
        self._definitions[definition.name] = definition

    def get(self, name: str) -> MetricDefinition | None:
        return self._definitions.get(name)

    def as_dict(self) -> dict[str, dict[str, str]]:
        return {
            name: {
                "source": definition.source,
                "unit": definition.unit,
                "description": definition.description,
            }
            for name, definition in sorted(self._definitions.items())
        }


def default_metric_registry() -> MetricRegistry:
    registry = MetricRegistry()
    for name, source in [
        ("success_rate", "rollout"),
        ("trajectories", "rollout"),
        ("samples", "rollout"),
        ("actor/pg_loss", "verl_actor"),
        ("actor/grad_norm", "verl_actor"),
        ("actor/entropy", "verl_actor"),
        ("critic/score/mean", "verl_critic"),
        ("critic/advantages/max", "verl_critic"),
        ("critic/advantages/min", "verl_critic"),
        ("training/num_turns/mean", "runtime"),
    ]:
        registry.register(MetricDefinition(name=name, source=source))
    return registry
