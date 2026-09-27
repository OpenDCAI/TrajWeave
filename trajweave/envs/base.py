from __future__ import annotations

from typing import Any, Protocol

from trajweave.core.trajectory import MultiAgentTrajectory


class Environment(Protocol):
    name: str

    def initial_observation(self, task: Any) -> str: ...

    def evaluate(self, task: Any, final_answer: str) -> tuple[float, bool]: ...


def evaluate_trajectory(
    environment: Environment,
    task: Any,
    trajectory: MultiAgentTrajectory,
) -> tuple[float, bool]:
    evaluator = getattr(environment, "evaluate_trajectory", None)
    if callable(evaluator):
        return evaluator(task, trajectory)
    return environment.evaluate(task, trajectory.final_answer)
