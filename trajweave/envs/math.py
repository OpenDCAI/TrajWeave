from __future__ import annotations

from dataclasses import dataclass

from trajweave.backends.local import extract_final_int


@dataclass(frozen=True)
class MathTask:
    task_id: str
    question: str
    answer: int


class SolverVerifierMathEnvironment:
    name = "solver_verifier_math"

    def initial_observation(self, task: MathTask) -> str:
        return task.question

    def evaluate(self, task: MathTask, final_answer: str) -> tuple[float, bool]:
        predicted = extract_final_int(final_answer)
        success = predicted == task.answer
        return (1.0 if success else 0.0), success
