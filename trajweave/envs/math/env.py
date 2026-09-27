from __future__ import annotations

from dataclasses import dataclass

from trajweave.envs.math.c3 import C3MathEnvironment, C3MathTask


@dataclass(frozen=True)
class MathTask:
    task_id: str
    question: str
    answer: int | str


@dataclass(frozen=True)
class SolverVerifierMathEnvironment:
    use_math_verify: bool = True
    name: str = "solver_verifier_math"

    def initial_observation(self, task: MathTask) -> str:
        return task.question

    def evaluate(self, task: MathTask, final_answer: str) -> tuple[float, bool]:
        # Keep the dataset label intact and use the shared math judge so
        # fractions, decimals, LaTeX, and self-corrected answers are handled
        # consistently. C3MathEnvironment falls back deterministically when
        # the optional math-verify dependency is unavailable.
        return C3MathEnvironment(use_math_verify=self.use_math_verify).evaluate(
            C3MathTask(task_id=task.task_id, question=task.question, answer=str(task.answer)),
            final_answer,
        )
