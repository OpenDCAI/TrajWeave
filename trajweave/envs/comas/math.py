from __future__ import annotations

from trajweave.core.trajectory import MultiAgentTrajectory
from trajweave.envs.math import MathTask, SolverVerifierMathEnvironment


class CoMASMathEnvironment(SolverVerifierMathEnvironment):
    """CoMAS 数学评测环境；正确率只用于评测，不参与交互奖励。"""

    name = "comas_math_evaluation"

    def evaluate_trajectory(self, task: MathTask, trajectory: MultiAgentTrajectory) -> tuple[float, bool]:
        solutions = list(trajectory.metadata.get("comas_final_solutions", []))
        if not solutions:
            solutions = [trajectory.final_answer]
        correctness = [self.evaluate(task, str(solution))[1] for solution in solutions]
        accuracy = sum(bool(value) for value in correctness) / len(correctness)
        trajectory.metadata["comas_final_correctness"] = correctness
        trajectory.metadata["comas_final_accuracy"] = accuracy
        return float(accuracy), bool(accuracy == 1.0)
