from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from trajweave.core.trajectory import AgentTurn, MultiAgentTrajectory
from trajweave.orchestration.solver_verifier import SolverVerifierOrchestra


@dataclass
class GiGPOSolverVerifierOrchestra(SolverVerifierOrchestra):
    """复用 Solver-Verifier 协议，并补齐 GiGPO 的 step transition。"""

    require_environment_success: bool = True

    def run(self, **kwargs: Any) -> MultiAgentTrajectory:
        trajectory = super().run(**kwargs)
        task_observation = str(kwargs["observation"])
        latest_feedback = ""
        latest_solver: AgentTurn | None = None

        for turn in trajectory.turns:
            if turn.agent_name == self.solver_name:
                turn.anchor_observation = {
                    "agent_id": self.solver_name,
                    "task": task_observation,
                    "verifier_feedback": latest_feedback,
                }
                latest_solver = turn
                continue
            if turn.agent_name == self.verifier_name:
                latest_feedback = turn.action_text
                if latest_solver is not None:
                    latest_solver.next_observation = {
                        "source": self.verifier_name,
                        "feedback": latest_feedback,
                        "done": bool(turn.done),
                    }

        if latest_solver is not None and latest_solver.next_observation is None:
            latest_solver.next_observation = {"source": "environment", "state": "terminal", "done": True}
        trajectory.metadata["step_transition_schema"] = "anchor_action_next_observation_v1"
        return trajectory
