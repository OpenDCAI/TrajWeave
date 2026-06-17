from __future__ import annotations

from dataclasses import dataclass

from trajweave.backends.policy import PolicyBackend, PolicyRequest
from trajweave.core.specs import TeamSpec
from trajweave.core.trajectory import AgentTurn, MultiAgentTrajectory
from trajweave.envs.math import MathTask
from trajweave.orchestration.base import TeamContext


@dataclass
class SolverVerifierOrchestra:
    solver_name: str = "solver"
    verifier_name: str = "verifier"
    approval_keyword: str = "APPROVED"

    def run(
        self,
        *,
        episode_id: str,
        rollout_group: str,
        task: MathTask,
        team: TeamSpec,
        observation: str,
        policy_backend: PolicyBackend,
    ) -> MultiAgentTrajectory:
        trajectory = MultiAgentTrajectory(
            episode_id=episode_id,
            task_id=task.task_id,
            rollout_group=rollout_group,
            team_name=team.name,
        )
        context = TeamContext()
        final_answer = ""
        turn_id = 0

        for loop_index in range(team.max_turns):
            solver = team.agent(self.solver_name)
            solver_prompt = self._build_prompt(solver.role, observation, context)
            solver_response = policy_backend.generate(
                PolicyRequest(
                    agent=solver,
                    task_id=task.task_id,
                    observation=observation,
                    prompt=solver_prompt,
                    team_context=context.render(),
                    metadata={"answer": task.answer, "loop_index": loop_index},
                )
            )
            final_answer = solver_response.text
            context.append(solver.name, solver_response.text)
            trajectory.add_turn(
                AgentTurn(
                    episode_id=episode_id,
                    task_id=task.task_id,
                    turn_id=turn_id,
                    agent_name=solver.name,
                    role=solver.role,
                    policy_group=solver.policy_group,
                    observation=observation,
                    prompt=solver_prompt,
                    action_text=solver_response.text,
                    action_token_ids=solver_response.token_ids,
                    action_logprobs=solver_response.logprobs,
                    metadata=solver_response.metadata | {"loop_index": loop_index},
                )
            )
            turn_id += 1

            if loop_index == team.max_turns - 1:
                break

            verifier = team.agent(self.verifier_name)
            verifier_prompt = self._build_prompt(verifier.role, observation, context)
            verifier_response = policy_backend.generate(
                PolicyRequest(
                    agent=verifier,
                    task_id=task.task_id,
                    observation=observation,
                    prompt=verifier_prompt,
                    team_context=context.render(),
                    metadata={"answer": task.answer, "loop_index": loop_index},
                )
            )
            context.append(verifier.name, verifier_response.text)
            approved = self.approval_keyword in verifier_response.text.upper()
            trajectory.add_turn(
                AgentTurn(
                    episode_id=episode_id,
                    task_id=task.task_id,
                    turn_id=turn_id,
                    agent_name=verifier.name,
                    role=verifier.role,
                    policy_group=verifier.policy_group,
                    observation=observation,
                    prompt=verifier_prompt,
                    action_text=verifier_response.text,
                    action_token_ids=verifier_response.token_ids,
                    action_logprobs=verifier_response.logprobs,
                    done=approved,
                    metadata=verifier_response.metadata | {"approved": approved, "loop_index": loop_index},
                )
            )
            turn_id += 1
            if approved:
                break

        trajectory.final_answer = final_answer
        trajectory.metadata["team_context"] = context.render()
        return trajectory

    def _build_prompt(self, role: str, observation: str, context: TeamContext) -> str:
        rendered = context.render()
        if role.lower() == "solver":
            return f"Task:\n{observation}\n\nTeam context:\n{rendered}\n\nSolve the task and return 'Final answer: <number>'."
        if role.lower() == "verifier":
            return f"Task:\n{observation}\n\nTeam context:\n{rendered}\n\nVerify the latest solver answer. Return APPROVED or REVISE."
        return f"Task:\n{observation}\n\nTeam context:\n{rendered}"
