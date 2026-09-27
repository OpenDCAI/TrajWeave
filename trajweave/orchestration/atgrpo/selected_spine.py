from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from trajweave.backends.policy import PolicyBackend, PolicyRequest
from trajweave.core.specs import TeamSpec
from trajweave.core.trajectory import AgentTurn, MultiAgentTrajectory
from trajweave.envs.math import MathTask
from trajweave.orchestration.base import TeamContext, is_approved_response


@dataclass
class SelectedSpineSolverVerifierOrchestra:
    """AT-GRPO solver/verifier protocol with shared-observation sibling sampling."""

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
        environment: Any | None = None,
    ) -> MultiAgentTrajectory:
        return self.run_tree(
            episode_id=episode_id,
            rollout_group=rollout_group,
            task=task,
            team=team,
            observation=observation,
            policy_backend=policy_backend,
            environment=environment,
            branch_factor=1,
        )

    def run_tree(
        self,
        *,
        episode_id: str,
        rollout_group: str,
        task: MathTask,
        team: TeamSpec,
        observation: str,
        policy_backend: PolicyBackend,
        environment: Any,
        branch_factor: int,
    ) -> MultiAgentTrajectory:
        if branch_factor < 1:
            raise ValueError(f"AT-GRPO branch_factor must be positive, got {branch_factor}.")
        if environment is None or not hasattr(environment, "evaluate"):
            raise ValueError("AT-GRPO selected-spine sampling requires an environment with evaluate().")

        root_id = episode_id
        trajectory = MultiAgentTrajectory(
            episode_id=episode_id,
            task_id=task.task_id,
            rollout_group=rollout_group,
            team_name=team.name,
            metadata={"root_id": root_id, "sampling_mode": "selected_spine", "branch_factor": branch_factor},
        )
        context = TeamContext()
        parent_node_id: str | None = None
        final_answer = ""
        selected_solver_correct = False
        absolute_turn_id = 0

        for loop_index in range(team.max_turns):
            solver = team.agent(self.solver_name)
            solver_prompt = self._build_prompt(solver.role, observation, context)
            solver_group_id = self._observation_group_id(root_id, parent_node_id, absolute_turn_id, solver.name)
            solver_turns: list[AgentTurn] = []
            for branch_index in range(branch_factor):
                response = policy_backend.generate(
                    PolicyRequest(
                        agent=solver,
                        task_id=task.task_id,
                        observation=observation,
                        prompt=solver_prompt,
                        team_context=context.render(),
                        metadata={"answer": task.answer, "loop_index": loop_index, "branch_index": branch_index},
                    )
                )
                _, is_correct = environment.evaluate(task, response.text)
                solver_turns.append(
                    AgentTurn(
                        episode_id=episode_id,
                        task_id=task.task_id,
                        turn_id=absolute_turn_id,
                        agent_name=solver.name,
                        role=solver.role,
                        policy_group=solver.policy_group,
                        observation=observation,
                        prompt=solver_prompt,
                        action_text=response.text,
                        action_token_ids=response.token_ids,
                        action_logprobs=response.logprobs,
                        root_id=root_id,
                        node_id=self._node_id(root_id, absolute_turn_id, branch_index),
                        parent_node_id=parent_node_id,
                        observation_group_id=solver_group_id,
                        branch_index=branch_index,
                        local_score=float(bool(is_correct)),
                        metadata=response.metadata | {"loop_index": loop_index, "local_correct": bool(is_correct)},
                    )
                )
            selected_solver = self._select(solver_turns)
            selected_solver.selected_for_expansion = True
            trajectory.turns.extend(solver_turns)
            final_answer = selected_solver.action_text
            selected_solver_correct = bool(selected_solver.metadata["local_correct"])
            context.append(solver.name, selected_solver.action_text)
            parent_node_id = selected_solver.node_id
            absolute_turn_id += 1

            if loop_index == team.max_turns - 1:
                break

            verifier = team.agent(self.verifier_name)
            verifier_prompt = self._build_prompt(verifier.role, observation, context)
            verifier_group_id = self._observation_group_id(root_id, parent_node_id, absolute_turn_id, verifier.name)
            verifier_turns: list[AgentTurn] = []
            for branch_index in range(branch_factor):
                response = policy_backend.generate(
                    PolicyRequest(
                        agent=verifier,
                        task_id=task.task_id,
                        observation=observation,
                        prompt=verifier_prompt,
                        team_context=context.render(),
                        metadata={"answer": task.answer, "loop_index": loop_index, "branch_index": branch_index},
                    )
                )
                model_approved = is_approved_response(response.text, self.approval_keyword)
                local_correct = model_approved == selected_solver_correct
                verifier_turns.append(
                    AgentTurn(
                        episode_id=episode_id,
                        task_id=task.task_id,
                        turn_id=absolute_turn_id,
                        agent_name=verifier.name,
                        role=verifier.role,
                        policy_group=verifier.policy_group,
                        observation=observation,
                        prompt=verifier_prompt,
                        action_text=response.text,
                        action_token_ids=response.token_ids,
                        action_logprobs=response.logprobs,
                        done=model_approved,
                        root_id=root_id,
                        node_id=self._node_id(root_id, absolute_turn_id, branch_index),
                        parent_node_id=parent_node_id,
                        observation_group_id=verifier_group_id,
                        branch_index=branch_index,
                        local_score=1.0 if local_correct else -1.0,
                        metadata=response.metadata
                        | {
                            "loop_index": loop_index,
                            "model_approved": model_approved,
                            "approved": model_approved,
                            "local_solver_correct": selected_solver_correct,
                            "local_decision_correct": local_correct,
                        },
                    )
                )
            selected_verifier = self._select(verifier_turns)
            selected_verifier.selected_for_expansion = True
            trajectory.turns.extend(verifier_turns)
            context.append(verifier.name, selected_verifier.action_text)
            parent_node_id = selected_verifier.node_id
            absolute_turn_id += 1
            if selected_verifier.done:
                break

        trajectory.final_answer = final_answer
        trajectory.metadata["team_context"] = context.render()
        trajectory.metadata["selected_leaf_node_id"] = parent_node_id
        return trajectory

    @staticmethod
    def _select(turns: list[AgentTurn]) -> AgentTurn:
        return max(turns, key=lambda turn: (float(turn.local_score or 0.0), -int(turn.branch_index or 0)))

    @staticmethod
    def _node_id(root_id: str, turn_id: int, branch_index: int) -> str:
        return f"{root_id}:node:{turn_id}:{branch_index}"

    @staticmethod
    def _observation_group_id(root_id: str, parent_node_id: str | None, turn_id: int, agent_name: str) -> str:
        return f"{root_id}:obs:{parent_node_id or 'root'}:{turn_id}:{agent_name}"

    @staticmethod
    def _build_prompt(role: str, observation: str, context: TeamContext) -> str:
        rendered = context.render()
        if role.lower() == "solver":
            return (
                f"Task:\n{observation}\n\nTeam context:\n{rendered}\n\n"
                "Solve the task and return 'Final answer: <number>'."
            )
        if role.lower() == "verifier":
            return (
                f"Task:\n{observation}\n\nTeam context:\n{rendered}\n\n"
                "Verify the latest solver answer. Return APPROVED or REVISE."
            )
        return f"Task:\n{observation}\n\nTeam context:\n{rendered}"
