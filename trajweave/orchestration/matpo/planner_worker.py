from __future__ import annotations

from dataclasses import dataclass

from trajweave.backends.policy import PolicyBackend, PolicyRequest
from trajweave.core.specs import TeamSpec
from trajweave.core.trajectory import AgentTurn, MultiAgentTrajectory
from trajweave.envs.search import SearchAnswerEnvironment, SearchTask


@dataclass
class PlannerWorkerOrchestra:
    planner_name: str = "planner"
    worker_name: str = "browsing_agent"
    tool_name: str = "search_and_browse"

    def run(
        self,
        *,
        episode_id: str,
        rollout_group: str,
        task: SearchTask,
        team: TeamSpec,
        observation: str,
        policy_backend: PolicyBackend,
        environment: SearchAnswerEnvironment | None = None,
    ) -> MultiAgentTrajectory:
        if environment is None or not hasattr(environment, "search"):
            raise ValueError("PlannerWorkerOrchestra requires an environment with a search() method.")

        trajectory = MultiAgentTrajectory(
            episode_id=episode_id,
            task_id=task.task_id,
            rollout_group=rollout_group,
            team_name=team.name,
            metadata={"coordination_protocol": "planner_worker_agent_tool"},
        )
        planner = team.agent(self.planner_name)
        worker = team.agent(self.worker_name)
        planner_req_id = f"{episode_id}:planner:0"
        worker_req_id = f"{episode_id}:worker:0"

        planner_prompt = self._planner_prompt(observation, worker_result=None)
        planner_response = policy_backend.generate(
            PolicyRequest(
                agent=planner,
                task_id=task.task_id,
                observation=observation,
                prompt=planner_prompt,
                metadata={
                    "answer": task.answer,
                    "search_query": task.search_query or observation,
                    "stage": "delegate",
                    "tool_name": self.tool_name,
                },
            )
        )
        trajectory.add_turn(
            AgentTurn(
                episode_id=episode_id,
                task_id=task.task_id,
                turn_id=0,
                agent_name=planner.name,
                role=planner.role,
                policy_group=planner.policy_group,
                observation=observation,
                prompt=planner_prompt,
                action_text=planner_response.text,
                action_token_ids=planner_response.token_ids,
                action_logprobs=planner_response.logprobs,
                metadata=planner_response.metadata
                | {
                    "reqs_id": planner_req_id,
                    "parent_reqs_id": "",
                    "is_from_subagent_tool": False,
                    "turn_count": 0,
                    "agent_type": "main_agent",
                    "role_id": "planner",
                    "shared_model_id": "shared",
                    "tool_name": self.tool_name,
                    "sub_goal": task.search_query or observation,
                    "tool_result": "",
                    "matpo_tool_call_count": 1,
                    "matpo_tool_format_valid": True,
                },
            )
        )

        worker_observation = task.search_query or observation
        evidence = environment.search(task, worker_observation)
        worker_prompt = self._worker_prompt(observation, worker_observation, evidence)
        worker_response = policy_backend.generate(
            PolicyRequest(
                agent=worker,
                task_id=task.task_id,
                observation=worker_observation,
                prompt=worker_prompt,
                team_context=evidence,
                metadata={"answer": task.answer, "evidence": evidence, "stage": "worker"},
            )
        )
        trajectory.add_turn(
            AgentTurn(
                episode_id=episode_id,
                task_id=task.task_id,
                turn_id=1,
                agent_name=worker.name,
                role=worker.role,
                policy_group=worker.policy_group,
                observation=worker_observation,
                prompt=worker_prompt,
                action_text=worker_response.text,
                action_token_ids=worker_response.token_ids,
                action_logprobs=worker_response.logprobs,
                metadata=worker_response.metadata
                | {
                    "reqs_id": worker_req_id,
                    "parent_reqs_id": planner_req_id,
                    "is_from_subagent_tool": True,
                    "turn_count": 1,
                    "agent_type": worker.name,
                    "role_id": "worker",
                    "shared_model_id": "shared",
                    "tool_name": self.tool_name,
                    "tool_result": worker_response.text,
                    "sub_goal": worker_observation,
                    "matpo_tool_call_count": 0,
                    "matpo_tool_format_valid": True,
                },
            )
        )

        final_prompt = self._planner_prompt(observation, worker_result=worker_response.text)
        final_response = policy_backend.generate(
            PolicyRequest(
                agent=planner,
                task_id=task.task_id,
                observation=observation,
                prompt=final_prompt,
                team_context=worker_response.text,
                metadata={"answer": task.answer, "stage": "final"},
            )
        )
        trajectory.add_turn(
            AgentTurn(
                episode_id=episode_id,
                task_id=task.task_id,
                turn_id=2,
                agent_name=planner.name,
                role=planner.role,
                policy_group=planner.policy_group,
                observation=observation,
                prompt=final_prompt,
                action_text=final_response.text,
                action_token_ids=final_response.token_ids,
                action_logprobs=final_response.logprobs,
                done=True,
                metadata=final_response.metadata
                | {
                    "reqs_id": planner_req_id,
                    "parent_reqs_id": "",
                    "is_from_subagent_tool": False,
                    "turn_count": 2,
                    "agent_type": "main_agent",
                    "role_id": "planner",
                    "shared_model_id": "shared",
                    "tool_name": self.tool_name,
                    "sub_goal": task.search_query or observation,
                    "tool_result": worker_response.text,
                    "matpo_tool_call_count": 1,
                    "matpo_tool_format_valid": True,
                },
            )
        )
        trajectory.final_answer = final_response.text
        trajectory.metadata["worker_result"] = worker_response.text
        trajectory.metadata["planner_req_id"] = planner_req_id
        return trajectory

    def _planner_prompt(self, question: str, worker_result: str | None) -> str:
        if worker_result is None:
            return (
                "You are a MATPO planner. Decide whether to call search_and_browse.\n"
                f"Question: {question}\n"
                "Return a focused subtask for the browsing agent."
            )
        return (
            "You are a MATPO planner. Use the browsing agent result and answer.\n"
            f"Question: {question}\n"
            f"Browsing agent result: {worker_result}\n"
            "Return 'Final answer: <answer>'."
        )

    def _worker_prompt(self, question: str, subtask: str, evidence: str) -> str:
        return (
            "You are the browsing agent. Complete the delegated factual subtask.\n"
            f"Main question: {question}\n"
            f"Subtask: {subtask}\n"
            f"Offline evidence: {evidence}\n"
            "Return a concise evidence summary."
        )
