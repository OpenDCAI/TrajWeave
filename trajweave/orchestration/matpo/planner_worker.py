from __future__ import annotations

import re
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
        if environment is None or not hasattr(environment, "execute_tool"):
            raise ValueError("PlannerWorkerOrchestra requires an environment with an execute_tool() method.")

        trajectory = MultiAgentTrajectory(
            episode_id=episode_id,
            task_id=task.task_id,
            rollout_group=rollout_group,
            team_name=team.name,
            metadata={"coordination_protocol": "planner_worker_agent_tool"},
        )
        planner = team.agent(self.planner_name)
        worker = team.agent(self.worker_name)
        if planner.policy_group != worker.policy_group:
            raise ValueError("MATPO planner and worker must share one policy group.")
        if not planner.trainable or not worker.trainable:
            raise ValueError("MATPO planner and worker must both be trainable.")
        if self.worker_name not in planner.tools:
            raise ValueError("MATPO planner must expose the worker agent as its delegation tool.")
        if self.tool_name not in worker.tools:
            raise ValueError("MATPO worker must expose the configured offline search tool.")
        turn_id = 0
        history: list[str] = []
        worker_summaries: list[str] = []
        last_sub_goal = ""
        final_response = None
        final_reqs_id = ""

        for loop_index in range(max(1, int(team.max_turns))):
            planner_reqs_id = f"{episode_id}:planner:{loop_index}"
            planner_prompt = self._planner_prompt(observation, history)
            planner_response = policy_backend.generate(
                PolicyRequest(
                    agent=planner,
                    task_id=task.task_id,
                    observation=observation,
                    prompt=planner_prompt,
                    team_context=self._history_text(history),
                    metadata={
                        "answer": task.answer,
                        "search_query": task.search_query or observation,
                        "stage": "delegate" if loop_index == 0 else "final",
                        "worker_agent": self.worker_name,
                        "tool_name": self.tool_name,
                        "loop_index": loop_index,
                    },
                )
            )
            sub_goal, planner_call_count, planner_call_valid = self._parse_call(
                planner_response.text, expected_target=self.worker_name
            )
            planner_final_valid = planner_call_count == 0 and self._is_final_answer(planner_response.text)
            if planner_call_valid:
                planner_role = "delegate"
            elif planner_final_valid:
                planner_role = "final"
            else:
                planner_role = "invalid_planner_call"
            planner_format_valid = planner_call_valid or planner_final_valid
            trajectory.add_turn(
                AgentTurn(
                    episode_id=episode_id,
                    task_id=task.task_id,
                    turn_id=turn_id,
                    agent_name=planner.name,
                    role=planner.role,
                    policy_group=planner.policy_group,
                    observation=observation,
                    prompt=planner_prompt,
                    action_text=planner_response.text,
                    action_token_ids=planner_response.token_ids,
                    action_logprobs=planner_response.logprobs,
                    done=planner_role != "delegate",
                    metadata=planner_response.metadata
                    | {
                        "reqs_id": planner_reqs_id,
                        "parent_reqs_id": "",
                        "is_from_subagent_tool": False,
                        "turn_count": turn_id,
                        "agent_type": "main_agent",
                        "role_id": "planner",
                        "shared_model_id": "shared",
                        "tool_name": self.tool_name,
                        "call_target": self.worker_name,
                        "sub_goal": sub_goal or last_sub_goal,
                        "tool_result": worker_summaries[-1] if worker_summaries else "",
                        "matpo_tool_call_count": planner_call_count,
                        "matpo_tool_format_valid": planner_format_valid,
                        "matpo_turn_role": planner_role,
                    },
                )
            )
            turn_id += 1
            history.append(f"Planner: {planner_response.text}")

            if planner_role != "delegate":
                final_response = planner_response
                final_reqs_id = planner_reqs_id
                break

            last_sub_goal = sub_goal
            worker_call_prompt = self._worker_call_prompt(observation, sub_goal)
            worker_call_reqs_id = f"{episode_id}:worker_call:{loop_index}"
            worker_call_response = policy_backend.generate(
                PolicyRequest(
                    agent=worker,
                    task_id=task.task_id,
                    observation=sub_goal,
                    prompt=worker_call_prompt,
                    team_context=self._history_text(history),
                    metadata={
                        "answer": task.answer,
                        "search_query": sub_goal,
                        "stage": "worker_call",
                        "tool_name": self.tool_name,
                        "parent_reqs_id": planner_reqs_id,
                    },
                )
            )
            tool_request, worker_call_count, worker_format_valid = self._parse_call(
                worker_call_response.text, expected_target=self.tool_name
            )
            worker_call_role = "worker_call" if worker_format_valid else "invalid_worker_call"
            trajectory.add_turn(
                AgentTurn(
                    episode_id=episode_id,
                    task_id=task.task_id,
                    turn_id=turn_id,
                    agent_name=worker.name,
                    role=worker.role,
                    policy_group=worker.policy_group,
                    observation=sub_goal,
                    prompt=worker_call_prompt,
                    action_text=worker_call_response.text,
                    action_token_ids=worker_call_response.token_ids,
                    action_logprobs=worker_call_response.logprobs,
                    metadata=worker_call_response.metadata
                    | {
                        "reqs_id": worker_call_reqs_id,
                        "parent_reqs_id": planner_reqs_id,
                        "is_from_subagent_tool": True,
                        "turn_count": turn_id,
                        "agent_type": worker.name,
                        "role_id": "worker",
                        "shared_model_id": "shared",
                        "tool_name": self.tool_name,
                        "sub_goal": sub_goal,
                        "tool_request": tool_request,
                        "tool_observation": "",
                        "tool_result": "",
                        "matpo_tool_call_count": worker_call_count,
                        "matpo_tool_format_valid": worker_format_valid,
                        "matpo_turn_role": worker_call_role,
                    },
                )
            )
            turn_id += 1
            history.append(f"Worker tool call: {worker_call_response.text}")

            if not worker_format_valid:
                continue

            evidence = environment.execute_tool(self.tool_name, task, tool_request)
            history.append(f"Offline tool observation: {evidence}")
            worker_summary_prompt = self._worker_summary_prompt(observation, sub_goal, evidence, history)
            worker_summary_reqs_id = f"{episode_id}:worker_summary:{loop_index}"
            worker_summary_response = policy_backend.generate(
                PolicyRequest(
                    agent=worker,
                    task_id=task.task_id,
                    observation=evidence,
                    prompt=worker_summary_prompt,
                    team_context=self._history_text(history),
                    metadata={
                        "answer": task.answer,
                        "evidence": evidence,
                        "stage": "worker_summary",
                        "tool_name": self.tool_name,
                        "parent_reqs_id": planner_reqs_id,
                    },
                )
            )
            trajectory.add_turn(
                AgentTurn(
                    episode_id=episode_id,
                    task_id=task.task_id,
                    turn_id=turn_id,
                    agent_name=worker.name,
                    role=worker.role,
                    policy_group=worker.policy_group,
                    observation=evidence,
                    prompt=worker_summary_prompt,
                    action_text=worker_summary_response.text,
                    action_token_ids=worker_summary_response.token_ids,
                    action_logprobs=worker_summary_response.logprobs,
                    metadata=worker_summary_response.metadata
                    | {
                        "reqs_id": worker_summary_reqs_id,
                        "parent_reqs_id": planner_reqs_id,
                        "is_from_subagent_tool": True,
                        "turn_count": turn_id,
                        "agent_type": worker.name,
                        "role_id": "worker",
                        "shared_model_id": "shared",
                        "tool_name": self.tool_name,
                        "sub_goal": sub_goal,
                        "tool_request": tool_request,
                        "tool_observation": evidence,
                        "tool_result": worker_summary_response.text,
                        "matpo_tool_call_count": 0,
                        "matpo_tool_format_valid": True,
                        "matpo_turn_role": "worker_summary",
                    },
                )
            )
            turn_id += 1
            worker_summaries.append(worker_summary_response.text)
            history.append(f"Worker summary: {worker_summary_response.text}")

        if final_response is None:
            final_reqs_id = f"{episode_id}:planner:final"
            forced_prompt = self._forced_final_prompt(observation, history)
            final_response = policy_backend.generate(
                PolicyRequest(
                    agent=planner,
                    task_id=task.task_id,
                    observation=observation,
                    prompt=forced_prompt,
                    team_context=self._history_text(history),
                    metadata={"answer": task.answer, "stage": "final", "forced": True},
                )
            )
            _, forced_call_count, _ = self._parse_call(final_response.text, expected_target=self.worker_name)
            forced_final_valid = forced_call_count == 0 and self._is_final_answer(final_response.text)
            forced_role = "final" if forced_final_valid else "invalid_planner_call"
            trajectory.add_turn(
                AgentTurn(
                    episode_id=episode_id,
                    task_id=task.task_id,
                    turn_id=turn_id,
                    agent_name=planner.name,
                    role=planner.role,
                    policy_group=planner.policy_group,
                    observation=observation,
                    prompt=forced_prompt,
                    action_text=final_response.text,
                    action_token_ids=final_response.token_ids,
                    action_logprobs=final_response.logprobs,
                    done=True,
                    metadata=final_response.metadata
                    | {
                        "reqs_id": final_reqs_id,
                        "parent_reqs_id": "",
                        "is_from_subagent_tool": False,
                        "turn_count": turn_id,
                        "agent_type": "main_agent",
                        "role_id": "planner",
                        "shared_model_id": "shared",
                        "tool_name": self.tool_name,
                        "call_target": self.worker_name,
                        "sub_goal": last_sub_goal,
                        "tool_result": worker_summaries[-1] if worker_summaries else "",
                        "matpo_tool_call_count": forced_call_count,
                        "matpo_tool_format_valid": forced_final_valid,
                        "matpo_turn_role": forced_role,
                    },
                )
            )

        trajectory.final_answer = final_response.text
        trajectory.metadata["worker_result"] = worker_summaries[-1] if worker_summaries else ""
        trajectory.metadata["planner_history"] = tuple(history)
        trajectory.metadata["planner_req_id"] = final_reqs_id
        return trajectory

    def _planner_prompt(self, question: str, history: list[str]) -> str:
        history_text = self._history_text(history)
        if not history:
            return (
                "You are a MATPO planner. Decide whether to delegate research.\n"
                f"Question: {question}\n"
                f"To delegate, return exactly 'CALL {self.worker_name}: <sub-goal>'.\n"
                "Otherwise return 'Final answer: <answer>'."
            )
        return (
            "You are a MATPO planner. Use the complete interaction history and decide the next step.\n"
            f"Question: {question}\n"
            f"Interaction history:\n{history_text}\n"
            "If you have enough evidence, return 'Final answer: <answer>'.\n"
            f"Otherwise, return exactly 'CALL {self.worker_name}: <sub-goal>'."
        )

    def _forced_final_prompt(self, question: str, history: list[str]) -> str:
        return (
            "You are a MATPO planner. No delegation rounds remain.\n"
            f"Question: {question}\n"
            f"Interaction history:\n{self._history_text(history)}\n"
            "You must return 'Final answer: <answer>' now."
        )

    def _worker_call_prompt(self, question: str, subtask: str) -> str:
        return (
            "You are the MATPO browsing worker. Choose one offline search request.\n"
            f"Main question: {question}\n"
            f"Delegated subtask: {subtask}\n"
            f"Return exactly 'CALL {self.tool_name}: <query>'."
        )

    def _worker_summary_prompt(self, question: str, subtask: str, evidence: str, history: list[str]) -> str:
        return (
            "You are the MATPO browsing worker. Summarize the offline tool evidence.\n"
            f"Main question: {question}\n"
            f"Delegated subtask: {subtask}\n"
            f"Offline evidence: {evidence}\n"
            f"Interaction history:\n{self._history_text(history)}\n"
            "Return a concise evidence summary for the planner."
        )

    def _parse_call(self, text: str, *, expected_target: str) -> tuple[str, int, bool]:
        matches = re.findall(r"CALL\s+([^\s:]+)\s*:\s*([^\r\n]*)", text, flags=re.IGNORECASE)
        call_count = len(re.findall(r"\bCALL\s+[^\s:]+\s*:", text, flags=re.IGNORECASE))
        full_match = re.fullmatch(r"\s*CALL\s+([^\s:]+)\s*:\s*(\S(?:.*\S)?)\s*", text, flags=re.IGNORECASE)
        if not matches:
            return "", 0, False
        first_target, first_request = matches[0]
        request = first_request.strip()
        contains_nested_call = re.search(r"\bCALL\s+[^\s:]+\s*:", request, flags=re.IGNORECASE) is not None
        valid = (
            call_count == 1
            and full_match is not None
            and bool(request)
            and not contains_nested_call
            and _normalize_tool_token(first_target) == _normalize_tool_token(expected_target)
        )
        return request, call_count, valid

    @staticmethod
    def _is_final_answer(text: str) -> bool:
        return re.fullmatch(r"\s*Final answer:\s*\S(?:.*\S)?\s*", text, flags=re.IGNORECASE) is not None

    @staticmethod
    def _history_text(history: list[str]) -> str:
        return "\n".join(history)


def _normalize_tool_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
