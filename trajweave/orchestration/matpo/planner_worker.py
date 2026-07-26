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
        turn_id = 0
        worker_history: list[str] = []
        last_sub_goal = ""
        final_response = None
        final_reqs_id = ""

        max_turns = max(1, int(team.max_turns))
        for loop_index in range(max_turns):
            planner_reqs_id = f"{episode_id}:planner:{loop_index}"
            if loop_index == 0:
                planner_prompt = self._planner_prompt(observation, worker_result=None)
                planner_metadata = {
                    "answer": task.answer,
                    "search_query": task.search_query or observation,
                    "stage": "delegate",
                    "tool_name": self.tool_name,
                }
            else:
                planner_prompt = self._planner_prompt(observation, worker_result=worker_history[-1])
                planner_metadata = {
                    "answer": task.answer,
                    "stage": "final",
                    "tool_name": self.tool_name,
                    "loop_index": loop_index,
                }
            planner_response = policy_backend.generate(
                PolicyRequest(
                    agent=planner,
                    task_id=task.task_id,
                    observation=observation,
                    prompt=planner_prompt,
                    team_context=worker_history[-1] if worker_history else "",
                    metadata=planner_metadata,
                )
            )
            sub_goal, tool_call_count, tool_format_valid = self._parse_delegation(planner_response.text)

            if loop_index > 0 and tool_call_count == 0:
                # Planner explicitly stopped calling tools: treat this response as the
                # converged final answer instead of forcing another worker round.
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
                        done=True,
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
                            "sub_goal": last_sub_goal,
                            "tool_result": worker_history[-1],
                            "matpo_tool_call_count": tool_call_count,
                            "matpo_tool_format_valid": tool_format_valid,
                            "matpo_turn_role": "final",
                        },
                    )
                )
                turn_id += 1
                final_response = planner_response
                final_reqs_id = planner_reqs_id
                break

            # Round 0 always delegates to the worker (falling back to the dataset's
            # preset search_query when the planner's delegation could not be parsed).
            # Rounds >= 1 only reach here when the planner explicitly issued another
            # CALL directive.
            worker_observation = sub_goal or (task.search_query if loop_index == 0 else observation) or observation
            last_sub_goal = worker_observation
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
                        "sub_goal": worker_observation,
                        "tool_result": "",
                        "matpo_tool_call_count": tool_call_count,
                        "matpo_tool_format_valid": tool_format_valid,
                        "matpo_turn_role": "delegate",
                    },
                )
            )
            turn_id += 1

            evidence = environment.search(task, worker_observation)
            worker_prompt = self._worker_prompt(observation, worker_observation, evidence)
            worker_reqs_id = f"{episode_id}:worker:{loop_index}"
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
                    turn_id=turn_id,
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
                        "reqs_id": worker_reqs_id,
                        "parent_reqs_id": planner_reqs_id,
                        "is_from_subagent_tool": True,
                        "turn_count": turn_id,
                        "agent_type": worker.name,
                        "role_id": "worker",
                        "shared_model_id": "shared",
                        "tool_name": self.tool_name,
                        "tool_result": worker_response.text,
                        "sub_goal": worker_observation,
                        "matpo_tool_call_count": 0,
                        "matpo_tool_format_valid": True,
                        "matpo_turn_role": "worker",
                    },
                )
            )
            turn_id += 1
            worker_history.append(worker_response.text)

        if final_response is None:
            # The round budget (team.max_turns) was exhausted without the planner
            # explicitly converging. Force one last call that must produce an answer,
            # using a reqs_id distinct from the loop's numeric suffixes.
            final_reqs_id = f"{episode_id}:planner:final"
            forced_prompt = self._forced_final_prompt(observation, worker_history[-1])
            final_response = policy_backend.generate(
                PolicyRequest(
                    agent=planner,
                    task_id=task.task_id,
                    observation=observation,
                    prompt=forced_prompt,
                    team_context=worker_history[-1],
                    metadata={"answer": task.answer, "stage": "final", "forced": True},
                )
            )
            _, forced_tool_call_count, forced_tool_format_valid = self._parse_delegation(final_response.text)
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
                        "sub_goal": last_sub_goal,
                        "tool_result": worker_history[-1],
                        "matpo_tool_call_count": forced_tool_call_count,
                        "matpo_tool_format_valid": forced_tool_format_valid,
                        "matpo_turn_role": "final",
                    },
                )
            )
            turn_id += 1

        trajectory.final_answer = final_response.text
        trajectory.metadata["worker_result"] = worker_history[-1] if worker_history else ""
        trajectory.metadata["planner_req_id"] = final_reqs_id
        return trajectory

    def _planner_prompt(self, question: str, worker_result: str | None) -> str:
        if worker_result is None:
            return (
                "You are a MATPO planner. Decide whether to call search_and_browse.\n"
                f"Question: {question}\n"
                "Return a focused subtask for the browsing agent."
            )
        return (
            "You are a MATPO planner. Use the browsing agent result and decide the next step.\n"
            f"Question: {question}\n"
            f"Browsing agent result: {worker_result}\n"
            "If you have enough evidence, return 'Final answer: <answer>'.\n"
            "Otherwise, delegate another subtask with 'CALL <tool>: <sub-goal>'."
        )

    def _forced_final_prompt(self, question: str, worker_result: str) -> str:
        return (
            "You are a MATPO planner. No delegation rounds remain.\n"
            f"Question: {question}\n"
            f"Browsing agent result: {worker_result}\n"
            "You must return 'Final answer: <answer>' now."
        )

    def _worker_prompt(self, question: str, subtask: str, evidence: str) -> str:
        return (
            "You are the browsing agent. Complete the delegated factual subtask.\n"
            f"Main question: {question}\n"
            f"Subtask: {subtask}\n"
            f"Offline evidence: {evidence}\n"
            "Return a concise evidence summary."
        )

    def _parse_delegation(self, text: str) -> tuple[str, int, bool]:
        """Parse the planner's ``CALL <tool>: <sub-goal>`` delegation out of its response text.

        Returns the parsed sub-goal (empty string if none could be parsed), how many
        tool-call directives were found, and whether the first one names this
        orchestra's configured tool. A ``tool_call_count`` of 0 means the planner did
        not ask to call any tool in this response, which (outside of the mandatory
        first round) is treated as an implicit convergence signal.
        """
        matches = re.findall(r"CALL\s+([^\s:]+)\s*:\s*(.*?)(?=\n|$)", text, flags=re.IGNORECASE)
        if not matches:
            return "", 0, False
        first_tool, first_sub_goal = matches[0]
        tool_format_valid = _normalize_tool_token(first_tool) == _normalize_tool_token(self.tool_name)
        return first_sub_goal.strip(), len(matches), tool_format_valid


def _normalize_tool_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
