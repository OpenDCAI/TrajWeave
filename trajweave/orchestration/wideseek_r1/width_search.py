from __future__ import annotations

import re
from dataclasses import dataclass

from trajweave.backends.policy import PolicyBackend, PolicyRequest, PolicyResponse
from trajweave.core.specs import AgentSpec, TeamSpec
from trajweave.core.trajectory import AgentTurn, MultiAgentTrajectory
from trajweave.envs.search import SearchAnswerEnvironment, SearchTask


@dataclass
class WideSeekR1Orchestra:
    """Lead/subagent width-scaling workflow adapted from WideSeek-R1.

    Every subagent receives an isolated context.  ``parallel_wave`` records the
    logical parallel frontier even when a local policy backend executes calls
    synchronously.
    """

    lead_agent: str = "lead_agent"
    subagent_prefix: str = "subagent_"
    max_parallel_subagents: int = 3
    search_tool: str = "search"
    access_tool: str = "access"

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
            raise ValueError("WideSeekR1Orchestra requires an environment with execute_tool().")
        if self.max_parallel_subagents < 1:
            raise ValueError("WideSeek-R1 max_parallel_subagents must be at least 1.")

        lead = team.agent(self.lead_agent)
        workers = tuple(agent for agent in team.agents if agent.name != lead.name)
        self._validate_team(lead=lead, workers=workers)
        trajectory = MultiAgentTrajectory(
            episode_id=episode_id,
            task_id=task.task_id,
            rollout_group=rollout_group,
            team_name="wideseek_r1_broad_search",
            metadata={
                "coordination_protocol": "wideseek_r1_width_search",
                "shared_model": True,
                "max_parallel_subagents": self.max_parallel_subagents,
            },
        )

        plan_prompt = self._plan_prompt(observation)
        plan_response = policy_backend.generate(
            PolicyRequest(
                agent=lead,
                task_id=task.task_id,
                observation=observation,
                prompt=plan_prompt,
                team_context="",
                metadata={
                    "stage": "wideseek_plan",
                    "search_query": task.search_query or observation,
                    "max_parallel_subagents": self.max_parallel_subagents,
                    "answer": task.answer,
                },
            )
        )
        subtasks = _parse_subtasks(plan_response.text, limit=min(len(workers), self.max_parallel_subagents))
        plan_valid = bool(subtasks)
        if not subtasks:
            subtasks = (str(task.search_query or observation),)
        trajectory.add_turn(
            self._turn(
                trajectory=trajectory,
                turn_id=0,
                agent=lead,
                observation=observation,
                prompt=plan_prompt,
                response=plan_response,
                turn_role="lead_plan",
                agent_instance=lead.name,
                subtrajectory_id=0,
                format_valid=plan_valid,
                parallel_wave=0,
                sub_goal=" | ".join(subtasks),
            )
        )

        summaries: list[str] = []
        turn_id = 1
        for worker_index, (worker, subtask) in enumerate(zip(workers, subtasks, strict=False), start=1):
            # Each worker history begins here and never includes another worker's
            # prompts, tool observations, or summary.
            search_prompt = self._search_prompt(observation, subtask)
            search_response = policy_backend.generate(
                PolicyRequest(
                    agent=worker,
                    task_id=task.task_id,
                    observation=subtask,
                    prompt=search_prompt,
                    team_context="",
                    metadata={
                        "stage": "wideseek_search",
                        "search_query": subtask,
                        "tool_name": self.search_tool,
                        "worker_index": worker_index,
                        "answer": task.answer,
                    },
                )
            )
            search_query, search_valid = _parse_call(search_response.text, self.search_tool)
            search_query = search_query or subtask
            search_evidence = environment.execute_tool(self.search_tool, task, search_query)
            parent_turn_id = f"{episode_id}:{turn_id}"
            trajectory.add_turn(
                self._turn(
                    trajectory=trajectory,
                    turn_id=turn_id,
                    agent=worker,
                    observation=subtask,
                    prompt=search_prompt,
                    response=search_response,
                    turn_role="subagent_search",
                    agent_instance=worker.name,
                    subtrajectory_id=worker_index,
                    format_valid=search_valid,
                    parallel_wave=1,
                    sub_goal=subtask,
                    tool_name=self.search_tool,
                    tool_observation=search_evidence,
                    parent_turn_id=f"{episode_id}:0",
                )
            )
            turn_id += 1

            access_prompt = self._access_prompt(observation, subtask, search_evidence)
            access_response = policy_backend.generate(
                PolicyRequest(
                    agent=worker,
                    task_id=task.task_id,
                    observation=search_evidence,
                    prompt=access_prompt,
                    team_context=f"Search observation: {search_evidence}",
                    metadata={
                        "stage": "wideseek_access",
                        "search_query": subtask,
                        "tool_name": self.access_tool,
                        "worker_index": worker_index,
                        "answer": task.answer,
                    },
                )
            )
            access_query, access_valid = _parse_call(access_response.text, self.access_tool)
            access_evidence = environment.execute_tool(self.access_tool, task, access_query or search_query)
            trajectory.add_turn(
                self._turn(
                    trajectory=trajectory,
                    turn_id=turn_id,
                    agent=worker,
                    observation=search_evidence,
                    prompt=access_prompt,
                    response=access_response,
                    turn_role="subagent_access",
                    agent_instance=worker.name,
                    subtrajectory_id=worker_index,
                    format_valid=access_valid,
                    parallel_wave=1,
                    sub_goal=subtask,
                    tool_name=self.access_tool,
                    tool_observation=access_evidence,
                    parent_turn_id=parent_turn_id,
                )
            )
            turn_id += 1

            summary_prompt = self._summary_prompt(observation, subtask, access_evidence)
            summary_response = policy_backend.generate(
                PolicyRequest(
                    agent=worker,
                    task_id=task.task_id,
                    observation=access_evidence,
                    prompt=summary_prompt,
                    team_context=f"Access observation: {access_evidence}",
                    metadata={
                        "stage": "wideseek_summary",
                        "evidence": access_evidence,
                        "worker_index": worker_index,
                        "answer": task.answer,
                    },
                )
            )
            summary_valid = _has_prefix(summary_response.text, "Subagent result")
            summaries.append(summary_response.text)
            trajectory.add_turn(
                self._turn(
                    trajectory=trajectory,
                    turn_id=turn_id,
                    agent=worker,
                    observation=access_evidence,
                    prompt=summary_prompt,
                    response=summary_response,
                    turn_role="subagent_summary",
                    agent_instance=worker.name,
                    subtrajectory_id=worker_index,
                    format_valid=summary_valid,
                    parallel_wave=1,
                    sub_goal=subtask,
                    tool_result=summary_response.text,
                    parent_turn_id=parent_turn_id,
                )
            )
            turn_id += 1

        final_prompt = self._final_prompt(observation, summaries)
        final_response = policy_backend.generate(
            PolicyRequest(
                agent=lead,
                task_id=task.task_id,
                observation=observation,
                prompt=final_prompt,
                team_context="\n".join(summaries),
                metadata={"stage": "wideseek_final", "answer": task.answer},
            )
        )
        final_valid = _has_prefix(final_response.text, "Final answer")
        trajectory.add_turn(
            self._turn(
                trajectory=trajectory,
                turn_id=turn_id,
                agent=lead,
                observation=observation,
                prompt=final_prompt,
                response=final_response,
                turn_role="lead_final",
                agent_instance=lead.name,
                subtrajectory_id=0,
                format_valid=final_valid,
                parallel_wave=2,
                sub_goal="",
                done=True,
            )
        )
        trajectory.final_answer = final_response.text
        trajectory.metadata.update(
            {
                "active_subagents": len(summaries),
                "agent_count": 1 + len(summaries),
                "plan_format_valid": plan_valid,
                "final_format_valid": final_valid,
                "subagent_contexts_isolated": True,
            }
        )
        for turn in trajectory.turns:
            turn.metadata["wideseek_agent_count"] = 1 + len(summaries)
        return trajectory

    def _turn(
        self,
        *,
        trajectory: MultiAgentTrajectory,
        turn_id: int,
        agent: AgentSpec,
        observation: str,
        prompt: str,
        response: PolicyResponse,
        turn_role: str,
        agent_instance: str,
        subtrajectory_id: int,
        format_valid: bool,
        parallel_wave: int,
        sub_goal: str,
        tool_name: str = "",
        tool_observation: str = "",
        tool_result: str = "",
        parent_turn_id: str = "",
        done: bool = False,
    ) -> AgentTurn:
        return AgentTurn(
            episode_id=trajectory.episode_id,
            task_id=trajectory.task_id,
            turn_id=turn_id,
            agent_name=agent.name,
            role=agent.role,
            policy_group=agent.policy_group,
            observation=observation,
            prompt=prompt,
            action_text=response.text,
            action_token_ids=response.token_ids,
            action_logprobs=response.logprobs,
            done=done,
            metadata=response.metadata
            | {
                "wideseek_turn_role": turn_role,
                "wideseek_agent_instance": agent_instance,
                "wideseek_subtrajectory_id": subtrajectory_id,
                "wideseek_format_valid": bool(format_valid),
                "wideseek_parallel_wave": parallel_wave,
                "wideseek_parent_turn_id": parent_turn_id,
                "tool_name": tool_name,
                "sub_goal": sub_goal,
                "tool_observation": tool_observation,
                "tool_result": tool_result,
            },
        )

    def _validate_team(self, *, lead: AgentSpec, workers: tuple[AgentSpec, ...]) -> None:
        if len(workers) < self.max_parallel_subagents:
            raise ValueError("WideSeek-R1 team has fewer subagents than max_parallel_subagents.")
        if any(agent.policy_group != lead.policy_group for agent in workers):
            raise ValueError("WideSeek-R1 lead and subagents must share one policy group.")
        if not lead.trainable or any(not agent.trainable for agent in workers):
            raise ValueError("WideSeek-R1 jointly trains the lead and every active subagent.")
        if "subagent" not in lead.tools:
            raise ValueError("WideSeek-R1 lead must expose the subagent delegation tool.")
        for worker in workers[: self.max_parallel_subagents]:
            if self.search_tool not in worker.tools or self.access_tool not in worker.tools:
                raise ValueError("WideSeek-R1 subagents must expose both search and access tools.")

    def _plan_prompt(self, question: str) -> str:
        return (
            "You are the WideSeek-R1 lead agent. Decompose the broad request into independent research subtasks.\n"
            f"Question: {question}\n"
            f"Return 1 to {self.max_parallel_subagents} lines formatted exactly as 'CALL subagent: <subtask>'."
        )

    def _search_prompt(self, question: str, subtask: str) -> str:
        return (
            "You are an isolated WideSeek-R1 subagent. Search for evidence for only your assigned subtask.\n"
            f"Main question: {question}\nSubtask: {subtask}\n"
            f"Return exactly 'CALL {self.search_tool}: <query>'."
        )

    def _access_prompt(self, question: str, subtask: str, search_evidence: str) -> str:
        return (
            "Inspect the most useful search result without using another subagent's context.\n"
            f"Main question: {question}\nSubtask: {subtask}\nSearch result: {search_evidence}\n"
            f"Return exactly 'CALL {self.access_tool}: <query-or-url>'."
        )

    @staticmethod
    def _summary_prompt(question: str, subtask: str, evidence: str) -> str:
        return (
            "Summarize the evidence for the lead agent.\n"
            f"Main question: {question}\nSubtask: {subtask}\nEvidence: {evidence}\n"
            "Return exactly 'Subagent result: <concise evidence>'."
        )

    @staticmethod
    def _final_prompt(question: str, summaries: list[str]) -> str:
        evidence = "\n".join(summaries) if summaries else "(no valid subagent result)"
        return (
            "Synthesize the isolated subagent results into the requested answer.\n"
            f"Question: {question}\nSubagent results:\n{evidence}\n"
            "Return exactly 'Final answer: <answer>'."
        )


def _parse_subtasks(text: str, *, limit: int) -> tuple[str, ...]:
    subtasks = []
    for match in re.finditer(r"(?im)^\s*CALL\s+subagent\s*:\s*(\S(?:.*\S)?)\s*$", text):
        value = match.group(1).strip()
        if value and value not in subtasks:
            subtasks.append(value)
        if len(subtasks) >= limit:
            break
    return tuple(subtasks)


def _parse_call(text: str, target: str) -> tuple[str, bool]:
    match = re.fullmatch(r"\s*CALL\s+([^\s:]+)\s*:\s*(\S(?:.*\S)?)\s*", text, flags=re.IGNORECASE)
    if match is None:
        return "", False
    actual, request = match.groups()
    return request.strip(), actual.lower() == target.lower()


def _has_prefix(text: str, prefix: str) -> bool:
    return re.fullmatch(rf"\s*{re.escape(prefix)}\s*:\s*\S(?:.*\S)?\s*", text, flags=re.IGNORECASE) is not None
