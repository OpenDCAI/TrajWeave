from __future__ import annotations

import re
from dataclasses import dataclass

from trajweave.backends.policy import PolicyBackend, PolicyRequest, PolicyResponse
from trajweave.core.specs import AgentSpec, TeamSpec
from trajweave.core.trajectory import AgentTurn, MultiAgentTrajectory
from trajweave.envs.search import SearchAnswerEnvironment, SearchTask


@dataclass
class MrlXResearchOrchestra:
    """Main/sub-agent research loop adapted from MrlX-DeepResearch."""

    explorer_name: str = "main_explorer"
    adapter_name: str = "sub_adapter"
    tool_name: str = "search_and_browse"
    research_rounds: int = 1

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
            raise ValueError("MrlXResearchOrchestra requires an environment with execute_tool().")
        if self.research_rounds < 1:
            raise ValueError("MrlX research_rounds must be at least 1.")

        explorer = team.agent(self.explorer_name)
        adapter = team.agent(self.adapter_name)
        self._validate_team(explorer=explorer, adapter=adapter)
        trajectory = MultiAgentTrajectory(
            episode_id=episode_id,
            task_id=task.task_id,
            rollout_group=rollout_group,
            team_name=team.name,
            metadata={
                "coordination_protocol": "mrlx_async_research",
                "explorer_agent": explorer.name,
                "adapter_agent": adapter.name,
                "adapter_policy_lag": 1,
            },
        )
        main_history: list[str] = []
        turn_id = 0

        for round_id in range(self.research_rounds):
            delegate_prompt = self._delegate_prompt(observation, main_history)
            delegate_response = policy_backend.generate(
                PolicyRequest(
                    agent=explorer,
                    task_id=task.task_id,
                    observation=observation,
                    prompt=delegate_prompt,
                    team_context=self._history(main_history),
                    metadata={
                        "stage": "mrlx_delegate",
                        "answer": task.answer,
                        "search_query": task.search_query or observation,
                        "adapter_agent": adapter.name,
                        "round_id": round_id,
                    },
                )
            )
            delegated_query, delegate_valid = _parse_call(delegate_response.text, expected_target=adapter.name)
            effective_query = delegated_query if delegate_valid else str(task.search_query or observation)
            trajectory.add_turn(
                self._turn(
                    trajectory=trajectory,
                    turn_id=turn_id,
                    agent_name=explorer.name,
                    role=explorer.role,
                    policy_group=explorer.policy_group,
                    observation=observation,
                    prompt=delegate_prompt,
                    response=delegate_response,
                    turn_role="delegate",
                    format_valid=delegate_valid,
                    training_mode="on_policy_explorer",
                    round_id=round_id,
                    sub_goal=effective_query,
                    parent_turn_id="",
                    done=not delegate_valid,
                )
            )
            parent_turn_id = f"{episode_id}:{turn_id}"
            turn_id += 1
            delegate_history = f"Main explorer: {delegate_response.text}"
            if not delegate_valid:
                main_history.append(delegate_history)
                trajectory.metadata["mrlx_stop_reason"] = "invalid_explorer_delegation"
                trajectory.metadata["research_history"] = tuple(main_history)
                trajectory.final_answer = ""
                return trajectory
            adapter_history = [*main_history, delegate_history]

            tool_prompt = self._adapter_tool_prompt(observation, effective_query)
            tool_response = policy_backend.generate(
                PolicyRequest(
                    agent=adapter,
                    task_id=task.task_id,
                    observation=effective_query,
                    prompt=tool_prompt,
                    team_context=self._history(adapter_history),
                    metadata={
                        "stage": "mrlx_adapter_tool",
                        "answer": task.answer,
                        "search_query": effective_query,
                        "tool_name": self.tool_name,
                        "round_id": round_id,
                    },
                )
            )
            tool_query, tool_valid = _parse_call(tool_response.text, expected_target=self.tool_name)
            evidence = (
                environment.execute_tool(self.tool_name, task, tool_query)
                if tool_valid
                else "No evidence: adapter tool call was invalid."
            )
            trajectory.add_turn(
                self._turn(
                    trajectory=trajectory,
                    turn_id=turn_id,
                    agent_name=adapter.name,
                    role=adapter.role,
                    policy_group=adapter.policy_group,
                    observation=effective_query,
                    prompt=tool_prompt,
                    response=tool_response,
                    turn_role="adapter_tool",
                    format_valid=tool_valid,
                    training_mode="off_policy_adapter",
                    round_id=round_id,
                    sub_goal=effective_query,
                    parent_turn_id=parent_turn_id,
                    tool_observation=evidence,
                )
            )
            turn_id += 1
            adapter_history.extend(
                (f"Sub adapter tool call: {tool_response.text}", f"Tool observation: {evidence}")
            )

            result_prompt = self._adapter_result_prompt(observation, effective_query, evidence)
            result_response = policy_backend.generate(
                PolicyRequest(
                    agent=adapter,
                    task_id=task.task_id,
                    observation=evidence,
                    prompt=result_prompt,
                    team_context=self._history(adapter_history),
                    metadata={
                        "stage": "mrlx_adapter_result",
                        "answer": task.answer,
                        "evidence": evidence,
                        "round_id": round_id,
                    },
                )
            )
            result_valid = _has_prefixed_content(result_response.text, "Research result")
            trajectory.add_turn(
                self._turn(
                    trajectory=trajectory,
                    turn_id=turn_id,
                    agent_name=adapter.name,
                    role=adapter.role,
                    policy_group=adapter.policy_group,
                    observation=evidence,
                    prompt=result_prompt,
                    response=result_response,
                    turn_role="adapter_result",
                    format_valid=result_valid,
                    training_mode="off_policy_adapter",
                    round_id=round_id,
                    sub_goal=effective_query,
                    parent_turn_id=parent_turn_id,
                    tool_observation=evidence,
                    tool_result=result_response.text,
                )
            )
            turn_id += 1
            main_history.extend((delegate_history, f"Sub adapter result: {result_response.text}"))

        final_prompt = self._final_prompt(observation, main_history)
        final_response = policy_backend.generate(
            PolicyRequest(
                agent=explorer,
                task_id=task.task_id,
                observation=observation,
                prompt=final_prompt,
                team_context=self._history(main_history),
                metadata={"stage": "mrlx_final", "answer": task.answer},
            )
        )
        final_valid = _has_prefixed_content(final_response.text, "Final answer")
        trajectory.add_turn(
            self._turn(
                trajectory=trajectory,
                turn_id=turn_id,
                agent_name=explorer.name,
                role=explorer.role,
                policy_group=explorer.policy_group,
                observation=observation,
                prompt=final_prompt,
                response=final_response,
                turn_role="final",
                format_valid=final_valid,
                training_mode="on_policy_explorer",
                round_id=self.research_rounds,
                sub_goal="",
                parent_turn_id="",
                done=True,
            )
        )
        trajectory.final_answer = final_response.text
        trajectory.metadata["research_history"] = tuple(main_history)
        return trajectory

    def _turn(
        self,
        *,
        trajectory: MultiAgentTrajectory,
        turn_id: int,
        agent_name: str,
        role: str,
        policy_group: str,
        observation: str,
        prompt: str,
        response: PolicyResponse,
        turn_role: str,
        format_valid: bool,
        training_mode: str,
        round_id: int,
        sub_goal: str,
        parent_turn_id: str,
        tool_observation: str = "",
        tool_result: str = "",
        done: bool = False,
    ) -> AgentTurn:
        return AgentTurn(
            episode_id=trajectory.episode_id,
            task_id=trajectory.task_id,
            turn_id=turn_id,
            agent_name=agent_name,
            role=role,
            policy_group=policy_group,
            observation=observation,
            prompt=prompt,
            action_text=response.text,
            action_token_ids=response.token_ids,
            action_logprobs=response.logprobs,
            done=done,
            metadata=response.metadata
            | {
                "mrlx_turn_role": turn_role,
                "mrlx_format_valid": bool(format_valid),
                "mrlx_training_mode": training_mode,
                "mrlx_policy_lag": 0 if training_mode == "on_policy_explorer" else 1,
                "mrlx_parent_turn_id": parent_turn_id,
                "round_id": round_id,
                "tool_name": self.tool_name,
                "sub_goal": sub_goal,
                "tool_observation": tool_observation,
                "tool_result": tool_result,
            },
        )

    def _validate_team(self, *, explorer: AgentSpec, adapter: AgentSpec) -> None:
        if explorer.policy_group == adapter.policy_group:
            raise ValueError("MrlX explorer and adapter require distinct policy groups.")
        if not explorer.trainable or not adapter.trainable:
            raise ValueError("MrlX explorer and adapter must both be trainable.")
        if adapter.name not in explorer.tools:
            raise ValueError("MrlX explorer must expose the adapter as a tool.")
        if self.tool_name not in adapter.tools:
            raise ValueError("MrlX adapter must expose the configured research tool.")

    def _delegate_prompt(self, question: str, history: list[str]) -> str:
        return (
            "You are the MrlX on-policy main explorer. Delegate one focused research query.\n"
            f"Question: {question}\n"
            f"Previous research:\n{self._history(history)}\n"
            f"Return exactly 'CALL {self.adapter_name}: <query>'."
        )

    def _adapter_tool_prompt(self, question: str, sub_goal: str) -> str:
        return (
            "You are the MrlX off-policy sub adapter. Select one research tool request.\n"
            f"Question: {question}\nDelegated query: {sub_goal}\n"
            f"Return exactly 'CALL {self.tool_name}: <query>'."
        )

    @staticmethod
    def _adapter_result_prompt(question: str, sub_goal: str, evidence: str) -> str:
        return (
            "Summarize the evidence for the main explorer.\n"
            f"Question: {question}\nDelegated query: {sub_goal}\nEvidence: {evidence}\n"
            "Return exactly 'Research result: <summary>'."
        )

    @staticmethod
    def _final_prompt(question: str, history: list[str]) -> str:
        return (
            "Use the research history to answer the question.\n"
            f"Question: {question}\nResearch history:\n{MrlXResearchOrchestra._history(history)}\n"
            "Return exactly 'Final answer: <answer>'."
        )

    @staticmethod
    def _history(history: list[str]) -> str:
        return "\n".join(history) if history else "(empty)"


def _parse_call(text: str, *, expected_target: str) -> tuple[str, bool]:
    match = re.fullmatch(r"\s*CALL\s+([^\s:]+)\s*:\s*(\S(?:.*\S)?)\s*", text, flags=re.IGNORECASE)
    if match is None:
        return "", False
    target, request = match.groups()
    valid = _normalize_token(target) == _normalize_token(expected_target) and "CALL " not in request.upper()
    return request.strip(), valid


def _has_prefixed_content(text: str, prefix: str) -> bool:
    return re.fullmatch(rf"\s*{re.escape(prefix)}\s*:\s*\S(?:.*\S)?\s*", text, flags=re.IGNORECASE) is not None


def _normalize_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
