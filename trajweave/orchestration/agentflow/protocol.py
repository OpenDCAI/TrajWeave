from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from trajweave.backends.local import extract_final_int
from trajweave.backends.policy import PolicyBackend, PolicyRequest, StableByteTokenizer
from trajweave.core.specs import TeamSpec
from trajweave.core.trajectory import AgentTurn, MultiAgentTrajectory
from trajweave.envs.math import MathTask
from trajweave.orchestration.base import TeamContext


@dataclass
class AgentFlowMemory:
    actions: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add_action(self, step_id: int, tool_name: str, sub_goal: str, command: str, result: str) -> None:
        self.actions[f"Action Step {step_id}"] = {
            "tool_name": tool_name,
            "sub_goal": sub_goal,
            "command": command,
            "result": result,
        }

    def render(self) -> str:
        if not self.actions:
            return "{}"
        lines: list[str] = []
        for step, item in self.actions.items():
            lines.append(
                f"{step}: tool={item['tool_name']}; sub_goal={item['sub_goal']}; "
                f"command={item['command']}; result={item['result']}"
            )
        return "\n".join(lines)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {key: dict(value) for key, value in self.actions.items()}


@dataclass
class AgentFlowPlannerToolOrchestra:
    planner_name: str = "planner"
    executor_name: str = "executor"
    verifier_name: str = "verifier"
    tool_name: str = "base_generator"
    tokenizer: StableByteTokenizer = field(default_factory=StableByteTokenizer)

    def run(
        self,
        *,
        episode_id: str,
        rollout_group: str,
        task: MathTask,
        team: TeamSpec,
        observation: str,
        policy_backend: PolicyBackend,
        environment: object | None = None,
    ) -> MultiAgentTrajectory:
        del environment
        trajectory = MultiAgentTrajectory(
            episode_id=episode_id,
            task_id=task.task_id,
            rollout_group=rollout_group,
            team_name=team.name,
        )
        context = TeamContext()
        memory = AgentFlowMemory()
        final_answer = ""
        turn_id = 0

        for step_id in range(1, team.max_turns + 1):
            planner = team.agent(self.planner_name)
            planner_prompt = self._planner_prompt(observation, memory, step_id, team.max_turns)
            planner_response = policy_backend.generate(
                PolicyRequest(
                    agent=planner,
                    task_id=task.task_id,
                    observation=observation,
                    prompt=planner_prompt,
                    team_context=context.render(),
                    metadata={"answer": task.answer, "step_id": step_id},
                )
            )
            context_text, sub_goal, selected_tool = self._parse_planner_step(planner_response.text)
            selected_tool = _resolve_allowed_tool(planner_response.text, selected_tool, planner.tools)
            plan_valid = bool(sub_goal and selected_tool in planner.tools)
            planner_metadata = planner_response.metadata | {
                "agentflow_stage": "planner_next_step",
                "step_id": step_id,
                "tool_name": selected_tool,
                "sub_goal": sub_goal,
                "context": context_text,
                "plan_valid": plan_valid,
                "memory_snapshot": memory.snapshot(),
            }
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
                    metadata=planner_metadata,
                )
            )
            turn_id += 1

            executor = team.agent(self.executor_name)
            command = self._executor_command(observation, sub_goal, selected_tool)
            trajectory.add_turn(
                self._frozen_turn(
                    episode_id=episode_id,
                    task_id=task.task_id,
                    turn_id=turn_id,
                    agent_name=executor.name,
                    role=executor.role,
                    policy_group=executor.policy_group,
                    observation=observation,
                    prompt=self._executor_prompt(observation, sub_goal, selected_tool),
                    action_text=command,
                    metadata={
                        "agentflow_stage": "executor_command",
                        "step_id": step_id,
                        "tool_name": selected_tool,
                        "sub_goal": sub_goal,
                    },
                )
            )
            turn_id += 1

            tool_result = self._execute_tool(task, selected_tool, plan_valid=plan_valid)
            memory.add_action(step_id, selected_tool, sub_goal, command, tool_result)
            tool_agent_name = selected_tool or "invalid_tool"
            trajectory.add_turn(
                self._frozen_turn(
                    episode_id=episode_id,
                    task_id=task.task_id,
                    turn_id=turn_id,
                    agent_name=tool_agent_name,
                    role="tool",
                    policy_group="tool",
                    observation=observation,
                    prompt=command,
                    action_text=tool_result,
                    metadata={
                        "agentflow_stage": "tool_result",
                        "step_id": step_id,
                        "tool_name": selected_tool,
                        "memory_snapshot": memory.snapshot(),
                    },
                )
            )
            turn_id += 1

            verifier = team.agent(self.verifier_name)
            verifier_prompt = self._verifier_prompt(observation, memory)
            verifier_text, verified = self._verify(task, memory)
            for turn in trajectory.turns:
                if (
                    turn.metadata.get("agentflow_stage") == "planner_next_step"
                    and turn.metadata.get("step_id") == step_id
                ):
                    turn.metadata["verifier_decision"] = "STOP" if verified else "CONTINUE"
                    turn.metadata["tool_result"] = tool_result
            trajectory.add_turn(
                self._frozen_turn(
                    episode_id=episode_id,
                    task_id=task.task_id,
                    turn_id=turn_id,
                    agent_name=verifier.name,
                    role=verifier.role,
                    policy_group=verifier.policy_group,
                    observation=observation,
                    prompt=verifier_prompt,
                    action_text=verifier_text,
                    done=verified,
                    metadata={
                        "agentflow_stage": "verifier_decision",
                        "step_id": step_id,
                        "verifier_decision": "STOP" if verified else "CONTINUE",
                        "memory_snapshot": memory.snapshot(),
                    },
                )
            )
            turn_id += 1
            if verified:
                final_answer = tool_result
                break

        trajectory.final_answer = final_answer or self._final_from_memory(memory)
        trajectory.metadata["agentflow_protocol"] = {
            "trainable_agent": self.planner_name,
            "frozen_agents": [self.executor_name, self.verifier_name],
            "tool_name": self.tool_name,
            "credit": "planner_only",
        }
        trajectory.metadata["memory"] = memory.snapshot()
        return trajectory

    def _planner_prompt(self, observation: str, memory: AgentFlowMemory, step_id: int, max_steps: int) -> str:
        return (
            "You are the Planner in AgentFlow. Choose one allowed tool and one short sub-goal.\n"
            f"Question: {observation}\n"
            f"Memory:\n{memory.render()}\n"
            f"Current step: {step_id}/{max_steps}\n"
            "Output exactly these three short lines, with no explanation:\n"
            f"Tool Name: {self.tool_name}\n"
            "Sub-Goal: calculate the answer\n"
            "Context: use the question and memory"
        )

    def _executor_prompt(self, observation: str, sub_goal: str, tool_name: str) -> str:
        return f"Question: {observation}\nSub-goal: {sub_goal}\nTool: {tool_name}\nGenerate a tool command."

    def _verifier_prompt(self, observation: str, memory: AgentFlowMemory) -> str:
        return f"Question: {observation}\nMemory:\n{memory.render()}\nReturn STOP if enough, otherwise CONTINUE."

    def _parse_planner_step(self, text: str) -> tuple[str, str, str]:
        context = _find_field(text, "Context") or ""
        sub_goal = _find_field(text, "Sub-Goal") or ""
        tool_name = _find_field(text, "Tool Name") or ""
        return context.strip(), sub_goal.strip(), tool_name.strip()

    def _executor_command(self, observation: str, sub_goal: str, tool_name: str) -> str:
        del sub_goal
        return f'execution = tool.execute(query="{observation}", tool="{tool_name}")'

    def _execute_tool(self, task: MathTask, tool_name: str, *, plan_valid: bool) -> str:
        if not plan_valid:
            return "Tool error: invalid planner action"
        if tool_name not in {"base_generator", "python_stub"}:
            return f"Tool error: unsupported tool {tool_name}"
        answer = _parse_arithmetic(task.question)
        if answer is None:
            return "Tool error: the selected tool could not solve this task"
        return f"Final answer: {answer}"

    def _verify(self, task: MathTask, memory: AgentFlowMemory) -> tuple[str, bool]:
        predicted = extract_final_int(self._final_from_memory(memory))
        verified = predicted == task.answer
        decision = "STOP" if verified else "CONTINUE"
        return f"Conclusion: {decision}", verified

    def _final_from_memory(self, memory: AgentFlowMemory) -> str:
        for item in reversed(list(memory.actions.values())):
            result = str(item.get("result", ""))
            if result:
                return result
        return "Final answer: 0"

    def _frozen_turn(
        self,
        *,
        episode_id: str,
        task_id: str,
        turn_id: int,
        agent_name: str,
        role: str,
        policy_group: str,
        observation: str,
        prompt: str,
        action_text: str,
        metadata: dict[str, Any],
        done: bool = False,
    ) -> AgentTurn:
        token_ids = self.tokenizer.encode(action_text)
        return AgentTurn(
            episode_id=episode_id,
            task_id=task_id,
            turn_id=turn_id,
            agent_name=agent_name,
            role=role,
            policy_group=policy_group,
            observation=observation,
            prompt=prompt,
            action_text=action_text,
            action_token_ids=token_ids,
            action_logprobs=[0.0] * len(token_ids),
            done=done,
            metadata=metadata,
        )


def _find_field(text: str, field: str) -> str | None:
    pattern = rf"{re.escape(field)}\s*:\s*(.*?)(?=\n[A-Z][A-Za-z -]*\s*:|$)"
    match = re.search(pattern, text, flags=re.DOTALL)
    return match.group(1).strip() if match else None


def _resolve_allowed_tool(text: str, parsed_tool: str, allowed_tools: tuple[str, ...]) -> str:
    candidates = [parsed_tool] if parsed_tool else []
    candidates.extend(line.strip() for line in text.splitlines()[:3])
    for candidate in candidates:
        normalized = re.sub(r"[^a-z0-9]+", "_", candidate.lower()).strip("_")
        for tool in allowed_tools:
            if normalized == re.sub(r"[^a-z0-9]+", "_", tool.lower()).strip("_"):
                return tool
    return parsed_tool.strip()


def _parse_arithmetic(text: str) -> int | None:
    match = re.search(r"(-?\d+)\s*([+\-*x])\s*(-?\d+)", text)
    if not match:
        return None
    left = int(match.group(1))
    op = match.group(2)
    right = int(match.group(3))
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    return left * right
