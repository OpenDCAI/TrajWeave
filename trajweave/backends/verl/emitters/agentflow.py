from __future__ import annotations

from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput

from trajweave.backends.verl.runtime_config import config_get
from trajweave.backends.verl.schema import to_python


class AgentFlowEmitterMixin:
    def _build_agentflow_planner_tool_outputs(
        self,
        prompt: dict[str, Any],
        *,
        session_id: int = 0,
    ) -> list[AgentLoopOutput]:
        raw_prompt = to_python(prompt.get("raw_prompt", []))
        prompt_ids = self._encode_prompt(raw_prompt)
        reward_model = to_python(prompt.get("reward_model", {})) or {}
        ground_truth = str(reward_model.get("ground_truth", "2"))
        is_correct = session_id % 2 == 0
        reward = 1.0 if is_correct else 0.0
        max_steps = self._agentflow_max_steps()
        metrics = AgentLoopMetrics(generate_sequences=0.0, tool_calls=float(max_steps), compute_score=0.0, num_preempted=-1)
        outputs: list[AgentLoopOutput] = []
        running_prompt_ids = list(prompt_ids)
        final_answer = ground_truth if is_correct else "__wrong__"
        for step_id in range(1, max_steps + 1):
            tool_name = self._agentflow_enabled_tools()[0]
            verifier_decision = "STOP" if is_correct or step_id == max_steps else "CONTINUE"
            text = (
                f" Context: solve with memory step {step_id}.\n"
                f"Sub-Goal: compute final answer.\n"
                f"Tool Name: {tool_name}\n"
                f"Planner answer: Final answer: {final_answer}"
            )
            response_ids = self._encode_text(text)
            outputs.append(
                AgentLoopOutput(
                    prompt_ids=list(running_prompt_ids),
                    response_ids=response_ids,
                    response_mask=[1] * len(response_ids),
                    reward_score=reward,
                    num_turns=step_id,
                    metrics=metrics,
                    extra_fields={
                        "turn_scores": [],
                        "tool_rewards": [],
                        "trajweave_agent_name": "planner",
                        "trajweave_role": "planner",
                        "agent_id": "planner",
                        "policy_group": "planner",
                        "agentflow_stage": "planner_next_step",
                        "tool_name": tool_name,
                        "sub_goal": "compute final answer",
                        "tool_result": f"Final answer: {final_answer}",
                        "verifier_decision": verifier_decision,
                        "memory_snapshot": f"step={step_id}; result=Final answer: {final_answer}",
                        "step_id": step_id,
                        "agentflow_trace": self._agentflow_trace(
                            step_id=step_id,
                            tool_name=tool_name,
                            sub_goal="compute final answer",
                            tool_result=f"Final answer: {final_answer}",
                            verifier_decision=verifier_decision,
                        ),
                    },
                )
            )
            running_prompt_ids = running_prompt_ids + response_ids
            if verifier_decision == "STOP":
                break
        return outputs

    def _build_hf_agentflow_planner_tool_outputs(
        self,
        prompt: dict[str, Any],
        *,
        session_id: int = 0,
    ) -> list[AgentLoopOutput]:
        raw_prompt = to_python(prompt.get("raw_prompt", []))
        prompt_ids = self._encode_prompt(raw_prompt)
        reward_model = to_python(prompt.get("reward_model", {})) or {}
        ground_truth = str(reward_model.get("ground_truth", "2"))
        is_correct = session_id % 2 == 0
        reward = 1.0 if is_correct else 0.0
        max_steps = self._agentflow_max_steps()
        tool_name = self._agentflow_enabled_tools()[0]
        metrics = AgentLoopMetrics(generate_sequences=1.0, tool_calls=float(max_steps), compute_score=0.0, num_preempted=-1)
        outputs: list[AgentLoopOutput] = []
        running_prompt_ids = list(prompt_ids)
        for step_id in range(1, max_steps + 1):
            response_ids = self._generate_local_response_ids(running_prompt_ids, policy_group="planner")
            verifier_decision = "STOP" if is_correct or step_id == max_steps else "CONTINUE"
            tool_result = f"Final answer: {ground_truth if is_correct else '__wrong__'}"
            outputs.append(
                AgentLoopOutput(
                    prompt_ids=list(running_prompt_ids),
                    response_ids=response_ids,
                    response_mask=[1] * len(response_ids),
                    reward_score=reward,
                    num_turns=step_id,
                    metrics=metrics,
                    extra_fields={
                        "turn_scores": [],
                        "tool_rewards": [],
                        "trajweave_agent_name": "planner",
                        "trajweave_role": "planner",
                        "agent_id": "planner",
                        "policy_group": "planner",
                        "agentflow_stage": "planner_next_step",
                        "tool_name": tool_name,
                        "sub_goal": "compute final answer",
                        "tool_result": tool_result,
                        "verifier_decision": verifier_decision,
                        "memory_snapshot": f"step={step_id}; result={tool_result}",
                        "step_id": step_id,
                        "rollout_source": "hf_local_tq",
                        "agentflow_trace": self._agentflow_trace(
                            step_id=step_id,
                            tool_name=tool_name,
                            sub_goal="compute final answer",
                            tool_result=tool_result,
                            verifier_decision=verifier_decision,
                        ),
                    },
                )
            )
            running_prompt_ids = running_prompt_ids + response_ids
            if verifier_decision == "STOP":
                break
        return outputs

    def _agentflow_max_steps(self) -> int:
        agentflow_cfg = self._agentflow_orchestra_config()
        return int(config_get(agentflow_cfg, "max_steps", default=3))

    def _agentflow_enabled_tools(self) -> list[str]:
        agentflow_cfg = self._agentflow_orchestra_config()
        tools = to_python(config_get(agentflow_cfg, "enabled_tools", default=["base_generator"]))
        if not tools:
            return ["base_generator"]
        return [str(tool) for tool in tools]

    def _agentflow_orchestra_config(self) -> Any:
        agent_cfg = config_get(self.config, "agent", default={}) or {}
        orchestra_cfg = config_get(agent_cfg, "orchestra", default={}) or {}
        return config_get(orchestra_cfg, "agentflow", default={}) or {}

    def _agentflow_trace(
        self,
        *,
        step_id: int,
        tool_name: str,
        sub_goal: str,
        tool_result: str,
        verifier_decision: str,
    ) -> list[dict[str, Any]]:
        return [
            {
                "agent_name": "executor",
                "role": "executor",
                "policy_group": "frozen",
                "agent_id": "executor",
                "stage": "executor_command",
                "step_id": step_id,
                "tool_name": tool_name,
                "sub_goal": sub_goal,
                "action_text": f'execution = tool.execute(tool="{tool_name}")',
            },
            {
                "agent_name": tool_name,
                "role": "tool",
                "policy_group": "tool",
                "agent_id": tool_name,
                "stage": "tool_result",
                "step_id": step_id,
                "tool_name": tool_name,
                "tool_result": tool_result,
            },
            {
                "agent_name": "verifier",
                "role": "verifier",
                "policy_group": "frozen",
                "agent_id": "verifier",
                "stage": "verifier_decision",
                "step_id": step_id,
                "verifier_decision": verifier_decision,
            },
        ]
