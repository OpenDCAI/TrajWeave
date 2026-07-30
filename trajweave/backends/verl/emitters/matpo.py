from __future__ import annotations

import hashlib
import json
from typing import Any

from trajweave.backends.verl.runtime_config import config_get
from trajweave.backends.verl.schema import to_python
from trajweave.credit.matpo.parent_broadcast import combined_matpo_reward
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput


class MATPOEmitterMixin:
    def _build_matpo_browse_outputs(self, prompt: dict[str, Any], *, session_id: int = 0) -> list[AgentLoopOutput]:
        raw_prompt = to_python(prompt.get("raw_prompt", []))
        reward_model = to_python(prompt.get("reward_model", {})) or {}
        extra_info = to_python(prompt.get("extra_info", {})) or {}
        ground_truth = str(reward_model.get("ground_truth", "Paris"))
        search_query = str(extra_info.get("search_query", ground_truth))
        prompt_ids = self._encode_prompt(raw_prompt)
        answer = ground_truth if session_id % 2 == 0 else "__wrong__"
        accuracy = 1.0 if session_id % 2 == 0 else 0.0
        accuracy_weight, tool_format_weight = self._matpo_reward_weights()
        combined_reward = combined_matpo_reward(
            accuracy=accuracy,
            planner_format=1.0,
            worker_formats=(1.0,),
            accuracy_reward_weight=accuracy_weight,
            tool_format_reward_weight=tool_format_weight,
        )

        planner_agent = self._matpo_planner_agent()
        worker_agent = self._matpo_worker_agent()
        tool_name = self._matpo_tool_name()
        request_prefix = self._matpo_request_prefix(prompt, raw_prompt=raw_prompt, session_id=session_id)
        planner_delegate_req_id = f"{request_prefix}:planner_delegate"
        worker_call_req_id = f"{request_prefix}:worker_call"
        worker_summary_req_id = f"{request_prefix}:worker_summary"
        planner_final_req_id = f"{request_prefix}:planner_final"

        planner_delegate_text = f" CALL {worker_agent}: {search_query}"
        worker_call_text = f" CALL {tool_name}: {search_query}"
        tool_observation = f"Offline evidence for {search_query}: {ground_truth}"
        worker_summary_text = f" Evidence summary: {tool_observation}. Suggested final answer: {answer}"
        planner_final_text = f" Final answer: {answer}"
        observation_text = self._matpo_observation_text(raw_prompt)
        final_answer = planner_final_text.strip()
        workflow_success = bool(accuracy)
        planner_delegate_prompt = observation_text
        worker_call_prompt = "\n".join((planner_delegate_prompt, planner_delegate_text.strip()))
        worker_summary_prompt = "\n".join((worker_call_prompt, worker_call_text.strip(), tool_observation))
        planner_final_prompt = "\n".join((worker_summary_prompt, worker_summary_text.strip()))
        planner_delegate_ids = self._encode_text(planner_delegate_text)
        worker_call_ids = self._encode_text(worker_call_text)
        tool_observation_ids = self._encode_text(f" {tool_observation}")
        worker_summary_ids = self._encode_text(worker_summary_text)
        planner_final_ids = self._encode_text(planner_final_text)
        metrics = AgentLoopMetrics(generate_sequences=4.0, tool_calls=1.0, compute_score=0.0, num_preempted=-1)

        return [
            AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=planner_delegate_ids,
                response_mask=[1] * len(planner_delegate_ids),
                reward_score=combined_reward,
                num_turns=1,
                metrics=metrics,
                extra_fields=self._matpo_output_fields(
                    agent_name=planner_agent,
                    role="planner",
                    reqs_id=planner_delegate_req_id,
                    parent_reqs_id="",
                    is_child=False,
                    turn_count=0,
                    tool_name=tool_name,
                    sub_goal=search_query,
                    tool_result="",
                    turn_role="delegate",
                    tool_format_valid=True,
                    tool_call_count=1,
                    matpo_accuracy_reward=accuracy,
                    matpo_combined_reward=combined_reward,
                    prompt_text=planner_delegate_prompt,
                    response_text=planner_delegate_text.strip(),
                    observation_text=observation_text,
                    final_answer=final_answer,
                    workflow_success=workflow_success,
                ),
            ),
            AgentLoopOutput(
                prompt_ids=prompt_ids + planner_delegate_ids,
                response_ids=worker_call_ids,
                response_mask=[1] * len(worker_call_ids),
                reward_score=combined_reward,
                num_turns=2,
                metrics=metrics,
                extra_fields=self._matpo_output_fields(
                    agent_name=worker_agent,
                    role="worker",
                    reqs_id=worker_call_req_id,
                    parent_reqs_id=planner_delegate_req_id,
                    is_child=True,
                    turn_count=1,
                    tool_name=tool_name,
                    sub_goal=search_query,
                    tool_result="",
                    turn_role="worker_call",
                    tool_format_valid=True,
                    tool_call_count=1,
                    matpo_accuracy_reward=accuracy,
                    matpo_combined_reward=combined_reward,
                    prompt_text=worker_call_prompt,
                    response_text=worker_call_text.strip(),
                    observation_text=search_query,
                    final_answer=final_answer,
                    workflow_success=workflow_success,
                ),
            ),
            AgentLoopOutput(
                prompt_ids=prompt_ids + planner_delegate_ids + worker_call_ids + tool_observation_ids,
                response_ids=worker_summary_ids,
                response_mask=[1] * len(worker_summary_ids),
                reward_score=combined_reward,
                num_turns=3,
                metrics=metrics,
                extra_fields=self._matpo_output_fields(
                    agent_name=worker_agent,
                    role="worker",
                    reqs_id=worker_summary_req_id,
                    parent_reqs_id=planner_delegate_req_id,
                    is_child=True,
                    turn_count=2,
                    tool_name=tool_name,
                    sub_goal=search_query,
                    tool_result=worker_summary_text,
                    turn_role="worker_summary",
                    tool_format_valid=True,
                    tool_call_count=0,
                    tool_observation=tool_observation,
                    matpo_accuracy_reward=accuracy,
                    matpo_combined_reward=combined_reward,
                    prompt_text=worker_summary_prompt,
                    response_text=worker_summary_text.strip(),
                    observation_text=tool_observation,
                    final_answer=final_answer,
                    workflow_success=workflow_success,
                ),
            ),
            AgentLoopOutput(
                prompt_ids=(
                    prompt_ids + planner_delegate_ids + worker_call_ids + tool_observation_ids + worker_summary_ids
                ),
                response_ids=planner_final_ids,
                response_mask=[1] * len(planner_final_ids),
                reward_score=combined_reward,
                num_turns=4,
                metrics=metrics,
                extra_fields=self._matpo_output_fields(
                    agent_name=planner_agent,
                    role="planner",
                    reqs_id=planner_final_req_id,
                    parent_reqs_id="",
                    is_child=False,
                    turn_count=3,
                    tool_name=tool_name,
                    sub_goal=search_query,
                    tool_result=worker_summary_text,
                    turn_role="final",
                    tool_format_valid=True,
                    tool_call_count=0,
                    matpo_accuracy_reward=accuracy,
                    matpo_combined_reward=combined_reward,
                    prompt_text=planner_final_prompt,
                    response_text=planner_final_text.strip(),
                    observation_text=observation_text,
                    final_answer=final_answer,
                    workflow_success=workflow_success,
                ),
            ),
        ]

    def _build_hf_matpo_browse_outputs(self, prompt: dict[str, Any], *, session_id: int = 0) -> list[AgentLoopOutput]:
        from trajweave.backends.verl.workflow_runtime import build_hf_workflow_outputs

        return build_hf_workflow_outputs(
            self,
            recipe="matpo_browse",
            prompt=prompt,
            session_id=session_id,
        )

    def _matpo_orchestra_config(self) -> Any:
        agent_cfg = config_get(self.config, "agent", default={}) or {}
        orchestra_cfg = config_get(agent_cfg, "orchestra", default={}) or {}
        return config_get(orchestra_cfg, "matpo", default={}) or {}

    def _matpo_planner_agent(self) -> str:
        return str(config_get(self._matpo_orchestra_config(), "planner_agent", default="planner"))

    def _matpo_worker_agent(self) -> str:
        return str(config_get(self._matpo_orchestra_config(), "worker_agent", default="browsing_agent"))

    def _matpo_tool_name(self) -> str:
        return str(config_get(self._matpo_orchestra_config(), "tool_name", default="search_and_browse"))

    def _matpo_reward_weights(self) -> tuple[float, float]:
        matpo_cfg = self._matpo_orchestra_config()
        if config_get(matpo_cfg, "tool_format_reward_scale", default=None) is not None:
            raise ValueError("MATPO tool_format_reward_scale is no longer supported; use the explicit reward weights.")
        return (
            float(config_get(matpo_cfg, "accuracy_reward_weight", default=0.9)),
            float(config_get(matpo_cfg, "tool_format_reward_weight", default=0.1)),
        )

    @staticmethod
    def _matpo_request_prefix(prompt: dict[str, Any], *, raw_prompt: Any, session_id: int) -> str:
        prompt_id = str(to_python(prompt.get("uid", prompt.get("index", "prompt"))))
        serialized_prompt = json.dumps(raw_prompt, ensure_ascii=False, sort_keys=True, default=str)
        prompt_digest = hashlib.sha1(serialized_prompt.encode("utf-8")).hexdigest()[:12]
        return f"matpo:{prompt_id}:{prompt_digest}:session:{session_id}"

    @staticmethod
    def _matpo_observation_text(raw_prompt: Any) -> str:
        if isinstance(raw_prompt, list):
            contents = [str(item.get("content", "")) for item in raw_prompt if isinstance(item, dict)]
            text = "\n".join(content for content in contents if content)
            if text:
                return text
        text = str(raw_prompt).strip()
        return text or "MATPO browse task"

    @staticmethod
    def _matpo_output_fields(
        *,
        agent_name: str,
        role: str,
        reqs_id: str,
        parent_reqs_id: str,
        is_child: bool,
        turn_count: int,
        tool_name: str,
        sub_goal: str,
        tool_result: str,
        turn_role: str,
        tool_format_valid: bool,
        tool_call_count: int,
        tool_observation: str = "",
        **extra: Any,
    ) -> dict[str, Any]:
        return {
            "turn_scores": [],
            "tool_rewards": [],
            "trajweave_agent_name": agent_name,
            "trajweave_role": role,
            "agent_id": agent_name,
            "policy_group": "shared",
            "reqs_id": reqs_id,
            "parent_reqs_id": parent_reqs_id,
            "is_from_subagent_tool": is_child,
            "turn_count": turn_count,
            "agent_type": agent_name if is_child else "main_agent",
            "role_id": role,
            "shared_model_id": "shared",
            "tool_name": tool_name,
            "sub_goal": sub_goal,
            "tool_observation": tool_observation,
            "tool_result": tool_result,
            "matpo_tool_format_valid": tool_format_valid,
            "matpo_tool_call_count": tool_call_count,
            "matpo_turn_role": turn_role,
            **extra,
        }
