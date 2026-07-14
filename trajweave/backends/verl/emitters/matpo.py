from __future__ import annotations

from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput

from trajweave.backends.verl.schema import to_python


class MATPOEmitterMixin:
    def _build_matpo_browse_outputs(self, prompt: dict[str, Any], *, session_id: int = 0) -> list[AgentLoopOutput]:
        raw_prompt = to_python(prompt.get("raw_prompt", []))
        reward_model = to_python(prompt.get("reward_model", {})) or {}
        extra_info = to_python(prompt.get("extra_info", {})) or {}
        ground_truth = str(reward_model.get("ground_truth", "Paris"))
        search_query = str(extra_info.get("search_query", ground_truth))
        prompt_ids = self._encode_prompt(raw_prompt)
        is_correct = session_id % 2 == 0
        answer = ground_truth if is_correct else "__wrong__"
        reward = 1.0 if is_correct else 0.0
        parent_req_id = f"matpo-{session_id}-planner"
        child_req_id = f"matpo-{session_id}-worker-0"
        worker_text = f" Evidence summary: {search_query}. Suggested final answer: {answer}"
        planner_text = f" CALL search_and_browse: {search_query}\nFinal answer: {answer}"
        worker_ids = self._encode_text(worker_text)
        planner_ids = self._encode_text(planner_text)
        metrics = AgentLoopMetrics(generate_sequences=0.0, tool_calls=1.0, compute_score=0.0, num_preempted=-1)
        return [
            AgentLoopOutput(
                prompt_ids=prompt_ids + planner_ids[: max(1, len(planner_ids) // 2)],
                response_ids=worker_ids,
                response_mask=[1] * len(worker_ids),
                reward_score=reward,
                num_turns=1,
                metrics=metrics,
                extra_fields={
                    "turn_scores": [],
                    "tool_rewards": [],
                    "trajweave_agent_name": "browsing_agent",
                    "trajweave_role": "worker",
                    "agent_id": "browsing_agent",
                    "policy_group": "shared",
                    "reqs_id": child_req_id,
                    "parent_reqs_id": parent_req_id,
                    "is_from_subagent_tool": True,
                    "turn_count": 1,
                    "agent_type": "browsing_agent",
                    "role_id": "worker",
                    "shared_model_id": "shared",
                    "tool_name": "search_and_browse",
                    "sub_goal": search_query,
                    "tool_result": worker_text,
                    "matpo_tool_format_valid": True,
                    "matpo_tool_call_count": 0,
                },
            ),
            AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=planner_ids,
                response_mask=[1] * len(planner_ids),
                reward_score=reward,
                num_turns=2,
                metrics=metrics,
                extra_fields={
                    "turn_scores": [],
                    "tool_rewards": [],
                    "trajweave_agent_name": "planner",
                    "trajweave_role": "planner",
                    "agent_id": "planner",
                    "policy_group": "shared",
                    "reqs_id": parent_req_id,
                    "parent_reqs_id": "",
                    "is_from_subagent_tool": False,
                    "turn_count": 2,
                    "agent_type": "main_agent",
                    "role_id": "planner",
                    "shared_model_id": "shared",
                    "tool_name": "search_and_browse",
                    "sub_goal": search_query,
                    "tool_result": worker_text,
                    "matpo_tool_format_valid": True,
                    "matpo_tool_call_count": 1,
                },
            ),
        ]

    def _build_hf_matpo_browse_outputs(self, prompt: dict[str, Any], *, session_id: int = 0) -> list[AgentLoopOutput]:
        raw_prompt = to_python(prompt.get("raw_prompt", []))
        prompt_ids = self._encode_prompt(raw_prompt)
        reward = 1.0 if session_id % 2 == 0 else 0.0
        parent_req_id = f"matpo-{session_id}-planner"
        child_req_id = f"matpo-{session_id}-worker-0"
        worker_ids = self._generate_local_response_ids(prompt_ids, policy_group="shared")
        planner_ids = self._generate_local_response_ids(prompt_ids + worker_ids, policy_group="shared")
        metrics = AgentLoopMetrics(generate_sequences=1.0, tool_calls=1.0, compute_score=0.0, num_preempted=-1)
        return [
            AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=worker_ids,
                response_mask=[1] * len(worker_ids),
                reward_score=reward,
                num_turns=1,
                metrics=metrics,
                extra_fields={
                    "turn_scores": [],
                    "tool_rewards": [],
                    "trajweave_agent_name": "browsing_agent",
                    "trajweave_role": "worker",
                    "agent_id": "browsing_agent",
                    "policy_group": "shared",
                    "reqs_id": child_req_id,
                    "parent_reqs_id": parent_req_id,
                    "is_from_subagent_tool": True,
                    "turn_count": 1,
                    "agent_type": "browsing_agent",
                    "role_id": "worker",
                    "shared_model_id": "shared",
                    "rollout_source": "hf_local_tq",
                    "matpo_tool_format_valid": True,
                    "matpo_tool_call_count": 0,
                },
            ),
            AgentLoopOutput(
                prompt_ids=prompt_ids + worker_ids,
                response_ids=planner_ids,
                response_mask=[1] * len(planner_ids),
                reward_score=reward,
                num_turns=2,
                metrics=metrics,
                extra_fields={
                    "turn_scores": [],
                    "tool_rewards": [],
                    "trajweave_agent_name": "planner",
                    "trajweave_role": "planner",
                    "agent_id": "planner",
                    "policy_group": "shared",
                    "reqs_id": parent_req_id,
                    "parent_reqs_id": "",
                    "is_from_subagent_tool": False,
                    "turn_count": 2,
                    "agent_type": "main_agent",
                    "role_id": "planner",
                    "shared_model_id": "shared",
                    "rollout_source": "hf_local_tq",
                    "matpo_tool_format_valid": True,
                    "matpo_tool_call_count": 1,
                },
            ),
        ]
