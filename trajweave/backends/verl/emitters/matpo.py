from __future__ import annotations

from typing import Any

from trajweave.backends.verl.runtime_config import config_get
from trajweave.backends.verl.schema import to_python
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput


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
        planner_agent = self._matpo_planner_agent()
        worker_agent = self._matpo_worker_agent()
        tool_name = self._matpo_tool_name()
        parent_req_id = f"matpo-{session_id}-planner"
        child_req_id = f"matpo-{session_id}-worker-0"
        worker_text = f" Evidence summary: {search_query}. Suggested final answer: {answer}"
        planner_text = f" CALL {tool_name}: {search_query}\nFinal answer: {answer}"
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
                    "trajweave_agent_name": worker_agent,
                    "trajweave_role": "worker",
                    "agent_id": worker_agent,
                    "policy_group": "shared",
                    "reqs_id": child_req_id,
                    "parent_reqs_id": parent_req_id,
                    "is_from_subagent_tool": True,
                    "turn_count": 1,
                    "agent_type": worker_agent,
                    "role_id": "worker",
                    "shared_model_id": "shared",
                    "tool_name": tool_name,
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
                    "trajweave_agent_name": planner_agent,
                    "trajweave_role": "planner",
                    "agent_id": planner_agent,
                    "policy_group": "shared",
                    "reqs_id": parent_req_id,
                    "parent_reqs_id": "",
                    "is_from_subagent_tool": False,
                    "turn_count": 2,
                    "agent_type": "main_agent",
                    "role_id": "planner",
                    "shared_model_id": "shared",
                    "tool_name": tool_name,
                    "sub_goal": search_query,
                    "tool_result": worker_text,
                    "matpo_tool_format_valid": True,
                    "matpo_tool_call_count": 1,
                },
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
