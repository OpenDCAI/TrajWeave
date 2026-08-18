from __future__ import annotations

from typing import Any

from trajweave.backends.verl.runtime_config import config_get
from trajweave.backends.verl.schema import required_ground_truth, to_python
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput


class WideSeekR1EmitterMixin:
    def _build_wideseek_r1_outputs(self, prompt: dict[str, Any], *, session_id: int = 0) -> list[AgentLoopOutput]:
        raw_prompt = to_python(prompt.get("raw_prompt", []))
        extra_info = to_python(prompt.get("extra_info", {})) or {}
        answer = required_ground_truth(prompt)
        question = self._wideseek_observation(raw_prompt)
        query = str(extra_info.get("search_query") or question)
        lead = self._wideseek_lead_agent()
        workers = self._wideseek_subagent_ids()
        shared_group = self._wideseek_shared_model_id()
        correct = session_id % 2 == 0
        final_value = answer if correct else "__wrong__"
        reward = (1.0 if correct else 0.0) + self._wideseek_format_reward() + self._wideseek_search_reward()
        prompt_ids = self._encode_prompt(raw_prompt)
        metrics = AgentLoopMetrics(
            generate_sequences=float(2 + 3 * len(workers)),
            tool_calls=float(2 * len(workers)),
            compute_score=0.0,
            num_preempted=-1,
        )
        plan_text = "\n".join(f" CALL subagent: {query} focus {index + 1}" for index in range(len(workers)))
        plan_ids = self._encode_text(plan_text)
        outputs = [
            AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=plan_ids,
                response_mask=[1] * len(plan_ids),
                reward_score=reward,
                num_turns=1,
                metrics=metrics,
                extra_fields=self._wideseek_fields(
                    agent_name=lead,
                    role="lead",
                    shared_group=shared_group,
                    turn_role="lead_plan",
                    subtrajectory_id=0,
                    agent_count=1 + len(workers),
                    parallel_wave=0,
                    sub_goal=query,
                    prompt_text=question,
                    response_text=plan_text.strip(),
                    observation_text=question,
                    final_answer=f"Final answer: {final_value}",
                    workflow_success=correct,
                ),
            )
        ]
        for worker_index, worker in enumerate(workers, start=1):
            subtask = f"{query} focus {worker_index}"
            search_text = f" CALL search: {subtask}"
            search_evidence = f"Search evidence for {subtask}: {answer}"
            access_text = f" CALL access: {subtask}"
            access_evidence = f"Accessed evidence for {subtask}: {answer}"
            summary_text = f" Subagent result: {access_evidence}"
            search_ids = self._encode_text(search_text)
            access_ids = self._encode_text(access_text)
            summary_ids = self._encode_text(summary_text)
            isolated_prompt = prompt_ids + self._encode_text(f" Assigned subtask: {subtask}")
            common = {
                "agent_name": worker,
                "role": "subagent",
                "shared_group": shared_group,
                "subtrajectory_id": worker_index,
                "agent_count": 1 + len(workers),
                "parallel_wave": 1,
                "sub_goal": subtask,
                "final_answer": f"Final answer: {final_value}",
                "workflow_success": correct,
            }
            outputs.extend(
                [
                    AgentLoopOutput(
                        prompt_ids=isolated_prompt,
                        response_ids=search_ids,
                        response_mask=[1] * len(search_ids),
                        reward_score=reward,
                        num_turns=len(outputs) + 1,
                        metrics=metrics,
                        extra_fields=self._wideseek_fields(
                            **common,
                            turn_role="subagent_search",
                            tool_name="search",
                            tool_observation=search_evidence,
                            prompt_text=f"{question}\nSubtask: {subtask}",
                            response_text=search_text.strip(),
                            observation_text=subtask,
                        ),
                    ),
                    AgentLoopOutput(
                        prompt_ids=isolated_prompt + search_ids + self._encode_text(f" {search_evidence}"),
                        response_ids=access_ids,
                        response_mask=[1] * len(access_ids),
                        reward_score=reward,
                        num_turns=len(outputs) + 2,
                        metrics=metrics,
                        extra_fields=self._wideseek_fields(
                            **common,
                            turn_role="subagent_access",
                            tool_name="access",
                            tool_observation=access_evidence,
                            prompt_text=f"{question}\nSubtask: {subtask}\n{search_evidence}",
                            response_text=access_text.strip(),
                            observation_text=search_evidence,
                        ),
                    ),
                    AgentLoopOutput(
                        prompt_ids=isolated_prompt + access_ids + self._encode_text(f" {access_evidence}"),
                        response_ids=summary_ids,
                        response_mask=[1] * len(summary_ids),
                        reward_score=reward,
                        num_turns=len(outputs) + 3,
                        metrics=metrics,
                        extra_fields=self._wideseek_fields(
                            **common,
                            turn_role="subagent_summary",
                            tool_result=summary_text.strip(),
                            prompt_text=f"{question}\nSubtask: {subtask}\n{access_evidence}",
                            response_text=summary_text.strip(),
                            observation_text=access_evidence,
                        ),
                    ),
                ]
            )
        final_text = f" Final answer: {final_value}"
        final_ids = self._encode_text(final_text)
        outputs.append(
            AgentLoopOutput(
                prompt_ids=prompt_ids + plan_ids + self._encode_text(" " + " ".join(
                    str(item.extra_fields["response_text"])
                    for item in outputs
                    if item.extra_fields.get("wideseek_turn_role") == "subagent_summary"
                )),
                response_ids=final_ids,
                response_mask=[1] * len(final_ids),
                reward_score=reward,
                num_turns=len(outputs) + 1,
                metrics=metrics,
                extra_fields=self._wideseek_fields(
                    agent_name=lead,
                    role="lead",
                    shared_group=shared_group,
                    turn_role="lead_final",
                    subtrajectory_id=0,
                    agent_count=1 + len(workers),
                    parallel_wave=2,
                    sub_goal="",
                    prompt_text=question,
                    response_text=final_text.strip(),
                    observation_text=question,
                    final_answer=final_text.strip(),
                    workflow_success=correct,
                ),
            )
        )
        return outputs

    def _build_hf_wideseek_r1_outputs(
        self, prompt: dict[str, Any], *, session_id: int = 0
    ) -> list[AgentLoopOutput]:
        from trajweave.backends.verl.workflow_runtime import build_hf_workflow_outputs

        return build_hf_workflow_outputs(self, recipe="wideseek_r1_broad_search", prompt=prompt, session_id=session_id)

    def _wideseek_orchestra_config(self) -> Any:
        agent = config_get(self.config, "agent", default={}) or {}
        orchestra = config_get(agent, "orchestra", default={}) or {}
        return config_get(orchestra, "wideseek_r1", default={}) or {}

    def _wideseek_lead_agent(self) -> str:
        return str(config_get(self._wideseek_orchestra_config(), "lead_agent", default="lead_agent"))

    def _wideseek_max_subagents(self) -> int:
        return int(config_get(self._wideseek_orchestra_config(), "max_parallel_subagents", default=3))

    def _wideseek_subagent_prefix(self) -> str:
        return str(config_get(self._wideseek_orchestra_config(), "subagent_prefix", default="subagent_"))

    def _wideseek_subagent_ids(self) -> tuple[str, ...]:
        agent = config_get(self.config, "agent", default={}) or {}
        values = [str(value) for value in (to_python(config_get(agent, "agent_ids", default=[])) or [])]
        lead = self._wideseek_lead_agent()
        workers = tuple(value for value in values if value != lead)
        if len(workers) != self._wideseek_max_subagents():
            raise ValueError("WideSeek-R1 AgentLoop agent_ids do not match max_parallel_subagents.")
        return workers

    def _wideseek_shared_model_id(self) -> str:
        return str(config_get(self._wideseek_orchestra_config(), "shared_model_id", default="shared_policy"))

    def _wideseek_format_reward(self) -> float:
        return float(config_get(self._wideseek_orchestra_config(), "format_reward", default=0.1))

    def _wideseek_search_reward(self) -> float:
        return float(config_get(self._wideseek_orchestra_config(), "search_reward", default=0.05))

    @staticmethod
    def _wideseek_observation(raw_prompt: Any) -> str:
        if isinstance(raw_prompt, list):
            text = "\n".join(
                str(item.get("content", "")) for item in raw_prompt if isinstance(item, dict) and item.get("content")
            )
            if text:
                return text
        return str(raw_prompt).strip() or "WideSeek-R1 broad information-seeking task"

    @staticmethod
    def _wideseek_fields(
        *,
        agent_name: str,
        role: str,
        shared_group: str,
        turn_role: str,
        subtrajectory_id: int,
        agent_count: int,
        parallel_wave: int,
        sub_goal: str,
        tool_name: str = "",
        tool_observation: str = "",
        tool_result: str = "",
        **extra: Any,
    ) -> dict[str, Any]:
        return {
            "turn_scores": [],
            "tool_rewards": [],
            "trajweave_agent_name": agent_name,
            "trajweave_role": role,
            "agent_id": agent_name,
            "policy_group": shared_group,
            "worker_group": shared_group,
            "wideseek_turn_role": turn_role,
            "wideseek_agent_instance": agent_name,
            "wideseek_subtrajectory_id": subtrajectory_id,
            "wideseek_agent_count": agent_count,
            "wideseek_parallel_wave": parallel_wave,
            "wideseek_format_valid": True,
            "tool_name": tool_name,
            "sub_goal": sub_goal,
            "tool_observation": tool_observation,
            "tool_result": tool_result,
            **extra,
        }
