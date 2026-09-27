from __future__ import annotations

from typing import Any

from trajweave.backends.verl.runtime_config import config_get
from trajweave.backends.verl.schema import required_ground_truth, to_python
from trajweave.credit.mrlx import DEFAULT_ADAPTER_FORMAT_BONUS, DEFAULT_EXPLORER_FORMAT_BONUS
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput


class MrlXEmitterMixin:
    def _build_mrlx_research_outputs(self, prompt: dict[str, Any], *, session_id: int = 0) -> list[AgentLoopOutput]:
        raw_prompt = to_python(prompt.get("raw_prompt", []))
        extra_info = to_python(prompt.get("extra_info", {})) or {}
        answer = required_ground_truth(prompt)
        search_query = str(extra_info.get("search_query") or answer)
        explorer_agent, adapter_agent = self._mrlx_agent_ids()
        explorer_group, adapter_group = self._mrlx_model_ids()
        tool_name = self._mrlx_tool_name()
        correct = session_id % 2 == 0
        final_answer = answer if correct else "__wrong__"
        explorer_reward = 1.0 if correct else self._mrlx_explorer_format_bonus()
        adapter_reward = 1.0 if correct else self._mrlx_adapter_format_bonus()
        prompt_ids = self._encode_prompt(raw_prompt)
        observation = self._mrlx_observation(raw_prompt)
        metrics = AgentLoopMetrics(generate_sequences=4.0, tool_calls=1.0, compute_score=0.0, num_preempted=-1)

        delegate_text = f" CALL {adapter_agent}: {search_query}"
        adapter_tool_text = f" CALL {tool_name}: {search_query}"
        evidence = f"Evidence for {search_query}: {answer}"
        adapter_result_text = f" Research result: {evidence}"
        final_text = f" Final answer: {final_answer}"
        delegate_ids = self._encode_text(delegate_text)
        adapter_tool_ids = self._encode_text(adapter_tool_text)
        evidence_ids = self._encode_text(f" {evidence}")
        adapter_result_ids = self._encode_text(adapter_result_text)
        final_ids = self._encode_text(final_text)

        return [
            AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=delegate_ids,
                response_mask=[1] * len(delegate_ids),
                reward_score=explorer_reward,
                num_turns=1,
                metrics=metrics,
                extra_fields=self._mrlx_fields(
                    agent_name=explorer_agent,
                    role="main_explorer",
                    policy_group=explorer_group,
                    turn_role="delegate",
                    training_mode="on_policy_explorer",
                    policy_lag=0,
                    tool_name=tool_name,
                    sub_goal=search_query,
                    prompt_text=observation,
                    response_text=delegate_text.strip(),
                    observation_text=observation,
                    workflow_success=correct,
                    final_answer=final_text.strip(),
                ),
            ),
            AgentLoopOutput(
                prompt_ids=prompt_ids + delegate_ids,
                response_ids=adapter_tool_ids,
                response_mask=[1] * len(adapter_tool_ids),
                reward_score=adapter_reward,
                num_turns=2,
                metrics=metrics,
                extra_fields=self._mrlx_fields(
                    agent_name=adapter_agent,
                    role="sub_adapter",
                    policy_group=adapter_group,
                    turn_role="adapter_tool",
                    training_mode="off_policy_adapter",
                    policy_lag=1,
                    tool_name=tool_name,
                    sub_goal=search_query,
                    tool_observation=evidence,
                    prompt_text=f"{observation}\n{delegate_text.strip()}",
                    response_text=adapter_tool_text.strip(),
                    observation_text=search_query,
                    workflow_success=correct,
                    final_answer=final_text.strip(),
                ),
            ),
            AgentLoopOutput(
                prompt_ids=prompt_ids + delegate_ids + adapter_tool_ids + evidence_ids,
                response_ids=adapter_result_ids,
                response_mask=[1] * len(adapter_result_ids),
                reward_score=adapter_reward,
                num_turns=3,
                metrics=metrics,
                extra_fields=self._mrlx_fields(
                    agent_name=adapter_agent,
                    role="sub_adapter",
                    policy_group=adapter_group,
                    turn_role="adapter_result",
                    training_mode="off_policy_adapter",
                    policy_lag=1,
                    tool_name=tool_name,
                    sub_goal=search_query,
                    tool_observation=evidence,
                    tool_result=adapter_result_text.strip(),
                    prompt_text=f"{observation}\n{delegate_text.strip()}\n{adapter_tool_text.strip()}\n{evidence}",
                    response_text=adapter_result_text.strip(),
                    observation_text=evidence,
                    workflow_success=correct,
                    final_answer=final_text.strip(),
                ),
            ),
            AgentLoopOutput(
                prompt_ids=prompt_ids + delegate_ids + adapter_result_ids,
                response_ids=final_ids,
                response_mask=[1] * len(final_ids),
                reward_score=explorer_reward,
                num_turns=4,
                metrics=metrics,
                extra_fields=self._mrlx_fields(
                    agent_name=explorer_agent,
                    role="main_explorer",
                    policy_group=explorer_group,
                    turn_role="final",
                    training_mode="on_policy_explorer",
                    policy_lag=0,
                    tool_name=tool_name,
                    sub_goal="",
                    tool_result=adapter_result_text.strip(),
                    prompt_text=(
                        f"{observation}\n{delegate_text.strip()}\n{adapter_result_text.strip()}"
                    ),
                    response_text=final_text.strip(),
                    observation_text=observation,
                    workflow_success=correct,
                    final_answer=final_text.strip(),
                ),
            ),
        ]

    def _build_hf_mrlx_research_outputs(
        self, prompt: dict[str, Any], *, session_id: int = 0
    ) -> list[AgentLoopOutput]:
        from trajweave.backends.verl.workflow_runtime import build_hf_workflow_outputs

        return build_hf_workflow_outputs(self, recipe="mrlx_research_qa", prompt=prompt, session_id=session_id)

    def _mrlx_orchestra_config(self) -> Any:
        agent = config_get(self.config, "agent", default={}) or {}
        orchestra = config_get(agent, "orchestra", default={}) or {}
        return config_get(orchestra, "mrlx", default={}) or {}

    def _mrlx_agent_ids(self) -> tuple[str, str]:
        agent = config_get(self.config, "agent", default={}) or {}
        values = to_python(config_get(agent, "agent_ids", default=[])) or []
        if len(values) != 2:
            raise ValueError("MrlX AgentLoop requires exactly two agent_ids.")
        return str(values[0]), str(values[1])

    def _mrlx_model_ids(self) -> tuple[str, str]:
        agent = config_get(self.config, "agent", default={}) or {}
        values = to_python(config_get(agent, "model_ids", default=[])) or []
        if len(values) != 2 or str(values[0]) == str(values[1]):
            raise ValueError("MrlX AgentLoop requires two distinct model_ids.")
        return str(values[0]), str(values[1])

    def _mrlx_tool_name(self) -> str:
        return str(config_get(self._mrlx_orchestra_config(), "tool_name", default="search_and_browse"))

    def _mrlx_adapter_format_bonus(self) -> float:
        return float(
            config_get(
                self._mrlx_orchestra_config(),
                "adapter_format_bonus",
                default=DEFAULT_ADAPTER_FORMAT_BONUS,
            )
        )

    def _mrlx_explorer_format_bonus(self) -> float:
        return float(
            config_get(
                self._mrlx_orchestra_config(),
                "explorer_format_bonus",
                default=DEFAULT_EXPLORER_FORMAT_BONUS,
            )
        )

    @staticmethod
    def _mrlx_observation(raw_prompt: Any) -> str:
        if isinstance(raw_prompt, list):
            text = "\n".join(
                str(item.get("content", "")) for item in raw_prompt if isinstance(item, dict) and item.get("content")
            )
            if text:
                return text
        return str(raw_prompt).strip() or "MrlX research task"

    @staticmethod
    def _mrlx_fields(
        *,
        agent_name: str,
        role: str,
        policy_group: str,
        turn_role: str,
        training_mode: str,
        policy_lag: int,
        tool_name: str,
        sub_goal: str,
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
            "policy_group": policy_group,
            "worker_group": policy_group,
            "tool_name": tool_name,
            "sub_goal": sub_goal,
            "tool_observation": tool_observation,
            "tool_result": tool_result,
            "mrlx_turn_role": turn_role,
            "mrlx_training_mode": training_mode,
            "mrlx_policy_lag": policy_lag,
            "mrlx_format_valid": True,
            **extra,
        }
