from __future__ import annotations

from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput

from trajweave.backends.verl.schema import to_python


class DrMASEmitterMixin:
    def _build_solver_verifier_outputs(self, prompt: dict[str, Any], *, session_id: int = 0) -> list[AgentLoopOutput]:
        raw_prompt = to_python(prompt.get("raw_prompt", []))
        reward_model = to_python(prompt.get("reward_model", {})) or {}
        ground_truth = str(reward_model.get("ground_truth", "2"))
        prompt_ids = self._encode_prompt(raw_prompt)
        is_correct = session_id % 2 == 0
        solver_answer = ground_truth if is_correct else "__wrong__"
        solver_reward = 1.0 if is_correct else 0.0
        solver_text = f" Final answer: {solver_answer}"
        verifier_text = " APPROVED"
        solver_ids = self._encode_text(solver_text)
        verifier_ids = self._encode_text(verifier_text)
        metrics = AgentLoopMetrics(generate_sequences=0.0, tool_calls=0.0, compute_score=0.0, num_preempted=-1)
        return [
            AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=solver_ids,
                response_mask=[1] * len(solver_ids),
                reward_score=None,
                num_turns=1,
                metrics=metrics,
                extra_fields={
                    "turn_scores": [],
                    "tool_rewards": [],
                    "trajweave_agent_name": "solver",
                    "trajweave_role": "solver",
                },
            ),
            AgentLoopOutput(
                prompt_ids=prompt_ids + solver_ids,
                response_ids=verifier_ids,
                response_mask=[1] * len(verifier_ids),
                reward_score=solver_reward,
                num_turns=2,
                metrics=metrics,
                extra_fields={
                    "turn_scores": [],
                    "tool_rewards": [],
                    "trajweave_agent_name": "verifier",
                    "trajweave_role": "verifier",
                },
            ),
        ]

    def _build_hf_solver_verifier_outputs(self, prompt: dict[str, Any], *, session_id: int = 0) -> list[AgentLoopOutput]:
        raw_prompt = to_python(prompt.get("raw_prompt", []))
        prompt_ids = self._encode_prompt(raw_prompt)
        reward_model = to_python(prompt.get("reward_model", {})) or {}
        is_correct = session_id % 2 == 0
        reward = 1.0 if is_correct else 0.0
        solver_ids = self._generate_local_response_ids(prompt_ids)
        verifier_ids = self._generate_local_response_ids(prompt_ids + solver_ids)
        metrics = AgentLoopMetrics(generate_sequences=1.0, tool_calls=0.0, compute_score=0.0, num_preempted=-1)
        return [
            AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=solver_ids,
                response_mask=[1] * len(solver_ids),
                reward_score=None,
                num_turns=1,
                metrics=metrics,
                extra_fields={
                    "turn_scores": [],
                    "tool_rewards": [],
                    "trajweave_agent_name": "solver",
                    "trajweave_role": "solver",
                    "ground_truth": str(reward_model.get("ground_truth", "")),
                    "rollout_source": "hf_local_tq",
                },
            ),
            AgentLoopOutput(
                prompt_ids=prompt_ids + solver_ids,
                response_ids=verifier_ids,
                response_mask=[1] * len(verifier_ids),
                reward_score=reward,
                num_turns=2,
                metrics=metrics,
                extra_fields={
                    "turn_scores": [],
                    "tool_rewards": [],
                    "trajweave_agent_name": "verifier",
                    "trajweave_role": "verifier",
                    "rollout_source": "hf_local_tq",
                },
            ),
        ]

    def _build_search_answer_outputs(self, prompt: dict[str, Any], *, session_id: int = 0) -> list[AgentLoopOutput]:
        raw_prompt = to_python(prompt.get("raw_prompt", []))
        reward_model = to_python(prompt.get("reward_model", {})) or {}
        extra_info = to_python(prompt.get("extra_info", {})) or {}
        ground_truth = str(reward_model.get("ground_truth", "Paris"))
        search_query = str(extra_info.get("search_query", ground_truth))
        prompt_ids = self._encode_prompt(raw_prompt)
        is_correct = session_id % 2 == 0
        answer_text = ground_truth if is_correct else "__wrong__"
        reward = 1.0 if is_correct else 0.0
        verifier_ids = self._encode_text(" SEARCH: evidence is missing.")
        searcher_ids = self._encode_text(f" SEARCH: {search_query}")
        answer_ids = self._encode_text(f" Final answer: {answer_text}")
        metrics = AgentLoopMetrics(generate_sequences=0.0, tool_calls=1.0, compute_score=0.0, num_preempted=-1)
        return [
            AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=verifier_ids,
                response_mask=[1] * len(verifier_ids),
                reward_score=None,
                num_turns=1,
                metrics=metrics,
                extra_fields={
                    "turn_scores": [],
                    "tool_rewards": [],
                    "trajweave_agent_name": "verifier",
                    "trajweave_role": "verifier",
                },
            ),
            AgentLoopOutput(
                prompt_ids=prompt_ids + verifier_ids,
                response_ids=searcher_ids,
                response_mask=[1] * len(searcher_ids),
                reward_score=None,
                num_turns=2,
                metrics=metrics,
                extra_fields={
                    "turn_scores": [],
                    "tool_rewards": [],
                    "trajweave_agent_name": "searcher",
                    "trajweave_role": "searcher",
                },
            ),
            AgentLoopOutput(
                prompt_ids=prompt_ids + verifier_ids + searcher_ids,
                response_ids=answer_ids,
                response_mask=[1] * len(answer_ids),
                reward_score=reward,
                num_turns=3,
                metrics=metrics,
                extra_fields={
                    "turn_scores": [],
                    "tool_rewards": [],
                    "trajweave_agent_name": "answer",
                    "trajweave_role": "answer",
                },
            ),
        ]

    def _build_hf_search_answer_outputs(self, prompt: dict[str, Any], *, session_id: int = 0) -> list[AgentLoopOutput]:
        raw_prompt = to_python(prompt.get("raw_prompt", []))
        prompt_ids = self._encode_prompt(raw_prompt)
        is_correct = session_id % 2 == 0
        reward = 1.0 if is_correct else 0.0
        verifier_ids = self._generate_local_response_ids(prompt_ids)
        searcher_ids = self._generate_local_response_ids(prompt_ids + verifier_ids)
        answer_ids = self._generate_local_response_ids(prompt_ids + verifier_ids + searcher_ids)
        metrics = AgentLoopMetrics(generate_sequences=1.0, tool_calls=1.0, compute_score=0.0, num_preempted=-1)
        return [
            AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=verifier_ids,
                response_mask=[1] * len(verifier_ids),
                reward_score=None,
                num_turns=1,
                metrics=metrics,
                extra_fields={
                    "turn_scores": [],
                    "tool_rewards": [],
                    "trajweave_agent_name": "verifier",
                    "trajweave_role": "verifier",
                    "rollout_source": "hf_local_tq",
                },
            ),
            AgentLoopOutput(
                prompt_ids=prompt_ids + verifier_ids,
                response_ids=searcher_ids,
                response_mask=[1] * len(searcher_ids),
                reward_score=None,
                num_turns=2,
                metrics=metrics,
                extra_fields={
                    "turn_scores": [],
                    "tool_rewards": [],
                    "trajweave_agent_name": "searcher",
                    "trajweave_role": "searcher",
                    "rollout_source": "hf_local_tq",
                },
            ),
            AgentLoopOutput(
                prompt_ids=prompt_ids + verifier_ids + searcher_ids,
                response_ids=answer_ids,
                response_mask=[1] * len(answer_ids),
                reward_score=reward,
                num_turns=3,
                metrics=metrics,
                extra_fields={
                    "turn_scores": [],
                    "tool_rewards": [],
                    "trajweave_agent_name": "answer",
                    "trajweave_role": "answer",
                    "rollout_source": "hf_local_tq",
                },
            ),
        ]
