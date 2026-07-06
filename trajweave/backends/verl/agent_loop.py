from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import ray
import torch
import transfer_queue as tq
from tensordict import NonTensorData, NonTensorStack

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopMetrics,
    AgentLoopOutput,
    AgentLoopWorker,
    get_trajectory_info,
)
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ
from verl.utils.chat_template import apply_chat_template
from verl.utils.tensordict_utils import list_of_dict_to_tensordict

logger = logging.getLogger(__name__)


TRAJWEAVE_AGENT_LOOP_MANAGER_FQN = "trajweave.backends.verl.agent_loop.TrajWeaveAgentLoopManager"

DRMAS_AGENT_IDS = {
    "solver": "Solver Agent",
    "verifier": "Verifier Agent",
    "searcher": "Search Agent",
    "search": "Search Agent",
    "answer": "Answer Agent",
}


@dataclass(frozen=True)
class TrajWeaveAgentLoopRuntimeConfig:
    recipe: str | None = None
    config_path: str | None = None
    coordination_protocol: str | None = None
    trajectory_schema: str | None = None
    credit_allocator: str | None = None
    agent_loop_backend: str = "verl_tq"

    @classmethod
    def from_verl_config(cls, config: Any) -> "TrajWeaveAgentLoopRuntimeConfig":
        trajweave = _get(config, "trajweave", default={})
        return cls(
            recipe=_get(trajweave, "recipe"),
            config_path=_get(trajweave, "config"),
            coordination_protocol=_get(trajweave, "coordination_protocol"),
            trajectory_schema=_get(trajweave, "trajectory_schema"),
            credit_allocator=_get(trajweave, "credit_allocator"),
            agent_loop_backend=str(_get(trajweave, "agent_loop_backend", "verl_tq")),
        )

    def as_overrides(self) -> dict[str, str]:
        values = {
            "trajweave.recipe": self.recipe,
            "trajweave.config": self.config_path,
            "trajweave.coordination_protocol": self.coordination_protocol,
            "trajweave.trajectory_schema": self.trajectory_schema,
            "trajweave.credit_allocator": self.credit_allocator,
            "trajweave.agent_loop_backend": self.agent_loop_backend,
        }
        return {key: value for key, value in values.items() if value is not None}


class TrajWeaveAgentLoopManager(AgentLoopManagerTQ):
    """VERL V1 TransferQueue manager that carries TrajWeave MAS config metadata."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trajweave_runtime_config = TrajWeaveAgentLoopRuntimeConfig.from_verl_config(self.config)
        if self.trajweave_runtime_config.agent_loop_backend in {"synthetic_tq", "hf_local_tq"}:
            self.agent_loop_workers_class = TrajWeaveSyntheticAgentLoopWorkerTQ
        if self.trajweave_runtime_config.recipe:
            logger.info("TrajWeave AgentLoopManager loaded for %s", self.trajweave_runtime_config.recipe)

    def generate_sequences(self, prompts):
        self._validate_trajweave_runtime()
        return super().generate_sequences(prompts)

    def _validate_trajweave_runtime(self) -> None:
        _validate_agent_loop_backend(
            recipe=self.trajweave_runtime_config.recipe,
            backend=self.trajweave_runtime_config.agent_loop_backend,
        )


@ray.remote
class TrajWeaveSyntheticAgentLoopWorkerTQ(AgentLoopWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tq.init()

    async def generate_sequences(self, batch) -> None:
        validate = bool(_to_python(batch["validate"])) if "validate" in batch else False
        batch.pop("validate", None)

        batch_size = len(batch)
        if "index" in batch:
            index = [_to_python(_batch_item(batch["index"], i)) for i in range(batch_size)]
        else:
            index = list(range(batch_size))

        global_steps = _to_python(batch["global_steps"])
        if isinstance(global_steps, list):
            global_steps = global_steps[0] if global_steps else -1
        trajectory_info = await get_trajectory_info(global_steps, index, validate)

        tasks = []
        for i in range(batch_size):
            prompt = {key: _batch_item(value, i) for key, value in batch.items()}
            tasks.append(asyncio.create_task(self._run_prompt(prompt, trajectory=trajectory_info[i])))
        await asyncio.gather(*tasks)

    async def _run_prompt(self, prompt: dict[str, Any], trajectory: dict[str, Any]) -> None:
        uid = str(_to_python(prompt["uid"]))
        partition_id = "train" if not trajectory["validate"] else "val"
        await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "running"})
        try:
            config = self.config.actor_rollout_ref.rollout
            n = int(_to_python(prompt.pop("__rollout_n__", config.n if not trajectory["validate"] else config.val_kwargs.n)))
            runtime = TrajWeaveAgentLoopRuntimeConfig.from_verl_config(self.config)
            for session_id in range(n):
                use_hf_local = runtime.agent_loop_backend == "hf_local_tq"
                if runtime.recipe == "agentflow_planner_tool" and use_hf_local:
                    outputs = self._build_hf_agentflow_planner_tool_outputs(prompt, session_id=session_id)
                elif runtime.recipe == "agentflow_planner_tool":
                    outputs = self._build_agentflow_planner_tool_outputs(prompt, session_id=session_id)
                elif runtime.recipe == "maporl_debate_math" and use_hf_local:
                    outputs = self._build_hf_maporl_debate_math_outputs(prompt, session_id=session_id)
                elif runtime.recipe == "maporl_debate_math":
                    outputs = self._build_maporl_debate_math_outputs(prompt, session_id=session_id)
                elif runtime.recipe == "doctor_mas_search" and use_hf_local:
                    outputs = self._build_hf_search_answer_outputs(prompt, session_id=session_id)
                elif runtime.recipe == "doctor_mas_search":
                    outputs = self._build_search_answer_outputs(prompt, session_id=session_id)
                elif use_hf_local:
                    outputs = self._build_hf_solver_verifier_outputs(prompt, session_id=session_id)
                else:
                    outputs = self._build_solver_verifier_outputs(prompt, session_id=session_id)
                await self._put_outputs(outputs, validate=trajectory["validate"], session_id=session_id, **prompt)
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "finished"})
        except Exception:
            logger.exception("TrajWeave synthetic TQ worker failed for uid=%s", uid)
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "failure"})

    def _build_solver_verifier_outputs(self, prompt: dict[str, Any], *, session_id: int = 0) -> list[AgentLoopOutput]:
        raw_prompt = _to_python(prompt.get("raw_prompt", []))
        reward_model = _to_python(prompt.get("reward_model", {})) or {}
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

    def _build_maporl_debate_math_outputs(
        self,
        prompt: dict[str, Any],
        *,
        session_id: int = 0,
    ) -> list[AgentLoopOutput]:
        raw_prompt = _to_python(prompt.get("raw_prompt", []))
        reward_model = _to_python(prompt.get("reward_model", {})) or {}
        ground_truth = str(reward_model.get("ground_truth", "2"))
        prompt_ids = self._encode_prompt(raw_prompt)
        is_correct = session_id % 2 == 0
        reward = 1.0 if is_correct else 0.0
        agent_ids = self._maporl_agent_ids()
        model_ids = self._maporl_model_ids(default_agent_ids=agent_ids)
        max_rounds = self._maporl_max_rounds()
        consensus_threshold = self._maporl_consensus_threshold(default=len(agent_ids))
        early_stop = self._maporl_early_stop()
        reward_feedback = self._maporl_reward_feedback()
        consensus_answer = ground_truth if is_correct else "__wrong__"
        consensus_reached = bool(is_correct)
        finished_round = 0 if consensus_reached and early_stop else -1
        outputs: list[AgentLoopOutput] = []
        running_prompt_ids = list(prompt_ids)
        metrics = AgentLoopMetrics(generate_sequences=0.0, tool_calls=0.0, compute_score=0.0, num_preempted=-1)
        for round_id in range(max_rounds):
            round_answer = ground_truth if is_correct else "__wrong__"
            for agent_index, agent_id in enumerate(agent_ids):
                text = f" Agent {agent_id} round {round_id}. Final answer: {round_answer}"
                response_ids = self._encode_text(text)
                outputs.append(
                    AgentLoopOutput(
                        prompt_ids=list(running_prompt_ids),
                        response_ids=response_ids,
                        response_mask=[1] * len(response_ids),
                        reward_score=reward,
                        num_turns=round_id + 1,
                        metrics=metrics,
                        extra_fields={
                            "turn_scores": [],
                            "tool_rewards": [],
                            "trajweave_agent_name": agent_id,
                            "trajweave_role": "solver",
                            "agent_id": agent_id,
                            "policy_group": model_ids[agent_index],
                            "worker_group": model_ids[agent_index],
                            "worker_group_model_path": self._maporl_worker_group_model_path(model_ids[agent_index]),
                            "agent_answer": round_answer,
                            "raw_score": reward,
                            "correctness": reward,
                            "consensus_answer": consensus_answer,
                            "consensus_reached": consensus_reached,
                            "communication_graph": "fully_connected",
                            "aggregation": "consensus",
                            "reward_feedback": reward_feedback,
                            "round_id": round_id,
                            "agent_index": agent_index,
                            "consensus_threshold": consensus_threshold,
                            "finished_round": finished_round,
                        },
                    )
                )
                running_prompt_ids = running_prompt_ids + response_ids
            if consensus_reached and early_stop:
                break

        if outputs:
            outputs[-1].extra_fields["final_consensus_turn"] = True
        return outputs

    def _build_hf_maporl_debate_math_outputs(
        self,
        prompt: dict[str, Any],
        *,
        session_id: int = 0,
    ) -> list[AgentLoopOutput]:
        raw_prompt = _to_python(prompt.get("raw_prompt", []))
        prompt_ids = self._encode_prompt(raw_prompt)
        is_correct = session_id % 2 == 0
        reward = 1.0 if is_correct else 0.0
        agent_ids = self._maporl_agent_ids()
        model_ids = self._maporl_model_ids(default_agent_ids=agent_ids)
        max_rounds = self._maporl_max_rounds()
        consensus_threshold = self._maporl_consensus_threshold(default=len(agent_ids))
        early_stop = self._maporl_early_stop()
        reward_feedback = self._maporl_reward_feedback()
        reward_model = _to_python(prompt.get("reward_model", {})) or {}
        ground_truth = str(reward_model.get("ground_truth", "2"))
        round_answer = ground_truth if is_correct else "__wrong__"
        consensus_answer = ground_truth if is_correct else "__wrong__"
        consensus_reached = bool(is_correct)
        finished_round = 0 if consensus_reached and early_stop else -1
        outputs: list[AgentLoopOutput] = []
        running_prompt_ids = list(prompt_ids)
        metrics = AgentLoopMetrics(generate_sequences=1.0, tool_calls=0.0, compute_score=0.0, num_preempted=-1)
        for round_id in range(max_rounds):
            for agent_index, agent_id in enumerate(agent_ids):
                policy_group = model_ids[agent_index]
                response_ids = self._generate_local_response_ids(running_prompt_ids, policy_group=policy_group)
                outputs.append(
                    AgentLoopOutput(
                        prompt_ids=list(running_prompt_ids),
                        response_ids=response_ids,
                        response_mask=[1] * len(response_ids),
                        reward_score=reward,
                        num_turns=round_id + 1,
                        metrics=metrics,
                        extra_fields={
                            "turn_scores": [],
                            "tool_rewards": [],
                            "trajweave_agent_name": agent_id,
                            "trajweave_role": "solver",
                            "agent_id": agent_id,
                            "policy_group": policy_group,
                            "worker_group": policy_group,
                            "worker_group_model_path": self._maporl_worker_group_model_path(policy_group),
                            "agent_answer": round_answer,
                            "raw_score": reward,
                            "correctness": reward,
                            "consensus_answer": consensus_answer,
                            "consensus_reached": consensus_reached,
                            "communication_graph": "fully_connected",
                            "aggregation": "consensus",
                            "reward_feedback": reward_feedback,
                            "round_id": round_id,
                            "agent_index": agent_index,
                            "consensus_threshold": consensus_threshold,
                            "finished_round": finished_round,
                            "rollout_source": "hf_local_tq",
                        },
                    )
                )
                running_prompt_ids = running_prompt_ids + response_ids
            if consensus_reached and early_stop:
                break

        if outputs:
            outputs[-1].extra_fields["final_consensus_turn"] = True
        return outputs

    def _build_hf_solver_verifier_outputs(self, prompt: dict[str, Any], *, session_id: int = 0) -> list[AgentLoopOutput]:
        raw_prompt = _to_python(prompt.get("raw_prompt", []))
        prompt_ids = self._encode_prompt(raw_prompt)
        reward_model = _to_python(prompt.get("reward_model", {})) or {}
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
        raw_prompt = _to_python(prompt.get("raw_prompt", []))
        reward_model = _to_python(prompt.get("reward_model", {})) or {}
        extra_info = _to_python(prompt.get("extra_info", {})) or {}
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
        raw_prompt = _to_python(prompt.get("raw_prompt", []))
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

    def _build_agentflow_planner_tool_outputs(
        self,
        prompt: dict[str, Any],
        *,
        session_id: int = 0,
    ) -> list[AgentLoopOutput]:
        raw_prompt = _to_python(prompt.get("raw_prompt", []))
        prompt_ids = self._encode_prompt(raw_prompt)
        reward_model = _to_python(prompt.get("reward_model", {})) or {}
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
        raw_prompt = _to_python(prompt.get("raw_prompt", []))
        prompt_ids = self._encode_prompt(raw_prompt)
        reward_model = _to_python(prompt.get("reward_model", {})) or {}
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
                    },
                )
            )
            running_prompt_ids = running_prompt_ids + response_ids
            if verifier_decision == "STOP":
                break
        return outputs

    def _encode_prompt(self, raw_prompt: Any) -> list[int]:
        try:
            token_ids = apply_chat_template(
                self.tokenizer,
                raw_prompt,
                add_generation_prompt=True,
                tokenize=True,
                **self.config.data.get("apply_chat_template_kwargs", {}),
            )
        except Exception:
            token_ids = self._encode_text(str(raw_prompt))
        token_ids = _flatten_token_ids(token_ids)
        return token_ids[-self.rollout_config.prompt_length :]

    def _encode_text(self, text: str) -> list[int]:
        try:
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            token_ids = self.tokenizer.encode(text)
        token_ids = _flatten_token_ids(token_ids)
        if not token_ids:
            token_ids = [self.tokenizer.eos_token_id or self.tokenizer.pad_token_id or 0]
        return token_ids[: self.rollout_config.response_length]

    def _generate_local_response_ids(self, prompt_ids: list[int], *, policy_group: str = "shared") -> list[int]:
        model = self._local_model(policy_group=policy_group)
        tokenizer = self._local_tokenizer(policy_group=policy_group)
        device = next(model.parameters()).device
        input_ids = self._local_prompt_ids(prompt_ids, policy_group=policy_group, tokenizer=tokenizer)
        input_ids = torch.tensor([input_ids[-self.rollout_config.prompt_length :]], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            sequences = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.rollout_config.response_length,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id or 0,
                eos_token_id=tokenizer.eos_token_id,
            )
        group_response_ids = sequences[0, input_ids.shape[-1] :].detach().cpu().tolist()
        if not group_response_ids:
            return [self.tokenizer.eos_token_id or self.tokenizer.pad_token_id or 0]
        if tokenizer is self.tokenizer:
            response_ids = group_response_ids
        else:
            response_text = tokenizer.decode(group_response_ids, skip_special_tokens=True)
            response_ids = self._encode_text(response_text)
        return [int(token_id) for token_id in response_ids[: self.rollout_config.response_length]]

    def _local_model(self, *, policy_group: str = "shared"):
        if not hasattr(self, "_trajweave_local_models"):
            self._trajweave_local_models = {}
        cache: dict[str, Any] = self._trajweave_local_models
        if policy_group in cache:
            return cache[policy_group]
        from transformers import AutoModelForCausalLM

        model_path = self._maporl_worker_group_model_path(policy_group) or self.model_config.local_path
        kwargs: dict[str, Any] = {"trust_remote_code": self.model_config.trust_remote_code}
        if str(model_path) == str(self.model_config.local_path):
            kwargs["config"] = self.model_config.hf_config
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **kwargs,
        )
        model.eval()
        cache[policy_group] = model
        return cache[policy_group]

    def _local_tokenizer(self, *, policy_group: str = "shared"):
        tokenizer_path = self._maporl_worker_group_tokenizer_path(policy_group)
        model_path = self._maporl_worker_group_model_path(policy_group)
        if not tokenizer_path and not model_path:
            return self.tokenizer
        tokenizer_source = tokenizer_path or model_path
        if str(tokenizer_source) == str(self.model_config.local_path):
            return self.tokenizer
        if not hasattr(self, "_trajweave_local_tokenizers"):
            self._trajweave_local_tokenizers = {}
        cache: dict[str, Any] = self._trajweave_local_tokenizers
        cache_key = str(tokenizer_source)
        if cache_key in cache:
            return cache[cache_key]
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=self.model_config.trust_remote_code,
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        cache[cache_key] = tokenizer
        return tokenizer

    def _local_prompt_ids(self, prompt_ids: list[int], *, policy_group: str, tokenizer: Any) -> list[int]:
        if tokenizer is self.tokenizer:
            return prompt_ids
        try:
            prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
        except TypeError:
            prompt_text = self.tokenizer.decode(prompt_ids)
        try:
            ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        except TypeError:
            ids = tokenizer.encode(prompt_text)
        ids = _flatten_token_ids(ids)
        if not ids:
            ids = [tokenizer.eos_token_id or tokenizer.pad_token_id or 0]
        return ids

    def _maporl_agent_ids(self) -> list[str]:
        agent_cfg = _get(self.config, "agent", default={}) or {}
        ids = _get(agent_cfg, "agent_ids", default=None)
        ids = _to_python(ids)
        if ids:
            return [str(item) for item in ids]
        return ["agent_0", "agent_1"]

    def _maporl_model_ids(self, *, default_agent_ids: list[str]) -> list[str]:
        agent_cfg = _get(self.config, "agent", default={}) or {}
        ids = _get(agent_cfg, "model_ids", default=None)
        ids = _to_python(ids)
        if ids:
            model_ids = [str(item) for item in ids]
            if len(model_ids) != len(default_agent_ids):
                raise ValueError("agent.model_ids must have the same length as agent.agent_ids.")
            return model_ids
        return ["shared" for _ in default_agent_ids]

    def _maporl_worker_group_model_path(self, group_id: str) -> str | None:
        group_cfg = self._maporl_worker_group_config(group_id)
        model_path = _get(group_cfg, "model_path", default=None)
        if model_path is None:
            return None
        return str(_to_python(model_path))

    def _maporl_worker_group_tokenizer_path(self, group_id: str) -> str | None:
        group_cfg = self._maporl_worker_group_config(group_id)
        tokenizer_path = _get(group_cfg, "tokenizer_path", default=None)
        if tokenizer_path is None:
            return None
        return str(_to_python(tokenizer_path))

    def _maporl_worker_group_config(self, group_id: str) -> Any:
        agent_cfg = _get(self.config, "agent", default={}) or {}
        worker_groups = _get(agent_cfg, "worker_groups", default={}) or {}
        worker_groups = _to_python(worker_groups)
        if isinstance(worker_groups, dict):
            return worker_groups.get(group_id, {}) or {}
        if isinstance(worker_groups, list):
            for group in worker_groups:
                if isinstance(group, dict) and str(group.get("id")) == str(group_id):
                    return group
        return {}

    def _maporl_max_rounds(self) -> int:
        agent_cfg = _get(self.config, "agent", default={}) or {}
        orchestra_cfg = _get(agent_cfg, "orchestra", default={}) or {}
        maporl_cfg = _get(orchestra_cfg, "maporl", default={}) or {}
        return int(_get(maporl_cfg, "max_rounds", default=2))

    def _maporl_consensus_threshold(self, *, default: int) -> int:
        agent_cfg = _get(self.config, "agent", default={}) or {}
        orchestra_cfg = _get(agent_cfg, "orchestra", default={}) or {}
        maporl_cfg = _get(orchestra_cfg, "maporl", default={}) or {}
        return int(_get(maporl_cfg, "consensus_threshold", default=default))

    def _maporl_early_stop(self) -> bool:
        agent_cfg = _get(self.config, "agent", default={}) or {}
        orchestra_cfg = _get(agent_cfg, "orchestra", default={}) or {}
        maporl_cfg = _get(orchestra_cfg, "maporl", default={}) or {}
        return bool(_get(maporl_cfg, "early_stop", default=True))

    def _maporl_reward_feedback(self) -> bool:
        agent_cfg = _get(self.config, "agent", default={}) or {}
        orchestra_cfg = _get(agent_cfg, "orchestra", default={}) or {}
        maporl_cfg = _get(orchestra_cfg, "maporl", default={}) or {}
        return bool(_get(maporl_cfg, "reward_feedback", default=False))

    def _agentflow_max_steps(self) -> int:
        agent_cfg = _get(self.config, "agent", default={}) or {}
        orchestra_cfg = _get(agent_cfg, "orchestra", default={}) or {}
        agentflow_cfg = _get(orchestra_cfg, "agentflow", default={}) or {}
        return int(_get(agentflow_cfg, "max_steps", default=3))

    def _agentflow_enabled_tools(self) -> list[str]:
        agent_cfg = _get(self.config, "agent", default={}) or {}
        orchestra_cfg = _get(agent_cfg, "orchestra", default={}) or {}
        agentflow_cfg = _get(orchestra_cfg, "agentflow", default={}) or {}
        tools = _to_python(_get(agentflow_cfg, "enabled_tools", default=["base_generator"]))
        if not tools:
            return ["base_generator"]
        return [str(tool) for tool in tools]

    async def _put_outputs(self, outputs: list[AgentLoopOutput], validate: bool, **kwargs) -> None:
        final_output = outputs[-1]
        if final_output.reward_score is not None:
            for output in outputs[:-1]:
                if output.reward_score is None:
                    output.reward_score = final_output.reward_score
                output.extra_fields.setdefault("reward_extra_info", final_output.extra_fields.get("reward_extra_info", {}))

        uid, session_id = str(_to_python(kwargs["uid"])), int(kwargs["session_id"])
        keys, fields, tags = [], [], []
        for index, output in enumerate(outputs):
            prompt_ids = output.prompt_ids[-self.rollout_config.prompt_length :]
            response_ids = output.response_ids[: self.rollout_config.response_length]
            response_mask_ids = output.response_mask[: len(response_ids)]
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.tokenizer.eos_token_id or 0

            prompt_pad = self.rollout_config.prompt_length - len(prompt_ids)
            response_pad = self.rollout_config.response_length - len(response_ids)
            prompts = torch.tensor([pad_token_id] * prompt_pad + prompt_ids, dtype=torch.int64)
            responses = torch.tensor(response_ids + [pad_token_id] * response_pad, dtype=torch.int64)
            response_mask = torch.tensor(response_mask_ids + [0] * response_pad, dtype=torch.int64)
            input_ids = torch.cat([prompts, responses], dim=0)
            attention_mask = torch.tensor(
                [0] * prompt_pad
                + [1] * len(prompt_ids)
                + [1] * len(response_ids)
                + [0] * response_pad,
                dtype=torch.int64,
            )
            multi_modal_inputs = self._compute_multi_modal_inputs(output, input_ids)
            position_ids = self._compute_position_ids(
                input_ids.unsqueeze(0), attention_mask.unsqueeze(0), multi_modal_inputs
            ).squeeze(0)

            field = output.as_dict()
            field.update(kwargs)
            field["prompts"] = prompts
            field["responses"] = responses
            field["response_mask"] = response_mask
            if output.reward_score is not None:
                field["rm_scores"] = _padded_rm_scores(response_mask, float(output.reward_score), len(response_ids))
            if "rollout_log_probs" in field:
                field["rollout_log_probs"] = _pad_or_trim_1d(
                    field["rollout_log_probs"],
                    self.rollout_config.response_length,
                    pad_value=0.0,
                    dtype=torch.float32,
                )
            field["agent_name"] = output.extra_fields.get("trajweave_agent_name", field.get("agent_name"))
            field["role"] = output.extra_fields.get("trajweave_role", field["agent_name"])
            field["policy_group"] = output.extra_fields.get("policy_group", field.get("policy_group", field["agent_name"]))
            field["worker_group"] = output.extra_fields.get("worker_group", field["policy_group"])
            field["worker_group_model_path"] = output.extra_fields.get("worker_group_model_path") or ""
            field["agent_id"] = output.extra_fields.get("agent_id", _canonical_drmas_agent_id(field["agent_name"]))
            field["traj_uid"] = output.extra_fields.get("traj_uid", f"{uid}_{session_id}")
            field["turn_id"] = index
            for mas_field in (
                "round_id",
                "agent_index",
                "raw_score",
                "correctness",
                "consensus_reached",
                "finished_round",
                "agentflow_stage",
                "tool_name",
                "sub_goal",
                "tool_result",
                "verifier_decision",
                "memory_snapshot",
                "step_id",
            ):
                if mas_field in output.extra_fields:
                    field[mas_field] = output.extra_fields[mas_field]
            field["session_id"] = session_id
            field["loss_mask"] = field["response_mask"]
            field["input_ids"] = input_ids
            field["attention_mask"] = attention_mask
            field["position_ids"] = position_ids
            field["temperature"] = float(_get(self.rollout_config, "temperature", 1.0))
            field["multi_modal_inputs"] = multi_modal_inputs
            fields.append(field)
            keys.append(f"{uid}_{session_id}_{index}")
            prompt_len, response_len = field["prompts"].size(0), field["responses"].size(0)
            tags.append(
                {
                    "status": "success",
                    "prompt_len": prompt_len,
                    "response_len": response_len,
                    "seq_len": prompt_len + response_len,
                    "global_steps": _to_python(kwargs["global_steps"]),
                    "min_global_steps": field["extra_fields"].get("min_global_steps"),
                    "max_global_steps": field["extra_fields"].get("max_global_steps"),
                }
            )

        await tq.async_kv_batch_put(
            keys=keys,
            fields=list_of_dict_to_tensordict(fields),
            tags=tags,
            partition_id="train" if not validate else "val",
        )


def _get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    try:
        return config.get(key, default)
    except (AttributeError, TypeError):
        return getattr(config, key, default)


def _padded_rm_scores(response_mask: torch.Tensor, reward_score: float, response_len: int) -> torch.Tensor:
    rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
    if rm_scores.numel() == 0:
        return rm_scores
    valid_indices = torch.nonzero(response_mask, as_tuple=False).flatten()
    if valid_indices.numel() > 0:
        reward_index = int(valid_indices[-1].item())
    else:
        reward_index = max(0, min(response_len, rm_scores.numel()) - 1)
    rm_scores[reward_index] = reward_score
    return rm_scores


def _pad_or_trim_1d(
    values: torch.Tensor,
    target_len: int,
    *,
    pad_value: float | int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    output = values.to(dtype=dtype) if dtype is not None else values
    if output.size(0) > target_len:
        return output[:target_len]
    if output.size(0) == target_len:
        return output
    pad = torch.full(
        (target_len - output.size(0),),
        pad_value,
        dtype=output.dtype,
        device=output.device,
    )
    return torch.cat([output, pad], dim=0)


def _validate_agent_loop_backend(recipe: str | None, backend: str) -> None:
    if recipe and recipe not in {"doctor_mas_math", "doctor_mas_search", "maporl_debate_math", "agentflow_planner_tool"}:
        raise ValueError(f"Unsupported TrajWeave recipe for VERL AgentLoopManager: {recipe}")
    if backend not in {"verl_tq", "synthetic_tq", "hf_local_tq"}:
        raise ValueError(f"Unsupported TrajWeave AgentLoop backend: {backend}")
    if recipe and backend == "verl_tq":
        raise ValueError(
            "TrajWeave MASRL recipes require agent_loop_backend in {'synthetic_tq', 'hf_local_tq'}; "
            "VERL native verl_tq does not emit agent_id/traj_uid/turn_id metadata required by agent-wise credit."
        )


def _batch_item(value: Any, index: int) -> Any:
    if isinstance(value, torch.Tensor):
        return value[index]
    if isinstance(value, NonTensorStack):
        return value[index].data
    if isinstance(value, NonTensorData):
        return value.data
    if isinstance(value, (list, tuple)) and len(value) > index:
        return value[index]
    return value


def _to_python(value: Any) -> Any:
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf

        if isinstance(value, (DictConfig, ListConfig)):
            return OmegaConf.to_container(value, resolve=True)
    except Exception:
        pass
    if isinstance(value, NonTensorData):
        return _to_python(value.data)
    if isinstance(value, NonTensorStack):
        return [_to_python(item.data if hasattr(item, "data") else item) for item in value]
    if isinstance(value, dict):
        return {key: _to_python(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_python(item) for item in value)
    if isinstance(value, list):
        return [_to_python(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def _canonical_drmas_agent_id(agent_name: Any) -> str:
    name = str(_to_python(agent_name))
    return DRMAS_AGENT_IDS.get(name, name)


def _flatten_token_ids(token_ids: Any) -> list[int]:
    token_ids = _to_python(token_ids)
    if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]
