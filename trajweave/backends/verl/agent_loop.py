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
        recipe = self.trajweave_runtime_config.recipe
        if recipe and recipe not in {"doctor_mas_math", "doctor_mas_search"}:
            raise ValueError(f"Unsupported TrajWeave recipe for VERL AgentLoopManager: {recipe}")
        backend = self.trajweave_runtime_config.agent_loop_backend
        if backend not in {"verl_tq", "synthetic_tq", "hf_local_tq"}:
            raise ValueError(f"Unsupported TrajWeave AgentLoop backend: {backend}")


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
                if runtime.recipe == "doctor_mas_search" and use_hf_local:
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

    def _generate_local_response_ids(self, prompt_ids: list[int]) -> list[int]:
        model = self._local_model()
        device = next(model.parameters()).device
        input_ids = torch.tensor([prompt_ids[-self.rollout_config.prompt_length :]], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            sequences = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.rollout_config.response_length,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        response_ids = sequences[0, input_ids.shape[-1] :].detach().cpu().tolist()
        if not response_ids:
            response_ids = [self.tokenizer.eos_token_id or self.tokenizer.pad_token_id or 0]
        return [int(token_id) for token_id in response_ids[: self.rollout_config.response_length]]

    def _local_model(self):
        if hasattr(self, "_trajweave_local_model"):
            return self._trajweave_local_model
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            self.model_config.local_path,
            config=self.model_config.hf_config,
            trust_remote_code=self.model_config.trust_remote_code,
        )
        model.eval()
        self._trajweave_local_model = model
        return self._trajweave_local_model

    async def _put_outputs(self, outputs: list[AgentLoopOutput], validate: bool, **kwargs) -> None:
        final_output = outputs[-1]
        if final_output.reward_score is not None:
            for output in outputs[:-1]:
                output.reward_score = final_output.reward_score
                output.extra_fields["reward_extra_info"] = final_output.extra_fields.get("reward_extra_info", {})

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
            field["agent_name"] = output.extra_fields.get("trajweave_agent_name", field.get("agent_name"))
            field["role"] = output.extra_fields.get("trajweave_role", field["agent_name"])
            field["agent_id"] = output.extra_fields.get("agent_id", _canonical_drmas_agent_id(field["agent_name"]))
            field["traj_uid"] = output.extra_fields.get("traj_uid", f"{uid}_{session_id}")
            field["turn_id"] = index
            field["session_id"] = session_id
            field["loss_mask"] = field["response_mask"]
            field["input_ids"] = input_ids
            field["attention_mask"] = attention_mask
            field["position_ids"] = position_ids
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
