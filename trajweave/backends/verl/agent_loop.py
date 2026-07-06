from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

import ray
import torch
import transfer_queue as tq

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopMetrics,
    AgentLoopOutput,
    AgentLoopWorker,
    get_trajectory_info,
)
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ
from verl.utils.tensordict_utils import list_of_dict_to_tensordict

from trajweave.backends.verl.emitters import AgentFlowEmitterMixin, DrMASEmitterMixin, MAPoRLEmitterMixin
from trajweave.backends.verl.local_generation import HFLocalGenerationMixin
from trajweave.backends.verl.runtime_config import (
    TrajWeaveAgentLoopRuntimeConfig,
    config_get as _get,
    validate_agent_loop_backend as _validate_agent_loop_backend,
)
from trajweave.backends.verl.schema import (
    MAS_EXTRA_FIELDS,
    batch_item as _batch_item,
    canonical_drmas_agent_id as _canonical_drmas_agent_id,
    flatten_token_ids as _flatten_token_ids,
    pad_or_trim_1d as _pad_or_trim_1d,
    padded_rm_scores as _padded_rm_scores,
    to_python as _to_python,
)
from trajweave.storage.jsonl import JsonlWriter

logger = logging.getLogger(__name__)


TRAJWEAVE_AGENT_LOOP_MANAGER_FQN = "trajweave.backends.verl.agent_loop.TrajWeaveAgentLoopManager"


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
class TrajWeaveSyntheticAgentLoopWorkerTQ(
    AgentFlowEmitterMixin,
    MAPoRLEmitterMixin,
    DrMASEmitterMixin,
    HFLocalGenerationMixin,
    AgentLoopWorker,
):
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

    async def _put_outputs(self, outputs: list[AgentLoopOutput], validate: bool, **kwargs) -> None:
        runtime = TrajWeaveAgentLoopRuntimeConfig.from_verl_config(self.config)
        final_output = outputs[-1]
        if final_output.reward_score is not None:
            for output in outputs[:-1]:
                if output.reward_score is None:
                    output.reward_score = final_output.reward_score
                output.extra_fields.setdefault("reward_extra_info", final_output.extra_fields.get("reward_extra_info", {}))

        uid, session_id = str(_to_python(kwargs["uid"])), int(kwargs["session_id"])
        keys, fields, tags = [], [], []
        online_turn_rows: list[dict[str, Any]] = []
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
            for mas_field in MAS_EXTRA_FIELDS:
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
            online_turn_rows.append(
                {
                    "recipe": runtime.recipe,
                    "uid": uid,
                    "session_id": session_id,
                    "turn_id": index,
                    "validate": validate,
                    "agent_name": _to_python(field["agent_name"]),
                    "role": _to_python(field["role"]),
                    "policy_group": _to_python(field["policy_group"]),
                    "worker_group": _to_python(field["worker_group"]),
                    "agent_id": _to_python(field["agent_id"]),
                    "traj_uid": _to_python(field["traj_uid"]),
                    "reward_score": _to_python(output.reward_score),
                    "prompt_len": prompt_len,
                    "response_len": response_len,
                    "global_steps": _to_python(kwargs["global_steps"]),
                    "metadata": {
                        key: _to_python(field[key])
                        for key in MAS_EXTRA_FIELDS
                        if key in field
                    },
                }
            )
            for trace_idx, trace_event in enumerate(output.extra_fields.get("agentflow_trace", [])):
                trace_event = _to_python(trace_event)
                online_turn_rows.append(
                    {
                        "recipe": runtime.recipe,
                        "uid": uid,
                        "session_id": session_id,
                        "turn_id": f"{index}.{trace_idx + 1}",
                        "validate": validate,
                        "agent_name": trace_event.get("agent_name"),
                        "role": trace_event.get("role"),
                        "policy_group": trace_event.get("policy_group"),
                        "worker_group": trace_event.get("policy_group"),
                        "agent_id": trace_event.get("agent_id"),
                        "traj_uid": _to_python(field["traj_uid"]),
                        "reward_score": _to_python(output.reward_score),
                        "prompt_len": 0,
                        "response_len": 0,
                        "global_steps": _to_python(kwargs["global_steps"]),
                        "metadata": {
                            "trace_only": True,
                            **trace_event,
                        },
                    }
                )

        await tq.async_kv_batch_put(
            keys=keys,
            fields=list_of_dict_to_tensordict(fields),
            tags=tags,
            partition_id="train" if not validate else "val",
        )
        self._write_online_turns(runtime, online_turn_rows)

    def _write_online_turns(self, runtime: TrajWeaveAgentLoopRuntimeConfig, rows: list[dict[str, Any]]) -> None:
        if not rows or not runtime.capture_online_turns or not runtime.run_dir or not runtime.run_id:
            return
        path = Path(str(runtime.run_dir)) / "trajectories" / "online_turns" / f"worker-{os.getpid()}.jsonl"
        writer = JsonlWriter(path)
        for row in rows:
            writer.write({"run_id": runtime.run_id, **row})
