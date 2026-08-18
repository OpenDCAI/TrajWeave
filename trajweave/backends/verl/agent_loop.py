from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

import ray
import torch
import transfer_queue as tq

from trajweave.backends.verl.batch_padding import pad_session_batch
from trajweave.backends.verl.emitters import (
    AgentFlowEmitterMixin,
    ATGRPOEmitterMixin,
    C3EmitterMixin,
    CoMASEmitterMixin,
    CoMLRLEmitterMixin,
    DrMASEmitterMixin,
    GiGPOEmitterMixin,
    MAPoRLEmitterMixin,
    MARFTEmitterMixin,
    MARSHALSelfPlayEmitterMixin,
    MATPOEmitterMixin,
    MrlXEmitterMixin,
    WideSeekR1EmitterMixin,
)
from trajweave.backends.verl.emitters.registry import build_recipe_outputs
from trajweave.backends.verl.local_generation import HFLocalGenerationMixin
from trajweave.backends.verl.runtime_config import (
    TrajWeaveAgentLoopRuntimeConfig,
)
from trajweave.backends.verl.runtime_config import (
    config_get as _get,
)
from trajweave.backends.verl.runtime_config import (
    validate_agent_loop_backend as _validate_agent_loop_backend,
)
from trajweave.backends.verl.schema import (
    MAS_EXTRA_FIELDS,
)
from trajweave.backends.verl.schema import (
    batch_item as _batch_item,
)
from trajweave.backends.verl.schema import (
    canonical_drmas_agent_id as _canonical_drmas_agent_id,
)
from trajweave.backends.verl.schema import (
    pad_or_trim_1d as _pad_or_trim_1d,
)
from trajweave.backends.verl.schema import (
    padded_rm_scores as _padded_rm_scores,
)
from trajweave.backends.verl.schema import (
    resolve_comlrl_extra_fields as _resolve_comlrl_extra_fields,
)
from trajweave.backends.verl.schema import (
    to_python as _to_python,
)
from trajweave.storage.jsonl import JsonlWriter
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopOutput,
    AgentLoopWorker,
    get_trajectory_info,
)
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ
from verl.utils.tensordict_utils import list_of_dict_to_tensordict

logger = logging.getLogger(__name__)


TRAJWEAVE_AGENT_LOOP_MANAGER_FQN = "trajweave.backends.verl.agent_loop.TrajWeaveAgentLoopManager"


def _schedule_background_task(
    background_tasks: set[asyncio.Task[Any]],
    coroutine: Any,
) -> asyncio.Task[Any]:
    """保留异步 rollout task 引用，同时让 Ray 调用立即返回。"""

    task = asyncio.create_task(coroutine)
    background_tasks.add(task)
    task.add_done_callback(lambda completed: _consume_background_task(background_tasks, completed))
    return task


def _consume_background_task(
    background_tasks: set[asyncio.Task[Any]],
    task: asyncio.Task[Any],
) -> None:
    background_tasks.discard(task)
    if task.cancelled():
        logger.warning("TrajWeave rollout background task was cancelled.")
        return
    error = task.exception()
    if error is not None:
        logger.error(
            "TrajWeave rollout background task terminated unexpectedly: %s",
            error,
            exc_info=(type(error), error, error.__traceback__),
        )


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

    def reload_local_models(self, model_paths: dict[str, str], *, policy_version: int) -> list[dict[str, Any]]:
        """让所有 AgentLoop worker 在下一批采样前使用同一版本权重。"""

        if self.trajweave_runtime_config.agent_loop_backend != "hf_local_tq":
            return []
        results = ray.get(
            [
                worker.reload_local_models.remote(model_paths, policy_version=policy_version)
                for worker in self.agent_loop_workers
            ]
        )
        versions = {int(result["policy_version"]) for result in results}
        if versions != {int(policy_version)}:
            raise RuntimeError(
                f"AgentLoop weight reload returned inconsistent versions: expected {policy_version}, got {versions}."
            )
        logger.info(
            "Synchronized %d AgentLoop workers to policy_version=%d",
            len(results),
            policy_version,
        )
        return results

    def release_local_models(self) -> list[dict[str, Any]]:
        """在 Actor 导出 HF 快照前释放所有 AgentLoop 本地模型。"""

        if self.trajweave_runtime_config.agent_loop_backend != "hf_local_tq":
            return []
        results = ray.get([worker.release_local_models.remote() for worker in self.agent_loop_workers])
        logger.info("Released local rollout models from %d AgentLoop workers", len(results))
        return results

    def set_comlrl_iterative_context(self, context: dict[str, Any] | None) -> list[dict[str, Any]]:
        normalized = dict(context or {})
        return ray.get([worker.set_comlrl_iterative_context.remote(normalized) for worker in self.agent_loop_workers])


@ray.remote
class TrajWeaveSyntheticAgentLoopWorkerTQ(
    AgentFlowEmitterMixin,
    ATGRPOEmitterMixin,
    C3EmitterMixin,
    CoMASEmitterMixin,
    CoMLRLEmitterMixin,
    MARSHALSelfPlayEmitterMixin,
    MARFTEmitterMixin,
    MATPOEmitterMixin,
    MrlXEmitterMixin,
    WideSeekR1EmitterMixin,
    MAPoRLEmitterMixin,
    GiGPOEmitterMixin,
    DrMASEmitterMixin,
    HFLocalGenerationMixin,
    AgentLoopWorker,
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tq.init()
        self._trajweave_policy_version = 0
        self.background_tasks: set[asyncio.Task[Any]] = set()

    def reload_local_models(self, model_paths: dict[str, str], *, policy_version: int) -> dict[str, Any]:
        return HFLocalGenerationMixin.reload_local_models(
            self,
            model_paths,
            policy_version=policy_version,
        )

    def release_local_models(self) -> dict[str, Any]:
        return HFLocalGenerationMixin.release_local_models(self)

    def set_comlrl_iterative_context(self, context: dict[str, Any]) -> dict[str, Any]:
        self._trajweave_comlrl_iterative_context = dict(context)
        return {
            "phase": self._trajweave_comlrl_iterative_context.get("phase"),
            "iteration": self._trajweave_comlrl_iterative_context.get("iteration"),
        }

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

        for i in range(batch_size):
            prompt = {key: _batch_item(value, i) for key, value in batch.items()}
            _schedule_background_task(
                self.background_tasks,
                self._run_prompt(prompt, trajectory=trajectory_info[i]),
            )

    async def _run_prompt(self, prompt: dict[str, Any], trajectory: dict[str, Any]) -> None:
        uid = "<unknown>"
        partition_id = "train"
        try:
            uid = str(_to_python(prompt["uid"]))
            partition_id = "train" if not trajectory["validate"] else "val"
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "running"})
            config = self.config.actor_rollout_ref.rollout
            n = int(
                _to_python(prompt.pop("__rollout_n__", config.n if not trajectory["validate"] else config.val_kwargs.n))
            )
            runtime = TrajWeaveAgentLoopRuntimeConfig.from_verl_config(self.config)
            iterative_context_getter = getattr(self, "_comlrl_iterative_context", None)
            iterative_context = iterative_context_getter() if callable(iterative_context_getter) else {}
            if iterative_context.get("phase") == "preference":
                n = int(iterative_context.get("num_target_candidates", n))
            use_hf_local = runtime.agent_loop_backend == "hf_local_tq"
            if runtime.recipe == "c3_reasoner_actor_math":
                if n != 1:
                    raise ValueError("C3 uses trajweave.c3.fanout for nested alternatives and requires rollout.n=1.")
                outputs = build_recipe_outputs(
                    self,
                    recipe=runtime.recipe,
                    use_hf_local=use_hf_local,
                    prompt=dict(prompt),
                    session_id=0,
                    validate=trajectory["validate"],
                )
                await self._put_outputs(outputs, validate=trajectory["validate"], session_id=0, **prompt)
            elif runtime.recipe == "comlrl_joint_math":
                tree_prompt = dict(prompt)
                tree_prompt["__comlrl_num_candidates__"] = n
                self._comlrl_num_candidates(tree_prompt)
                self._comlrl_joint_mode()
                if use_hf_local:
                    outputs = self._build_hf_comlrl_joint_math_outputs(
                        tree_prompt,
                        session_id=0,
                        validate=trajectory["validate"],
                    )
                else:
                    outputs = self._build_comlrl_joint_math_outputs(
                        tree_prompt,
                        session_id=0,
                        validate=trajectory["validate"],
                    )
                await self._put_outputs(outputs, validate=trajectory["validate"], session_id=0, **prompt)
            elif runtime.recipe == "atgrpo_solver_verifier_math" and not trajectory["validate"]:
                tree_prompt = dict(prompt)
                tree_prompt["__atgrpo_branch_factor__"] = n
                outputs = build_recipe_outputs(
                    self,
                    recipe=runtime.recipe,
                    use_hf_local=use_hf_local,
                    prompt=tree_prompt,
                    session_id=0,
                    validate=False,
                )
                await self._put_outputs(outputs, validate=False, session_id=0, **prompt)
            else:
                for session_id in range(n):
                    session_prompt = dict(prompt)
                    if runtime.recipe == "atgrpo_solver_verifier_math":
                        session_prompt["__atgrpo_branch_factor__"] = 1
                    outputs = build_recipe_outputs(
                        self,
                        recipe=runtime.recipe,
                        use_hf_local=use_hf_local,
                        prompt=session_prompt,
                        session_id=session_id,
                        validate=trajectory["validate"],
                    )
                    await self._put_outputs(outputs, validate=trajectory["validate"], session_id=session_id, **prompt)
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "finished"})
        except Exception as exc:
            logger.exception("TrajWeave synthetic TQ worker failed for uid=%s", uid)
            if uid == "<unknown>":
                raise
            try:
                await tq.async_kv_put(
                    key=uid,
                    partition_id=partition_id,
                    tag={
                        "status": "failure",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc)[:500],
                    },
                )
            except Exception as status_exc:
                raise RuntimeError(f"Failed to mark rollout prompt {uid!r} as failure.") from status_exc

    async def _put_outputs(self, outputs: list[AgentLoopOutput], validate: bool, **kwargs) -> None:
        runtime = TrajWeaveAgentLoopRuntimeConfig.from_verl_config(self.config)
        final_output = outputs[-1]
        if final_output.reward_score is not None:
            for output in outputs[:-1]:
                if output.reward_score is None:
                    output.reward_score = final_output.reward_score
                output.extra_fields.setdefault(
                    "reward_extra_info", final_output.extra_fields.get("reward_extra_info", {})
                )

        uid, session_id = str(_to_python(kwargs["uid"])), int(kwargs["session_id"])
        keys, fields, tags = [], [], []
        online_turn_rows: list[dict[str, Any]] = []
        for index, output in enumerate(outputs):
            output.extra_fields.setdefault("min_global_steps", self._local_policy_version())
            output.extra_fields.setdefault("max_global_steps", self._local_policy_version())
            output.extra_fields.setdefault("policy_version", self._local_policy_version())
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
                [0] * prompt_pad + [1] * len(prompt_ids) + [1] * len(response_ids) + [0] * response_pad,
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
            field["policy_group"] = output.extra_fields.get(
                "policy_group", field.get("policy_group", field["agent_name"])
            )
            field["worker_group"] = output.extra_fields.get("worker_group", field["policy_group"])
            field["worker_group_model_path"] = output.extra_fields.get("worker_group_model_path") or ""
            field["agent_id"] = output.extra_fields.get("agent_id", _canonical_drmas_agent_id(field["agent_name"]))
            field["traj_uid"] = output.extra_fields.get("traj_uid", f"{uid}_{session_id}")
            # `index` is the position within the *trainable* turns of this session, which
            # most recipes intentionally rely on (e.g. GiGPO's step-transition tracking
            # expects a dense 0..N-1 sequence over its trainable solver turns even though
            # the verifier is non-trainable). Recipes that need the orchestra's absolute
            # turn index instead (e.g. AT-GRPO's turn-wise credit grouping, which mixes
            # trainable and -- in principle -- non-trainable agents) can opt in by setting
            # `turn_id` in extra_fields; see workflow_runtime.py's `_trajectory_to_outputs`.
            field["turn_id"] = output.extra_fields.get("turn_id", index)
            for mas_field in MAS_EXTRA_FIELDS:
                if mas_field in output.extra_fields:
                    field[mas_field] = output.extra_fields[mas_field]
            field.update(
                _resolve_comlrl_extra_fields(
                    output.extra_fields,
                    row_id=f"{uid}_{session_id}_{index}",
                )
            )
            field["session_id"] = session_id
            field["loss_mask"] = field["response_mask"]
            field["input_ids"] = input_ids
            field["attention_mask"] = attention_mask
            field["position_ids"] = position_ids
            field["temperature"] = float(_get(self.rollout_config, "temperature", 1.0))
            field["multi_modal_inputs"] = multi_modal_inputs
            fields.append(field)
            keys.append(f"{uid}_{session_id}_{index}")
            stored_prompt_len, stored_response_len = field["prompts"].size(0), field["responses"].size(0)
            tags.append(
                {
                    "status": "success",
                    "prompt_len": stored_prompt_len,
                    "response_len": stored_response_len,
                    "seq_len": stored_prompt_len + stored_response_len,
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
                    "turn_id": _to_python(field["turn_id"]),
                    "validate": validate,
                    "agent_name": _to_python(field["agent_name"]),
                    "role": _to_python(field["role"]),
                    "policy_group": _to_python(field["policy_group"]),
                    "worker_group": _to_python(field["worker_group"]),
                    "worker_group_model_path": _to_python(field["worker_group_model_path"]),
                    "agent_id": _to_python(field["agent_id"]),
                    "traj_uid": _to_python(field["traj_uid"]),
                    "reward_score": _to_python(output.reward_score),
                    "prompt_text": _to_python(output.extra_fields.get("prompt_text", "")),
                    "response_text": _to_python(output.extra_fields.get("response_text", "")),
                    "observation_text": _to_python(output.extra_fields.get("observation_text", "")),
                    "final_answer": _to_python(output.extra_fields.get("final_answer", "")),
                    "workflow_success": bool(output.extra_fields.get("workflow_success", False)),
                    "anchor_observation": _to_python(output.extra_fields.get("anchor_obs")),
                    "next_observation": _to_python(output.extra_fields.get("next_obs")),
                    "step_reward": _to_python(output.extra_fields.get("step_reward")),
                    "prompt_len": len(prompt_ids),
                    "response_len": len(response_ids),
                    "global_steps": _to_python(kwargs["global_steps"]),
                    "metadata": {key: _to_python(field[key]) for key in MAS_EXTRA_FIELDS if key in field},
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
                        "worker_group_model_path": trace_event.get("worker_group_model_path", ""),
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

        pad_session_batch(
            keys=keys,
            fields=fields,
            tags=tags,
            multiple=runtime.turn_padding_multiple,
            uid=uid,
            session_id=session_id,
        )
        await tq.async_kv_batch_put(
            keys=keys,
            fields=list_of_dict_to_tensordict(fields),
            tags=tags,
            partition_id="train" if not validate else "val",
        )
        self._attach_worker_group_stats(online_turn_rows)
        self._write_online_turns(runtime, online_turn_rows)

    def _attach_worker_group_stats(self, rows: list[dict[str, Any]]) -> None:
        stats: dict[str, dict[str, float]] = {}
        for row in rows:
            metadata = row.get("metadata", {})
            if metadata.get("trace_only"):
                continue
            group_id = str(row.get("worker_group", ""))
            if not group_id:
                continue
            group_stats = stats.setdefault(group_id, {"count": 0.0, "reward_sum": 0.0, "response_len_sum": 0.0})
            group_stats["count"] += 1
            reward_score = row.get("reward_score")
            if reward_score is not None:
                group_stats["reward_sum"] += float(reward_score)
            group_stats["response_len_sum"] += float(row.get("response_len") or 0)

        for group_stats in stats.values():
            count = max(group_stats["count"], 1.0)
            group_stats["reward_mean"] = group_stats["reward_sum"] / count
            group_stats["response_len_mean"] = group_stats["response_len_sum"] / count

        for row in rows:
            group_id = str(row.get("worker_group", ""))
            if group_id in stats:
                row.setdefault("metadata", {})["worker_group_batch_stats"] = dict(stats[group_id])

    def _write_online_turns(self, runtime: TrajWeaveAgentLoopRuntimeConfig, rows: list[dict[str, Any]]) -> None:
        if not rows or not runtime.capture_online_turns or not runtime.run_dir or not runtime.run_id:
            return
        path = Path(str(runtime.run_dir)) / "trajectories" / "online_turns" / f"worker-{os.getpid()}.jsonl"
        writer = JsonlWriter(path)
        for row in rows:
            writer.write({"run_id": runtime.run_id, **row})
