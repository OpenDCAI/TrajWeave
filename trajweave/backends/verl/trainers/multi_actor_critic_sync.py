from __future__ import annotations

import copy
import json
import logging
import os
import uuid
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
import transfer_queue as tq
from omegaconf import OmegaConf, open_dict
from tensordict import TensorDict
from transfer_queue import KVBatchMeta

from trajweave.backends.verl.extensions.comlrl.actor_critic import (
    ACTOR_CRITIC_TQ_FIELDS,
    prepare_actor_critic_batch,
    verl_unclipped_mse_critic_loss,
)
from trajweave.backends.verl.multi_actor.critic_config import (
    CriticRouteSpec,
    normalize_critic_route_specs,
    resolve_actor_critic_settings,
)
from trajweave.backends.verl.routing import safe_critic_role_key, split_tq_batch_by_field
from trajweave.backends.verl.schema import flatten_token_ids, to_python
from trajweave.backends.verl.trainers.multi_actor_sync import (
    TrajWeaveMultiActorSyncTrainer,
    _actor_rollout_ref_config_for_group,
    _batch_len,
    _safe_path_name,
)
from trajweave.credit.comlrl.actor_critic import (
    build_iac_critic_text,
    build_maac_critic_text,
    deduplicate_maac_critic_indices,
    deduplicate_maac_critic_samples,
)
from verl.single_controller.ray import (
    RayClassWithInitArgs,
    RayWorkerGroup,
    ResourcePoolManager,
    create_colocated_worker_cls,
)
from verl.trainer.ppo.utils import Role
from verl.trainer.ppo.v1.trainer_base import TrainingWorkerConfig, register_trainer
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.tensordict_utils import list_of_dict_to_tensordict
from verl.workers.engine_workers import TrainingWorker

logger = logging.getLogger(__name__)

ACTOR_CRITIC_CHECKPOINT_MANIFEST = "actor_critic_manifest.json"


@dataclass(frozen=True)
class CriticRoutedBatch:
    critic_group: str
    batch: Any


@register_trainer("trajweave_multi_actor_critic_sync")
class TrajWeaveMultiActorCriticSyncTrainer(TrajWeaveMultiActorSyncTrainer):
    """Separate-critic IAC/MAAC trainer over TrajWeave joint TQ rows."""

    def _init_multi_actor_specs(self) -> None:
        super()._init_multi_actor_specs()
        raw_routes, topology, critic_type, max_length = _critic_routes_config(self.config)
        self.critic_route_specs = {
            route.critic_group: route
            for route in normalize_critic_route_specs(
                raw_routes,
                trainable_actor_groups=self.multi_actor_trainable_group_ids,
                topology=topology,
                critic_type=critic_type,
                max_length=max_length,
            )
        }
        self.critic_topology = next(iter(self.critic_route_specs.values())).topology
        self.critic_type = next(iter(self.critic_route_specs.values())).critic_type
        self.actor_to_critic_group = {
            actor_group: route.critic_group
            for route in self.critic_route_specs.values()
            for actor_group in route.actor_groups
        }

    def _validate_multi_actor_specs(self) -> None:
        super()._validate_multi_actor_specs()
        if not self.use_critic:
            raise ValueError("TrajWeave actor-critic trainer requires critic.enable=true.")
        actor = self.config.actor_rollout_ref.actor
        loss_mode = str(actor.policy_loss.get("loss_mode", "vanilla"))
        if loss_mode != "gpg":
            raise ValueError("CoMLRL IAC/MAAC requires actor.policy_loss.loss_mode=gpg (ratio-free actor loss).")
        if str(actor.loss_agg_mode) != "seq-mean-token-sum":
            raise ValueError("CoMLRL IAC/MAAC requires actor.loss_agg_mode=seq-mean-token-sum.")
        _validate_single_gpu_critic_routes(self.critic_route_specs.values())
        role_keys = [safe_critic_role_key(group_id) for group_id in self.critic_route_specs]
        if len(role_keys) != len(set(role_keys)):
            raise ValueError(f"Critic group ids produce colliding safe role keys: {role_keys}.")
        requested = sum(route.gpus for route in self.critic_route_specs.values())
        requested += sum(
            self.multi_actor_worker_group_specs[group_id].gpus for group_id in self.multi_actor_trainable_group_ids
        )
        available = int(self.config.trainer.n_gpus_per_node)
        if requested > available:
            raise ValueError(
                "Actor and separate critic groups request more GPUs than trainer.n_gpus_per_node: "
                f"requested={requested}, available={available}."
            )

    def _init_resource_pool_mgr(self):
        self.role_worker_mapping = {}
        self.mapping = {}
        resource_pool_spec = {
            self.multi_actor_worker_group_specs[group_id].role_key: [self.multi_actor_worker_group_specs[group_id].gpus]
            for group_id in self.multi_actor_trainable_group_ids
        }
        resource_pool_spec.update(
            {safe_critic_role_key(route.critic_group): [route.gpus] for route in self.critic_route_specs.values()}
        )
        self.resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec,
            mapping=self.mapping,
            max_colocate_count=1,
        )

    def _create_worker_groups(self) -> None:
        import ray

        wg_kwargs = {"device_name": self.config.trainer.device}
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")

        self.actor_rollout_wgs = {}
        for group_id in self.multi_actor_trainable_group_ids:
            group = self.multi_actor_worker_group_specs[group_id]
            class_dict = {
                group.role_key: RayClassWithInitArgs(
                    cls=ray.remote(
                        __import__(
                            "trajweave.backends.verl.extensions.drmas.agent_wise_grpo",
                            fromlist=["TrajWeaveActorRolloutRefWorker"],
                        ).TrajWeaveActorRolloutRefWorker
                    ),
                    config=_actor_rollout_ref_config_for_group(self.config, group),
                    distillation_config=self.config.get("distillation"),
                    role=str(Role.Actor),
                )
            }
            worker_cls = create_colocated_worker_cls(class_dict=class_dict)
            pool = self.resource_pool_manager.resource_pool_dict[group.role_key]
            spawned = RayWorkerGroup(resource_pool=pool, ray_cls_with_init=worker_cls, **wg_kwargs).spawn(
                prefix_set=class_dict.keys()
            )
            worker_group = spawned[group.role_key]
            worker_group.init_model()
            self.actor_rollout_wgs[group_id] = worker_group
        self.actor_rollout_wg = self.actor_rollout_wgs[self.multi_actor_trainable_group_ids[0]]

        self.critic_wgs = {}
        self._critic_cfgs = {}
        for route in self.critic_route_specs.values():
            critic_cfg = _critic_worker_config_for_route(self.config, route)
            worker_cfg = TrainingWorkerConfig(
                model_type="value_model",
                model_config=getattr(critic_cfg, "model_config", None) or critic_cfg.model,
                engine_config=critic_cfg.engine,
                optimizer_config=critic_cfg.optim,
                checkpoint_config=critic_cfg.checkpoint,
            )
            role_key = safe_critic_role_key(route.critic_group)
            class_dict = {
                role_key: RayClassWithInitArgs(cls=ray.remote(TrainingWorker), config=worker_cfg),
            }
            worker_cls = create_colocated_worker_cls(class_dict=class_dict)
            pool = self.resource_pool_manager.resource_pool_dict[role_key]
            spawned = RayWorkerGroup(resource_pool=pool, ray_cls_with_init=worker_cls, **wg_kwargs).spawn(
                prefix_set=class_dict.keys()
            )
            worker_group = spawned[role_key]
            worker_group.reset()
            value_loss_coef = float(_actor_critic_settings(self.config).get("value_loss_coef", 0.6))
            worker_group.set_loss_fn(
                partial(
                    verl_unclipped_mse_critic_loss,
                    critic_cfg,
                    value_loss_coef=value_loss_coef,
                )
            )
            self.critic_wgs[route.critic_group] = worker_group
            self._critic_cfgs[route.critic_group] = critic_cfg
        self.critic_wg = self.critic_wgs[next(iter(self.critic_wgs))]
        self.ref_in_actor = True
        self.ref_policy_wg = None

    def _critic_tokenizer(self, route: CriticRouteSpec):
        if not hasattr(self, "_critic_tokenizers"):
            self._critic_tokenizers = {}
        tokenizer = self._critic_tokenizers.get(route.tokenizer_path)
        if tokenizer is None:
            tokenizer = _load_critic_tokenizer(str(route.tokenizer_path))
            self._critic_tokenizers[str(route.tokenizer_path)] = tokenizer
        return tokenizer

    def _ensure_critic_fields(self, batch: Any) -> Any:
        fields = (
            "worker_group",
            "agent_id",
            "traj_uid",
            "turn_id",
            "prompt_text",
            "response_text",
            "joint_action_ids",
            "joint_transition_ids",
            "critic_group",
            "critic_type",
            "critic_input_ids",
            "critic_attention_mask",
            "critic_position_ids",
            "critic_loss_mask",
            "response_mask",
        )
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)
        rows = [
            {field: _tq_row_value(data[field], row, row_count=len(batch.keys)) for field in fields}
            for row in range(len(batch.keys))
        ]
        active_rows = [bool(torch.as_tensor(row["response_mask"]).bool().any()) for row in rows]
        for row_index, (row, active) in enumerate(zip(rows, active_rows, strict=True)):
            if not active:
                continue
            _validate_linear_row(row, row_index=row_index, batch_key=batch.keys[row_index])
            worker_group = str(row["worker_group"])
            critic_group = self.actor_to_critic_group.get(worker_group)
            if critic_group is None:
                raise ValueError(f"Active critic row references unknown worker_group {worker_group!r}.")
            route = self.critic_route_specs[critic_group]
            supplied_group = str(row["critic_group"])
            if not supplied_group.startswith(("__missing__", "__padding__")) and supplied_group != route.critic_group:
                raise ValueError(
                    f"Critic row {batch.keys[row_index]!r} supplies critic_group={supplied_group!r}; "
                    f"derived route is {route.critic_group!r}."
                )
            supplied_type = str(row["critic_type"])
            if not supplied_type.startswith(("__missing__", "__padding__")) and supplied_type != route.critic_type:
                raise ValueError(
                    f"Critic row {batch.keys[row_index]!r} supplies critic_type={supplied_type!r}; "
                    f"route requires {route.critic_type!r}."
                )
            row["critic_group"] = route.critic_group
            row["critic_type"] = route.critic_type
            row["critic_loss_mask"] = 1.0

        texts: dict[int, str] = {}
        if self.critic_topology == "independent":
            seen: set[tuple[str, str, int]] = set()
            for row_index, (row, active) in enumerate(zip(rows, active_rows, strict=True)):
                if not active:
                    continue
                linear_key = (str(row["traj_uid"]), str(row["agent_id"]), int(row["turn_id"]))
                if linear_key in seen:
                    raise ValueError(
                        "IAC supports one linear transition per (traj_uid, agent, turn); "
                        f"duplicate={linear_key!r}. Cross/full-tree candidates are unsupported."
                    )
                seen.add(linear_key)
                texts[row_index] = build_iac_critic_text(
                    str(row["prompt_text"]),
                    action=str(row["response_text"]) if self.critic_type == "q" else None,
                    critic_type=self.critic_type,
                    agent_name=str(row["agent_id"]),
                )
        else:
            grouped: dict[tuple[str, str], list[int]] = {}
            transitions_by_turn: dict[tuple[str, int], set[str]] = {}
            for row_index, (row, active) in enumerate(zip(rows, active_rows, strict=True)):
                if not active:
                    continue
                composite = (str(row["traj_uid"]), str(row["joint_transition_ids"][0]))
                grouped.setdefault(composite, []).append(row_index)
                turn_key = (str(row["traj_uid"]), int(row["turn_id"]))
                transitions_by_turn.setdefault(turn_key, set()).add(composite[1])
            ambiguous = {key: values for key, values in transitions_by_turn.items() if len(values) != 1}
            if ambiguous:
                raise ValueError(
                    "MAAC supports one unique joint transition per (traj_uid, turn); "
                    f"got {ambiguous!r}. Cross/full-tree candidates are unsupported."
                )
            expected_groups = tuple(self.multi_actor_trainable_group_ids)
            for composite, indexes in grouped.items():
                by_group: dict[str, int] = {}
                for row_index in indexes:
                    group = str(rows[row_index]["worker_group"])
                    if group in by_group:
                        raise ValueError(
                            f"MAAC transition {composite!r} has multiple active rows for actor group {group!r}."
                        )
                    by_group[group] = row_index
                if set(by_group) != set(expected_groups):
                    raise ValueError(
                        f"MAAC transition {composite!r} must contain exactly one row for every actor group; "
                        f"expected={list(expected_groups)}, got={sorted(by_group)}."
                    )
                prompts = {group: str(rows[by_group[group]]["prompt_text"]) for group in expected_groups}
                actions = {group: str(rows[by_group[group]]["response_text"]) for group in expected_groups}
                text = build_maac_critic_text(
                    expected_groups,
                    prompts,
                    actions_by_agent=actions if self.critic_type == "q" else None,
                    critic_type=self.critic_type,
                )
                for row_index in indexes:
                    texts[row_index] = text

        output_rows = []
        for row_index, (row, active) in enumerate(zip(rows, active_rows, strict=True)):
            if not active:
                output_rows.append(
                    {
                        "critic_group": row["critic_group"],
                        "critic_type": row["critic_type"],
                        "critic_input_ids": [],
                        "critic_attention_mask": [],
                        "critic_position_ids": [],
                        "critic_loss_mask": 0.0,
                    }
                )
                continue
            route = self.critic_route_specs[str(row["critic_group"])]
            token_fields = _validated_or_tokenized_critic_fields(
                row,
                text=texts[row_index],
                tokenizer=self._critic_tokenizer(route),
                route=route,
                batch_key=batch.keys[row_index],
            )
            output_rows.append(
                {
                    "critic_group": route.critic_group,
                    "critic_type": route.critic_type,
                    "critic_loss_mask": 1.0,
                    **token_fields,
                }
            )
        tq.kv_batch_put(
            keys=batch.keys,
            partition_id=batch.partition_id,
            fields=list_of_dict_to_tensordict(output_rows),
        )
        return batch

    def _route_critic_batch(self, batch: Any) -> list[CriticRoutedBatch]:
        self._ensure_critic_fields(batch)
        real_batch = _select_real_critic_rows(batch)
        if real_batch is None:
            return []
        routed = split_tq_batch_by_field(real_batch, field="critic_group")
        output: list[CriticRoutedBatch] = []
        for item in routed:
            if item.group_id not in self.critic_wgs:
                known = ", ".join(sorted(self.critic_wgs))
                raise KeyError(f"Unknown critic_group {item.group_id!r}. Known groups: {known}.")
            route = self.critic_route_specs[item.group_id]
            routed_batch = item.batch
            critic_types = set(_tq_string_values(routed_batch, "critic_type"))
            if critic_types != {route.critic_type}:
                raise ValueError(
                    f"Critic group {route.critic_group!r} expects critic_type={route.critic_type!r}, "
                    f"got {sorted(critic_types)}."
                )
            if route.topology == "independent":
                actor_groups = _tq_string_values(routed_batch, "worker_group")
                unexpected = sorted(set(actor_groups) - set(route.actor_groups))
                if unexpected:
                    raise ValueError(
                        f"IAC critic group {route.critic_group!r} received rows for actor groups {unexpected}."
                    )
            else:
                routed_batch = _deduplicate_joint_transition_batch(routed_batch)
            world_size = int(self.critic_wgs[item.group_id].world_size)
            if _batch_len(routed_batch) % world_size != 0:
                raise ValueError(
                    "Routed critic batch must be divisible by critic worker-group world size; "
                    f"group={item.group_id!r}, samples={_batch_len(routed_batch)}, world_size={world_size}."
                )
            output.append(CriticRoutedBatch(item.group_id, routed_batch))
        if self.critic_topology == "centralized" and len(output) > 1:
            raise ValueError("MAAC may route to exactly one centralized critic group per batch.")
        return output

    def _compute_values(self, batch, metrics: dict):
        del metrics
        scalar_values: dict[str, float] = {}
        transition_values: dict[tuple[str, str], float] = {}
        for routed in self._route_critic_batch(batch):
            critic_batch = _materialize_critic_tq_batch(routed.batch, include_targets=False)
            critic_batch.extra_info["compute_loss"] = False
            try:
                result = self.critic_wgs[routed.critic_group].infer_batch(critic_batch)
                if hasattr(result, "futures"):
                    __import__("ray").get(result.futures)
                values = _read_critic_scalar_values(critic_batch)
                for source_key, value in zip(routed.batch.keys, values, strict=True):
                    scalar_values[source_key] = value
                if self.critic_topology == "centralized":
                    transition_keys = _transition_keys(routed.batch)
                    transition_values.update(dict(zip(transition_keys, values, strict=True)))
            finally:
                tq.kv_clear(keys=critic_batch.keys, partition_id=critic_batch.partition_id)

        real_batch = _select_real_critic_rows(batch)
        if real_batch is None:
            raise ValueError("Actor-critic batch has no non-padding critic rows.")
        if self.critic_topology == "centralized":
            real_values = [transition_values[transition_key] for transition_key in _transition_keys(real_batch)]
        else:
            real_values = [scalar_values[key] for key in real_batch.keys]
        values_by_key = dict(zip(real_batch.keys, real_values, strict=True))
        all_values = [values_by_key.get(key, 0.0) for key in batch.keys]
        response_data = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=["response_mask"],
        )
        response_mask = response_data["response_mask"]
        scalar_tensor = torch.tensor(all_values, dtype=torch.float32)
        token_values = _scalar_values_to_response(scalar_tensor, response_mask)
        tq.kv_batch_put(
            keys=batch.keys,
            partition_id=batch.partition_id,
            fields=TensorDict(
                {"old_values": scalar_tensor, "values": token_values},
                batch_size=len(all_values),
            ),
        )
        return batch

    def _compute_advantage(self, batch, metrics: dict):
        del metrics
        fields = (*ACTOR_CRITIC_TQ_FIELDS, "response_mask", "old_values")
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)
        mapping = {key: _unpack_advantage_tq_field(data[key]) for key in fields if key in data}
        settings = _actor_critic_settings(self.config)
        prepared = prepare_actor_critic_batch(
            mapping,
            topology=self.critic_topology,
            gamma=float(settings.get("gamma", self.config.algorithm.gamma)),
            normalize_by_population_std=bool(settings.get("normalize_advantages", True)),
        )
        from verl.workers.utils.padding import response_to_nested

        output = TensorDict(
            {
                "advantages": response_to_nested(prepared.actor_advantages, data["response_mask"]),
                "returns": response_to_nested(prepared.actor_returns, data["response_mask"]),
                "critic_returns": prepared.scalar_targets,
            },
            batch_size=len(batch.keys),
        )
        tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=output)
        return batch

    def _update_critic(self, batch, metrics: dict):
        updated = 0
        for routed in self._route_critic_batch(batch):
            critic_batch = _materialize_critic_tq_batch(routed.batch, include_targets=True)
            critic_cfg = self._critic_cfgs[routed.critic_group]
            critic_batch.extra_info.update(
                {
                    "global_batch_size": len(critic_batch.keys),
                    "mini_batch_size": None,
                    "num_mini_batch": 1,
                    "epochs": critic_cfg.ppo_epochs,
                    "seed": critic_cfg.data_loader_seed,
                    "dataloader_kwargs": {"shuffle": critic_cfg.shuffle},
                }
            )
            try:
                output = self.critic_wgs[routed.critic_group].train_mini_batch(critic_batch).get()
                renamed = rename_dict(output["metrics"], f"critic/{routed.critic_group}/")
                metrics.update(reduce_metrics(renamed))
                updated += 1
            finally:
                tq.kv_clear(keys=critic_batch.keys, partition_id=critic_batch.partition_id)
        if updated == 0:
            raise RuntimeError("Actor-critic update did not update any critic worker group.")
        metrics["trajweave/comlrl/critic_groups/updated"] = updated
        return batch

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        root = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        local_mkdir_safe(root)
        expected_critics = set(self.critic_route_specs)
        actual_critics = set(self.critic_wgs)
        if actual_critics != expected_critics:
            raise RuntimeError(
                "Actor-critic checkpoint worker groups do not match configured critic routes; "
                f"missing={sorted(expected_critics - actual_critics)}, "
                f"extra={sorted(actual_critics - expected_critics)}."
            )
        paths = checkpoint_directory_map(
            root,
            actor_group_ids=self.multi_actor_trainable_group_ids,
            critic_group_ids=self.critic_wgs,
        )
        max_actor = self.config.trainer.get("max_actor_ckpt_to_keep", None)
        max_critic = self.config.trainer.get("max_critic_ckpt_to_keep", None)
        for group_id, worker_group in self.actor_rollout_wgs.items():
            if group_id in self.multi_actor_trainable_group_ids:
                worker_group.save_checkpoint(
                    paths["actors"][group_id], None, self.global_steps, max_ckpt_to_keep=max_actor
                )
        for group_id, worker_group in self.critic_wgs.items():
            worker_group.save_checkpoint(
                paths["critics"][group_id], None, self.global_steps, max_ckpt_to_keep=max_critic
            )
        torch.save(self.train_dataloader.state_dict(), os.path.join(root, "data.pt"))
        manifest = _actor_critic_checkpoint_manifest(self, root=root, paths=paths)
        _atomic_write_json(os.path.join(root, ACTOR_CRITIC_CHECKPOINT_MANIFEST), manifest)
        latest_path = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(latest_path, "w") as file:
            file.write(str(self.global_steps))

    def _load_checkpoint(self):
        self.global_steps = 0
        mode = str(self.config.trainer.resume_mode)
        if mode == "disable":
            return
        resume_from_path = self.config.trainer.get("resume_from_path", None)
        marker = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        if mode == "auto" and not resume_from_path and not os.path.exists(marker):
            logger.info("No multi-actor critic checkpoint marker found; training from scratch.")
            return
        marker_exists = os.path.exists(marker)
        raise ValueError(
            "TrajWeave multi-actor actor/critic checkpoint resume is not implemented. "
            f"Refusing resume_mode={mode!r}, resume_from_path={resume_from_path!r}, "
            f"marker_exists={marker_exists}."
        )


def _validate_single_gpu_critic_routes(routes: Any) -> None:
    unsupported = [route.critic_group for route in routes if route.gpus != 1]
    if unsupported:
        raise ValueError(
            "TrajWeave IAC/MAAC currently requires gpus=1 for every critic route; "
            f"multi-GPU critic padding is not implemented for {unsupported}."
        )


def _unpack_advantage_tq_field(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.is_nested:
            try:
                return torch.nested.to_padded_tensor(value, 0)
            except NotImplementedError:
                offsets = value.offsets()
                flat = value.values()
                lengths = offsets.diff()
                max_length = int(lengths.max().item()) if lengths.numel() else 0
                padded = flat.new_zeros((lengths.numel(), max_length, *flat.shape[1:]))
                for row, length in enumerate(lengths.tolist()):
                    start = int(offsets[row].item())
                    padded[row, :length] = flat[start : start + length]
                return padded
        return value
    return to_python(value)


def _load_critic_tokenizer(tokenizer_path: str):
    from verl.utils import hf_tokenizer

    return hf_tokenizer(tokenizer_path)


def _tq_row_value(value: Any, row: int, *, row_count: int) -> Any:
    if isinstance(value, torch.Tensor):
        return to_python(value[row])
    unpacked = to_python(value)
    if isinstance(unpacked, list | tuple) and len(unpacked) == row_count:
        return unpacked[row]
    return unpacked


def _validate_linear_row(row: dict[str, Any], *, row_index: int, batch_key: str) -> None:
    action_ids = row["joint_action_ids"]
    transition_ids = row["joint_transition_ids"]
    if isinstance(action_ids, str):
        action_ids = [action_ids]
    if isinstance(transition_ids, str):
        transition_ids = [transition_ids]
    if not isinstance(action_ids, list | tuple) or len(action_ids) != 1:
        raise ValueError(
            f"Active actor-critic row {batch_key!r} ({row_index}) must reference exactly one joint action; "
            "cross/full-tree candidates are unsupported."
        )
    if not isinstance(transition_ids, list | tuple) or len(transition_ids) != 1:
        raise ValueError(
            f"Active actor-critic row {batch_key!r} ({row_index}) must reference exactly one joint transition; "
            "cross/full-tree candidates are unsupported."
        )
    action_id = str(action_ids[0])
    transition_id = str(transition_ids[0])
    if not action_id or not transition_id or transition_id.startswith(("__missing__", "__padding__")):
        raise ValueError(f"Active actor-critic row {batch_key!r} must reference real joint action/transition IDs.")
    row["joint_action_ids"] = [action_id]
    row["joint_transition_ids"] = [transition_id]


def _validated_or_tokenized_critic_fields(
    row: dict[str, Any],
    *,
    text: str,
    tokenizer: Any,
    route: CriticRouteSpec,
    batch_key: str,
) -> dict[str, list[int]]:
    input_ids = _integer_list(row["critic_input_ids"], field="critic_input_ids", batch_key=batch_key)
    attention_mask = _integer_list(row["critic_attention_mask"], field="critic_attention_mask", batch_key=batch_key)
    position_ids = _integer_list(row["critic_position_ids"], field="critic_position_ids", batch_key=batch_key)
    model_limit = getattr(tokenizer, "model_max_length", None)
    if _is_real_tokenizer_limit(model_limit) and route.max_length > int(model_limit):
        raise ValueError(
            f"Critic route {route.critic_group!r} max_length={route.max_length} exceeds "
            f"tokenizer model_max_length={int(model_limit)}."
        )
    provided = bool(input_ids or attention_mask or position_ids)
    if provided:
        if not input_ids or len(input_ids) != len(attention_mask) or len(input_ids) != len(position_ids):
            raise ValueError(f"Critic row {batch_key!r} has partially provided or misaligned critic token fields.")
        if any(mask not in {0, 1} for mask in attention_mask) or not any(attention_mask):
            raise ValueError(f"Critic row {batch_key!r} requires a non-empty binary critic_attention_mask.")
        if sum(attention_mask) > route.max_length:
            raise ValueError(
                f"Critic row {batch_key!r} has {sum(attention_mask)} attended tokens, "
                f"exceeding route max_length={route.max_length}."
            )
        return {
            "critic_input_ids": input_ids,
            "critic_attention_mask": attention_mask,
            "critic_position_ids": position_ids,
        }

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=route.max_length,
        add_special_tokens=False,
    )
    tokens = flatten_token_ids(encoded)
    if not tokens:
        raise ValueError(f"Critic tokenizer produced no tokens for active row {batch_key!r}.")
    return {
        "critic_input_ids": tokens,
        "critic_attention_mask": [1] * len(tokens),
        "critic_position_ids": list(range(len(tokens))),
    }


def _is_real_tokenizer_limit(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    try:
        limit = int(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return 0 < limit < 1_000_000_000


def _integer_list(value: Any, *, field: str, batch_key: str) -> list[int]:
    value = to_python(value)
    if value is None:
        return []
    if not isinstance(value, list | tuple):
        raise ValueError(f"Critic row {batch_key!r} field {field!r} must be a token sequence.")
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Critic row {batch_key!r} field {field!r} must contain integers.") from exc


def checkpoint_directory_map(
    root: str,
    *,
    actor_group_ids: Any,
    critic_group_ids: Any,
) -> dict[str, dict[str, str]]:
    actors = {str(group): os.path.join(root, "actors", _safe_path_name(str(group))) for group in actor_group_ids}
    critics = {str(group): os.path.join(root, "critics", _safe_path_name(str(group))) for group in critic_group_ids}
    if len(set(actors.values())) != len(actors):
        raise ValueError("Actor group ids collide after checkpoint path sanitization.")
    if len(set(critics.values())) != len(critics):
        raise ValueError("Critic group ids collide after checkpoint path sanitization.")
    return {"actors": actors, "critics": critics}


def _actor_critic_checkpoint_manifest(
    trainer: TrajWeaveMultiActorCriticSyncTrainer,
    *,
    root: str,
    paths: dict[str, dict[str, str]],
) -> dict[str, Any]:
    actor_groups = []
    for group_id in trainer.multi_actor_trainable_group_ids:
        actor_groups.append(
            {
                "group_id": str(group_id),
                "path": os.path.relpath(paths["actors"][group_id], root),
            }
        )
    critic_groups = []
    for group_id, route in trainer.critic_route_specs.items():
        critic_groups.append(
            {
                "group_id": str(group_id),
                "path": os.path.relpath(paths["critics"][group_id], root),
                "actor_groups": list(route.actor_groups),
                "topology": route.topology,
                "critic_type": route.critic_type,
            }
        )
    return {
        "schema_version": 1,
        "checkpoint_kind": "trajweave_multi_actor_critic",
        "global_step": int(trainer.global_steps),
        "resume_supported": False,
        "critic_topology": str(trainer.critic_topology),
        "critic_type": str(trainer.critic_type),
        "actors": actor_groups,
        "critics": critic_groups,
        "dataloader_state": "data.pt",
    }


def _atomic_write_json(path: str, payload: dict[str, Any]) -> None:
    temporary_path = f"{path}.tmp-{uuid.uuid4().hex}"
    try:
        with open(temporary_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _critic_routes_config(config: Any) -> tuple[Any, str | None, str | None, int]:
    settings = _actor_critic_settings(config)
    raw = settings.get("critic_routes", settings.get("critic_groups", settings.get("critics")))
    if raw is None:
        raise ValueError("Actor-critic runtime settings require critic_routes (or critic_groups/critics).")
    return raw, settings.get("topology"), settings.get("critic_type"), settings.get("max_length", 2048)


def _actor_critic_settings(config: Any) -> dict[str, Any]:
    settings = resolve_actor_critic_settings(config, required=True)
    if settings is None:
        raise RuntimeError("Required actor-critic settings unexpectedly resolved to None.")
    return settings


def _critic_worker_config_for_route(config: Any, route: CriticRouteSpec):
    raw = OmegaConf.create(OmegaConf.to_container(config.critic, resolve=True))
    with open_dict(raw):
        raw.model.path = route.model_path
        raw.model.tokenizer_path = route.tokenizer_path
    critic_cfg = omega_conf_to_dataclass(raw)
    critic_cfg.engine.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
    critic_cfg.engine.max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
    return critic_cfg


def _select_real_critic_rows(batch: Any) -> Any | None:
    data = tq.kv_batch_get(
        keys=batch.keys,
        partition_id=batch.partition_id,
        select_fields=["critic_loss_mask"],
    )
    masks = to_python(data["critic_loss_mask"])
    if not isinstance(masks, list | tuple):
        masks = [masks]
    keys = [key for key, mask in zip(batch.keys, masks, strict=True) if bool(float(mask))]
    return batch.select_keys(keys) if keys else None


def _tq_string_values(batch: Any, field: str) -> list[str]:
    data = tq.kv_batch_get(
        keys=batch.keys,
        partition_id=batch.partition_id,
        select_fields=[field],
    )
    values = to_python(data[field])
    if not isinstance(values, list | tuple):
        values = [values]
    if len(values) != len(batch.keys):
        raise ValueError(f"TQ field {field!r} must have one value per routed row.")
    return [str(value) for value in values]


def _deduplicate_joint_transition_batch(batch: Any) -> Any:
    fields = (
        "traj_uid",
        "joint_transition_ids",
        "critic_group",
        "critic_type",
        "critic_input_ids",
        "critic_attention_mask",
        "critic_position_ids",
        "critic_loss_mask",
        "joint_reward",
        "joint_done",
        "joint_truncated",
    )
    data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)
    rows = [{field: to_python(data[field][row]) for field in fields} for row in range(len(batch.keys))]
    deduplicate_maac_critic_samples(rows)
    indices = deduplicate_maac_critic_indices(rows)
    return batch.select_keys([batch.keys[index] for index in indices])


def _transition_keys(batch: Any) -> list[tuple[str, str]]:
    data = tq.kv_batch_get(
        keys=batch.keys,
        partition_id=batch.partition_id,
        select_fields=["traj_uid", "joint_transition_ids"],
    )
    trajectory_ids = to_python(data["traj_uid"])
    transitions = to_python(data["joint_transition_ids"])
    output = []
    for row, (trajectory_id, values) in enumerate(zip(trajectory_ids, transitions, strict=True)):
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list | tuple) or len(values) != 1:
            raise ValueError(f"Critic row {row} must reference exactly one joint transition.")
        normalized_trajectory = str(trajectory_id)
        transition_id = str(values[0])
        if not normalized_trajectory or not transition_id:
            raise ValueError(f"Critic row {row} requires non-empty traj_uid and joint transition id.")
        output.append((normalized_trajectory, transition_id))
    return output


def _materialize_critic_tq_batch(batch: Any, *, include_targets: bool) -> KVBatchMeta:
    fields = [
        "critic_input_ids",
        "critic_attention_mask",
        "critic_position_ids",
        "critic_loss_mask",
    ]
    if include_targets:
        fields.append("critic_returns")
    data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)
    rows = []
    tags = []
    for row in range(len(batch.keys)):
        ids = torch.as_tensor(to_python(data["critic_input_ids"][row]), dtype=torch.long)
        attention = torch.as_tensor(to_python(data["critic_attention_mask"][row]), dtype=torch.long)
        positions = torch.as_tensor(to_python(data["critic_position_ids"][row]), dtype=torch.long)
        if ids.ndim != 1 or attention.shape != ids.shape or positions.shape != ids.shape or ids.numel() == 0:
            raise ValueError("Critic input ids, attention mask, and position ids must be aligned non-empty sequences.")
        # TransferQueue tracks one dtype per field within a partition. Rollout
        # rows register both masks as int64, so temporary critic rows must keep
        # that contract even though the loss consumes the mask as boolean.
        loss_mask = torch.zeros_like(attention, dtype=torch.long)
        valid = torch.nonzero(attention.bool(), as_tuple=False).flatten()
        if valid.numel() == 0:
            raise ValueError("A real critic row must have at least one attended token.")
        loss_mask[valid[-1]] = float(to_python(data["critic_loss_mask"][row]))
        item = {
            "input_ids": ids,
            "attention_mask": attention,
            "position_ids": positions,
            "loss_mask": loss_mask,
            "response_mask": loss_mask.clone(),
        }
        if include_targets:
            target = float(to_python(data["critic_returns"][row]))
            item["returns"] = loss_mask * target
        rows.append(item)
        tags.append({"seq_len": int(ids.numel()), "response_len": 1, "is_critic": True})
    suffix = uuid.uuid4().hex
    keys = [f"{key}::critic::{suffix}" for key in batch.keys]
    tq.kv_batch_put(
        keys=keys,
        partition_id=batch.partition_id,
        fields=list_of_dict_to_tensordict(rows),
        tags=tags,
    )
    return KVBatchMeta(
        keys=keys,
        tags=tags,
        partition_id=batch.partition_id,
        fields=None,
        extra_info=copy.deepcopy(batch.extra_info),
    )


def _read_critic_scalar_values(batch: KVBatchMeta) -> list[float]:
    data = tq.kv_batch_get(
        keys=batch.keys,
        partition_id=batch.partition_id,
        select_fields=["values", "loss_mask"],
    )
    output = []
    for row in range(len(batch.keys)):
        values = torch.as_tensor(to_python(data["values"][row]), dtype=torch.float32)
        mask = torch.as_tensor(to_python(data["loss_mask"][row]), dtype=torch.bool)
        selected = values[mask]
        if selected.numel() != 1 or not bool(torch.isfinite(selected).all()):
            raise FloatingPointError("Critic inference must produce exactly one finite value per transition.")
        output.append(float(selected.item()))
    return output


def _scalar_values_to_response(values: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Expand one transition value per row to VERL's response-token metric shape."""

    if values.ndim != 1:
        raise ValueError(f"Expected one scalar critic value per row, got shape={tuple(values.shape)}.")
    if getattr(response_mask, "is_nested", False):
        return response_mask.to(dtype=values.dtype) * values.unsqueeze(-1)
    if response_mask.ndim != 2 or response_mask.shape[0] != values.shape[0]:
        raise ValueError(
            "Critic values require a [batch, response] response_mask aligned with scalar rows; "
            f"got values={tuple(values.shape)}, response_mask={tuple(response_mask.shape)}."
        )
    return values.unsqueeze(-1) * response_mask.to(dtype=values.dtype)


__all__ = [
    "CriticRoutedBatch",
    "TrajWeaveMultiActorCriticSyncTrainer",
    "checkpoint_directory_map",
]
