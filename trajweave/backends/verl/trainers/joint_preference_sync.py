from __future__ import annotations

import copy
import os
import uuid
from dataclasses import dataclass
from functools import partial
from math import gcd, isfinite, lcm
from typing import Any

import torch
import transfer_queue as tq
from tensordict import TensorDict
from transfer_queue import KVBatchMeta

from trajweave.backends.verl.extensions.comlrl.preference import (
    PREFERENCE_TQ_FIELDS,
    detached_other_agent_deltas,
    madpo_actor_loss,
    materialize_pair_response_logprobs,
    validate_preference_pairs,
)
from trajweave.backends.verl.schema import batch_item, to_python
from trajweave.backends.verl.trainers.multi_actor_sync import (
    TrajWeaveMultiActorSyncTrainer,
    _batch_len,
    _safe_metric_name,
)
from trajweave.credit.comlrl.madpo import snapshot_detached_deltas
from verl.protocol import DataProtoFuture
from verl.trainer.ppo.padding_utils import construct_minimal_padding_template
from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.tensordict_utils import list_of_dict_to_tensordict
from verl.workers.utils.padding import response_to_nested


@dataclass(frozen=True)
class MADPOLossConfig:
    madpo_beta: float


@register_trainer("trajweave_joint_preference_sync")
class TrajWeaveJointPreferenceSyncTrainer(TrajWeaveMultiActorSyncTrainer):
    """Sequential multi-actor MADPO updates from one detached policy snapshot."""

    def _validate_multi_actor_specs(self) -> None:
        super()._validate_multi_actor_specs()
        validate_joint_preference_training_contract(
            trainable_actor_groups=self.multi_actor_trainable_group_ids,
            use_critic=self.use_critic,
            use_reference_policy=self.use_reference_policy,
            use_kl_in_reward=bool(self.config.algorithm.get("use_kl_in_reward", False)),
            ppo_epochs=int(self.config.actor_rollout_ref.actor.ppo_epochs),
            beta=_madpo_beta(self.config),
            reward_model_enabled=bool(_select(self.config, "reward.reward_model.enable") or False),
        )
        rollout_correction = self.config.algorithm.get("rollout_correction", None)
        if rollout_correction:
            raise ValueError("MADPO does not support PPO rollout correction or importance ratios")
        if bool(_select(self.config, "reward.reward_model.enable")):
            raise ValueError("MADPO uses task preference rewards and does not support a colocated reward model")

    def _create_worker_groups(self) -> None:
        super()._create_worker_groups()
        loss_config = MADPOLossConfig(madpo_beta=_madpo_beta(self.config))
        for group_id in self.multi_actor_trainable_group_ids:
            worker_group = self.actor_rollout_wgs[group_id]
            if not callable(getattr(worker_group, "set_loss_fn", None)):
                raise RuntimeError("VERL actor worker does not expose set_loss_fn; refusing to run MADPO with PPO loss")
            worker_group.set_loss_fn(partial(madpo_actor_loss, loss_config))

    def _balance_batch(self, batch, metrics: dict, logging_prefix="global_seqlen", keep_minibatch=False):
        """Keep chosen/rejected rows atomic in every actor data-parallel shard."""

        del logging_prefix, keep_minibatch
        fields = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=PREFERENCE_TQ_FIELDS,
        )
        active_masks = [
            float(to_python(batch_item(fields["preference_loss_mask"], row))) for row in range(len(batch.keys))
        ]
        if active_masks and not any(value != 0.0 for value in active_masks):
            batch.extra_info["madpo_skip_update"] = True
            metrics["trajweave/madpo/skipped_all_tied_batch"] = 1
            metrics["trajweave/madpo/padding_pairs"] = 0
            metrics["trajweave/madpo/balanced_rows"] = len(batch.keys)
            return batch
        batch.extra_info["madpo_skip_update"] = False
        pairs = validate_preference_pairs(
            fields,
            expected_worker_groups=self.multi_actor_trainable_group_ids,
        )
        pairs_by_group = {
            group_id: sorted(
                (pair for pair in pairs if pair.worker_group == group_id),
                key=lambda pair: pair.preference_pair_id,
            )
            for group_id in self.multi_actor_trainable_group_ids
        }
        active_rows = {row for pair in pairs for row in (pair.chosen_row, pair.rejected_row)}
        inactive_rows_by_group = {group_id: [] for group_id in self.multi_actor_trainable_group_ids}
        for row, key in enumerate(batch.keys):
            if row in active_rows:
                continue
            loss_mask = float(to_python(fields["preference_loss_mask"][row]))
            if loss_mask != 0.0:
                raise ValueError(f"MADPO row {key!r} is neither an active pair row nor zero-loss padding")
            group_id = str(to_python(fields["worker_group"][row]))
            if group_id not in inactive_rows_by_group:
                raise ValueError(f"MADPO padding row {key!r} references unknown worker group {group_id!r}")
            inactive_rows_by_group[group_id].append(row)

        ordered_keys: list[str] = []
        ordered_tags: list[dict[str, Any]] = []
        padding_keys: list[str] = []
        padding_tags: list[dict[str, Any]] = []
        padding_fields: list[dict[str, Any]] = []
        padding_pair_count = 0

        for group_id in self.multi_actor_trainable_group_ids:
            group_pairs = pairs_by_group[group_id]
            worker_group = self._actor_wg(group_id)
            dp_size = _madpo_actor_dp_size(worker_group)
            if len(group_pairs) < dp_size:
                raise ValueError(
                    "MADPO requires at least one active preference pair per actor data-parallel rank; "
                    f"group={group_id!r}, pairs={len(group_pairs)}, dp_size={dp_size}."
                )
            world_size = int(worker_group.world_size)
            pair_multiple = lcm(dp_size, world_size // gcd(world_size, 2))
            source_pair = group_pairs[0]
            source_row = source_pair.chosen_row
            source_td = tq.kv_batch_get(
                keys=[batch.keys[source_row]],
                partition_id=batch.partition_id,
            )[0]
            template, template_tag = construct_minimal_padding_template(
                source_td,
                batch.tags[source_row],
                self.tokenizer.eos_token_id,
            )
            existing_padding_pairs = _materialize_existing_padding_pairs(
                batch=batch,
                fields=fields,
                group_id=group_id,
                row_indices=inactive_rows_by_group[group_id],
                template=template,
                template_tag=template_tag,
                padding_keys=padding_keys,
                padding_tags=padding_tags,
                padding_fields=padding_fields,
            )
            minimum_pair_count = len(group_pairs) + len(existing_padding_pairs)
            balanced_pair_count = ((minimum_pair_count + pair_multiple - 1) // pair_multiple) * pair_multiple
            pairs_per_rank = balanced_pair_count // dp_size
            required_padding_pairs = balanced_pair_count - len(group_pairs)
            while len(existing_padding_pairs) < required_padding_pairs:
                existing_padding_pairs.append(
                    _append_padding_pair(
                        group_id=group_id,
                        template=template,
                        template_tag=template_tag,
                        padding_keys=padding_keys,
                        padding_tags=padding_tags,
                        padding_fields=padding_fields,
                    )
                )
            if len(existing_padding_pairs) != required_padding_pairs:
                raise RuntimeError("MADPO padding allocation exceeded the balanced actor batch size")

            base_pairs, extra_pairs = divmod(len(group_pairs), dp_size)
            active_pair_offset = 0
            padding_pair_offset = 0
            for dp_rank in range(dp_size):
                active_pair_count = base_pairs + int(dp_rank < extra_pairs)
                rank_pairs = group_pairs[active_pair_offset : active_pair_offset + active_pair_count]
                active_pair_offset += active_pair_count
                for pair in rank_pairs:
                    for row in (pair.chosen_row, pair.rejected_row):
                        ordered_keys.append(batch.keys[row])
                        ordered_tags.append(batch.tags[row])
                rank_padding_count = pairs_per_rank - active_pair_count
                rank_padding = existing_padding_pairs[padding_pair_offset : padding_pair_offset + rank_padding_count]
                padding_pair_offset += rank_padding_count
                for pair_keys, pair_tags in rank_padding:
                    ordered_keys.extend(pair_keys)
                    ordered_tags.extend(pair_tags)
            if active_pair_offset != len(group_pairs) or padding_pair_offset != len(existing_padding_pairs):
                raise RuntimeError(f"MADPO balancing lost preference pairs for actor group {group_id!r}")
            padding_pair_count += len(existing_padding_pairs)

        if padding_keys:
            tq.kv_batch_put(
                keys=padding_keys,
                partition_id=batch.partition_id,
                fields=list_of_dict_to_tensordict(padding_fields),
                tags=padding_tags,
            )
        metrics["trajweave/madpo/padding_pairs"] = padding_pair_count
        metrics["trajweave/madpo/balanced_rows"] = len(ordered_keys)
        return KVBatchMeta(
            keys=ordered_keys,
            tags=ordered_tags,
            partition_id=batch.partition_id,
            fields=batch.fields,
            extra_info=batch.extra_info,
        )

    def _compute_old_log_prob(self, batch, metrics: dict):
        """Compute all actor deltas before any actor optimizer step and cache them detached."""

        if bool(batch.extra_info.get("madpo_skip_update", False)):
            self._madpo_snapshot_deltas = {}
            self._madpo_snapshot_pair_ids = ()
            batch.extra_info["madpo_pair_count"] = 0
            batch.extra_info["madpo_global_pair_count"] = 0
            metrics["trajweave/madpo/pair_count"] = 0
            return batch

        batch.extra_info.update(
            {
                "calculate_entropy": False,
                "compute_loss": False,
                "temperature": self.config.actor_rollout_ref.rollout.temperature,
            }
        )
        routed_batches = self._route_batch(batch)
        for routed in routed_batches:
            output = self._actor_wg(routed.group_id).compute_log_prob(routed.batch)
            if len(output) != len(routed.batch):
                raise RuntimeError(f"MADPO log-probability output size mismatch for actor {routed.group_id!r}")

        fields = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=(*PREFERENCE_TQ_FIELDS, "log_probs"),
        )
        materialized = materialize_pair_response_logprobs(
            fields,
            expected_worker_groups=self.multi_actor_trainable_group_ids,
        )
        pair_ids = sorted({pair.preference_pair_id for pair in materialized.pairs})
        raw_by_actor = {
            group_id: torch.stack([materialized.deltas[(pair_id, group_id)] for pair_id in pair_ids])
            for group_id in self.multi_actor_trainable_group_ids
        }
        snapshot = snapshot_detached_deltas(raw_by_actor)
        keyed_snapshot = {
            (pair_id, group_id): snapshot[group_id][pair_index]
            for pair_index, pair_id in enumerate(pair_ids)
            for group_id in self.multi_actor_trainable_group_ids
        }
        other_deltas = detached_other_agent_deltas(keyed_snapshot)
        self._madpo_snapshot_deltas = snapshot
        self._madpo_snapshot_pair_ids = tuple(pair_ids)

        row_count = len(batch.keys)
        pair_index_by_id = {pair_id: index for index, pair_id in enumerate(pair_ids)}
        pair_indexes = torch.zeros(row_count, dtype=torch.long)
        side_values = torch.zeros(row_count, dtype=torch.float32)
        other_values = torch.zeros(row_count, dtype=materialized.sequence_logps.dtype)
        old_log_probs = (
            response_to_nested(materialized.response_log_probs, fields["response_mask"])
            if getattr(fields["response_mask"], "is_nested", False)
            else materialized.response_log_probs
        )
        for pair in materialized.pairs:
            pair_index = pair_index_by_id[pair.preference_pair_id]
            other = other_deltas[(pair.preference_pair_id, pair.worker_group)].detach().cpu()
            pair_indexes[pair.chosen_row] = pair_index
            pair_indexes[pair.rejected_row] = pair_index
            side_values[pair.chosen_row] = 1.0
            side_values[pair.rejected_row] = -1.0
            other_values[pair.chosen_row] = other
            other_values[pair.rejected_row] = other

        tq.kv_batch_put(
            keys=batch.keys,
            partition_id=batch.partition_id,
            fields=TensorDict(
                {
                    "old_log_probs": old_log_probs,
                    "madpo_pair_index": pair_indexes,
                    "madpo_preference_side": side_values,
                    "madpo_other_agent_delta": other_values,
                },
                batch_size=row_count,
            ),
        )
        batch.extra_info["madpo_pair_count"] = len(pair_ids)
        batch.extra_info["madpo_global_pair_count"] = len(pair_ids)
        metrics["trajweave/madpo/pair_count"] = len(pair_ids)
        return batch

    def _compute_advantage(self, batch, metrics: dict):
        """Materialize metric-compatible zero tensors for the non-PPO MADPO objective."""

        del metrics
        if not hasattr(self, "_madpo_snapshot_deltas"):
            raise RuntimeError("MADPO actor deltas must be snapshotted before the update phase")
        data = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=["response_mask"],
        )
        zeros = torch.zeros_like(data["response_mask"], dtype=torch.float32)
        tq.kv_batch_put(
            keys=batch.keys,
            partition_id=batch.partition_id,
            fields=TensorDict(
                {"advantages": zeros, "returns": zeros.clone()},
                batch_size=len(batch.keys),
            ),
        )
        return batch

    def _update_actor(self, batch, metrics: dict):
        if bool(batch.extra_info.get("madpo_skip_update", False)):
            metrics["trajweave/madpo/actor_groups/updated"] = 0
            metrics["trajweave/madpo/pair_count"] = 0
            return batch
        if not hasattr(self, "_madpo_snapshot_deltas"):
            raise RuntimeError("MADPO cannot update actors without a detached pre-update snapshot")
        routed_by_group = {routed.group_id: routed.batch for routed in self._route_batch(batch)}
        missing = [group for group in self.multi_actor_trainable_group_ids if group not in routed_by_group]
        if missing:
            raise ValueError(f"MADPO batch is missing trainable actor groups: {missing}")

        actor_config = self.config.actor_rollout_ref.actor
        updated = 0
        for group_id in self.multi_actor_trainable_group_ids:
            routed_batch = routed_by_group[group_id]
            routed_batch.extra_info.update(batch.extra_info)
            routed_batch.extra_info.update(
                {
                    "calculate_entropy": False,
                    "distillation_use_topk": False,
                    "global_batch_size": _batch_len(routed_batch),
                    "mini_batch_size": None,
                    "num_mini_batch": 1,
                    "epochs": 1,
                    "seed": actor_config.data_loader_seed,
                    "dataloader_kwargs": {"shuffle": False},
                    "force_group_size": 2,
                    "temperature": self.config.actor_rollout_ref.rollout.temperature,
                }
            )
            output = _resolve_actor_update_output(self._actor_wg(group_id).update_actor(routed_batch))
            self._record_actor_update(group_id)
            renamed = rename_dict(output["metrics"], f"actor/{group_id}/")
            metrics.update(reduce_metrics(renamed))
            metrics[f"trajweave/madpo/actor_groups/{_safe_metric_name(group_id)}/updated"] = 1
            updated += 1

        if updated != len(self.multi_actor_trainable_group_ids):
            raise RuntimeError("MADPO did not update every trainable actor exactly once")
        metrics["trajweave/madpo/actor_groups/updated"] = updated
        metrics["trajweave/madpo/pair_count"] = int(batch.extra_info.get("madpo_pair_count", 0))
        return batch

    def _load_checkpoint(self):
        self.global_steps = 0
        mode = str(self.config.trainer.resume_mode)
        if mode == "disable":
            return
        resume_from_path = self.config.trainer.get("resume_from_path", None)
        marker = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        if mode == "auto" and not resume_from_path and not os.path.exists(marker):
            return
        raise ValueError(
            "MADPO multi-actor checkpoint resume is not implemented; "
            f"resume_mode={mode!r}, resume_from_path={resume_from_path!r}, "
            f"marker_exists={os.path.exists(marker)}"
        )

    def _metric_namespace(self) -> str:
        return "madpo"


def _materialize_existing_padding_pairs(
    *,
    batch: Any,
    fields: Any,
    group_id: str,
    row_indices: list[int],
    template: dict[str, Any],
    template_tag: dict[str, Any],
    padding_keys: list[str],
    padding_tags: list[dict[str, Any]],
    padding_fields: list[dict[str, Any]],
) -> list[tuple[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]]]:
    rows_by_pair: dict[str, list[int]] = {}
    for row in row_indices:
        pair_id = str(to_python(batch_item(fields["preference_pair_id"], row)))
        rows_by_pair.setdefault(pair_id, []).append(row)

    output = []
    consumed: set[int] = set()
    for rows in rows_by_pair.values():
        sides = {str(to_python(batch_item(fields["preference_side"], row))).strip().lower(): row for row in rows}
        if len(rows) == 2 and set(sides) == {"chosen", "rejected"}:
            chosen_row = sides["chosen"]
            rejected_row = sides["rejected"]
            output.append(
                _append_padding_pair(
                    group_id=group_id,
                    template=template,
                    template_tag=template_tag,
                    padding_keys=padding_keys,
                    padding_tags=padding_tags,
                    padding_fields=padding_fields,
                    existing_keys=(batch.keys[chosen_row], batch.keys[rejected_row]),
                    existing_tags=(batch.tags[chosen_row], batch.tags[rejected_row]),
                )
            )
            consumed.update(rows)

    for row in row_indices:
        if row in consumed:
            continue
        existing_side = str(to_python(batch_item(fields["preference_side"], row))).strip().lower()
        existing_side = existing_side if existing_side in {"chosen", "rejected"} else "chosen"
        keys: list[str | None] = [None, None]
        tags: list[dict[str, Any] | None] = [None, None]
        side_index = 0 if existing_side == "chosen" else 1
        keys[side_index] = batch.keys[row]
        tags[side_index] = batch.tags[row]
        output.append(
            _append_padding_pair(
                group_id=group_id,
                template=template,
                template_tag=template_tag,
                padding_keys=padding_keys,
                padding_tags=padding_tags,
                padding_fields=padding_fields,
                existing_keys=(keys[0], keys[1]),
                existing_tags=(tags[0], tags[1]),
            )
        )
    return output


def _append_padding_pair(
    *,
    group_id: str,
    template: dict[str, Any],
    template_tag: dict[str, Any],
    padding_keys: list[str],
    padding_tags: list[dict[str, Any]],
    padding_fields: list[dict[str, Any]],
    existing_keys: tuple[str | None, str | None] = (None, None),
    existing_tags: tuple[dict[str, Any] | None, dict[str, Any] | None] = (None, None),
) -> tuple[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]]:
    pair_token = uuid.uuid4().hex
    padding_pair_id = f"__padding__:{group_id}:{pair_token}"
    pad_uid = f"madpo-pad-{pair_token}"
    pair_keys: list[str] = []
    pair_tags: list[dict[str, Any]] = []
    for side_index, side in enumerate(("chosen", "rejected")):
        sample = copy.deepcopy(template)
        metadata = {
            "uid": pad_uid,
            "worker_group": group_id,
            "preference_pair_id": padding_pair_id,
            "preference_side": side,
            "chosen_reward": 0.0,
            "rejected_reward": 0.0,
            "preference_loss_mask": 0.0,
        }
        sample.update(metadata)
        if "extra_fields" in sample:
            extra_fields = dict(sample.get("extra_fields") or {})
            extra_fields.update(metadata)
            sample["extra_fields"] = extra_fields
        key = existing_keys[side_index] or f"{pad_uid}-{side}"
        tag = copy.deepcopy(existing_tags[side_index] or template_tag)
        tag.update(is_padding=True, response_len=1, seq_len=2)
        padding_keys.append(key)
        padding_tags.append(tag)
        padding_fields.append(sample)
        pair_keys.append(key)
        pair_tags.append(tag)
    return (pair_keys[0], pair_keys[1]), (pair_tags[0], pair_tags[1])


def _madpo_actor_dp_size(worker_group: Any) -> int:
    dispatch_info = getattr(worker_group, "_dispatch_info", None)
    query_dispatch_info = getattr(worker_group, "_query_dispatch_info", None)
    if isinstance(dispatch_info, dict) and callable(query_dispatch_info):
        if "actor" not in dispatch_info:
            dispatch_info["actor"] = query_dispatch_info("actor")
        mapping = dispatch_info["actor"]
        if mapping:
            return max(int(rank) for rank in mapping) + 1
    world_size = int(worker_group.world_size)
    if world_size < 1:
        raise ValueError("MADPO actor worker-group world_size must be positive")
    return world_size


def _resolve_actor_update_output(output: Any) -> dict | TensorDict:
    if isinstance(output, dict | TensorDict):
        return output
    if isinstance(output, DataProtoFuture):
        resolved = output.get()
        if isinstance(resolved, dict | TensorDict):
            return resolved
        raise TypeError(f"MADPO actor update future resolved to unsupported type: {type(resolved)!r}")
    raise TypeError(f"Unsupported MADPO actor update output type: {type(output)!r}")


def validate_joint_preference_training_contract(
    *,
    trainable_actor_groups: list[str] | tuple[str, ...],
    use_critic: bool,
    use_reference_policy: bool,
    use_kl_in_reward: bool,
    ppo_epochs: int,
    beta: float,
    reward_model_enabled: bool = False,
) -> None:
    groups = tuple(str(group) for group in trainable_actor_groups)
    if len(groups) < 2 or len(groups) != len(set(groups)) or any(not group for group in groups):
        raise ValueError("MADPO requires at least two unique trainable actor groups")
    if use_critic:
        raise ValueError("MADPO does not use a critic")
    if use_reference_policy or use_kl_in_reward:
        raise ValueError("CoMLRL v1.4.1 MADPO is reference-free")
    if reward_model_enabled:
        raise ValueError("MADPO does not support a colocated reward model")
    if isinstance(ppo_epochs, bool) or int(ppo_epochs) != 1:
        raise ValueError("MADPO requires ppo_epochs=1 for one sequential update per batch")
    beta_value = float(beta)
    if not isfinite(beta_value) or beta_value <= 0:
        raise ValueError("MADPO beta must be finite and greater than zero")


def _madpo_beta(config: Any) -> float:
    value = _select(config, "trajweave.comlrl.madpo.beta")
    if value is None:
        value = _select(config, "trajweave.madpo.beta")
    if value is None:
        value = 0.1
    beta = float(to_python(value))
    if not isfinite(beta) or beta <= 0:
        raise ValueError("MADPO beta must be finite and greater than zero")
    return beta


def _select(config: Any, path: str) -> Any:
    try:
        from omegaconf import OmegaConf

        value = OmegaConf.select(config, path)
        if value is not None:
            return value
    except (AttributeError, TypeError, ValueError):
        pass
    current = config
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            try:
                current = current.get(part)
            except (AttributeError, TypeError):
                current = getattr(current, part, None)
        if current is None:
            return None
    return current


__all__ = [
    "MADPOLossConfig",
    "TrajWeaveJointPreferenceSyncTrainer",
    "validate_joint_preference_training_contract",
]
