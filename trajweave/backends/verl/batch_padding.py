from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any

import torch


def pad_session_batch(
    *,
    keys: list[str],
    fields: list[dict[str, Any]],
    tags: list[dict[str, Any]],
    multiple: int,
    uid: str,
    session_id: int,
) -> None:
    """用零 loss-mask turn 把动态 MAS 会话补齐到指定倍数。"""

    if multiple <= 1 or not fields:
        return
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for index, field in enumerate(fields):
        grouped_indices[str(field.get("worker_group") or "__default__")].append(index)
    if all(len(indices) % multiple == 0 for indices in grouped_indices.values()):
        return

    negative_index = -1
    for group_id, indices in grouped_indices.items():
        remainder = len(indices) % multiple
        if remainder == 0:
            continue
        for group_pad_index in range(multiple - remainder):
            pad_uid = f"__trajweave_pad__{uid}_{session_id}_{group_id}_{group_pad_index}"
            source_index = indices[0]
            sample = _clone_field(fields[source_index])
            response_mask = torch.zeros_like(sample["response_mask"])
            sample.update(
                uid=pad_uid,
                session_id=group_pad_index,
                agent_name="__padding__",
                role="__padding__",
                agent_id="__padding__",
                traj_uid=pad_uid,
                reward_score=0.0,
                num_turns=0,
                response_mask=response_mask,
                loss_mask=response_mask.clone(),
                rm_scores=torch.zeros_like(sample["rm_scores"], dtype=torch.float32),
            )
            for key, value in {
                "round_id": -1,
                "agent_index": -1,
                "raw_score": 0.0,
                "correctness": 0.0,
                "consensus_reached": False,
                "finished_round": -1,
                "anchor_obs": f"__padding__:{pad_uid}",
                "next_obs": "__padding__",
                "step_reward": 0.0,
                "active_mask": 0.0,
            }.items():
                if key in sample:
                    sample[key] = value
            if "rollout_log_probs" in sample:
                sample["rollout_log_probs"] = torch.zeros_like(sample["rollout_log_probs"], dtype=torch.float32)
            extra_fields = dict(sample.get("extra_fields") or {})
            extra_fields.update(
                {
                    "is_padding": True,
                    "agent_id": "__padding__",
                    "traj_uid": pad_uid,
                    "raw_score": 0.0,
                    "correctness": 0.0,
                    "round_id": -1,
                    "agent_index": -1,
                    "finished_round": -1,
                    "anchor_obs": f"__padding__:{pad_uid}",
                    "next_obs": "__padding__",
                    "step_reward": 0.0,
                    "active_mask": 0.0,
                }
            )
            sample["extra_fields"] = extra_fields

            tag = copy.deepcopy(tags[source_index])
            tag.update(is_padding=True, response_len=0)
            # ReplayBuffer 用首段匹配原 prompt；负 index 避免 VERL 把 padding 当作 session 最终 turn。
            keys.append(f"{uid}_{session_id}_{negative_index}")
            negative_index -= 1
            fields.append(sample)
            tags.append(tag)


def _clone_field(field: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value) for key, value in field.items()
    }
