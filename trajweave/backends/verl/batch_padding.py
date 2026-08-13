from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any

import torch

from trajweave.backends.verl.schema import comlrl_extra_field_defaults


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

    padding_specs: list[tuple[str, int, int]] = []
    for group_id, indices in grouped_indices.items():
        remainder = len(indices) % multiple
        if remainder:
            padding_specs.extend(
                (group_id, group_pad_index, indices[0]) for group_pad_index in range(multiple - remainder)
            )

    padded_reqs_ids: dict[tuple[str, int], str] = {}
    copied_ids_by_source: dict[str, list[str]] = defaultdict(list)
    for group_id, group_pad_index, source_index in padding_specs:
        source = fields[source_index]
        source_extra_fields = source.get("extra_fields") or {}
        source_reqs_id = str(source.get("reqs_id") or source_extra_fields.get("reqs_id") or "")
        if not source_reqs_id:
            continue
        padded_reqs_id = f"{source_reqs_id}__trajweave_pad__{uid}_{session_id}_{group_id}_{group_pad_index}"
        padded_reqs_ids[(group_id, group_pad_index)] = padded_reqs_id
        copied_ids_by_source[source_reqs_id].append(padded_reqs_id)

    negative_index = -1
    for group_id, group_pad_index, source_index in padding_specs:
        pad_uid = f"__trajweave_pad__{uid}_{session_id}_{group_id}_{group_pad_index}"
        source = fields[source_index]
        sample = _clone_field(source)
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
        comlrl_padding_fields = comlrl_extra_field_defaults(row_id=pad_uid, is_padding=True)
        sample.update(comlrl_padding_fields)
        for key, value in {
            "turn_id": -1,
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
            "discussion_id": -1,
            "interaction_id": f"__padding__:{pad_uid}",
            "comas_stage": "__padding__",
            "solver_agent_id": "__padding__",
            "evaluator_agent_id": "__padding__",
            "scorer_agent_id": "__padding__",
            "generated_score": -1,
            "score_valid": False,
            "normalized_score": -1.0,
            "intrinsic_reward": 0.0,
            "reward_source": "__padding__",
            "root_id": pad_uid,
            "node_id": pad_uid,
            "parent_node_id": "",
            "observation_group_id": pad_uid,
            "branch_index": -1,
            "selected_for_expansion": False,
            "local_score": 0.0,
            "c3_group_id": pad_uid,
            "c3_depth": -1,
            "c3_role_index": -1,
            "c3_parent_id": "",
            "c3_is_leaf": False,
            "c3_leaf_success": False,
            "c3_question": "",
            "c3_prefix_outputs": {},
            "c3_prefix_prompts": {},
            "c3_prefix_text": "",
            "c3_subtree_return": 0.0,
            "c3_leaf_count": 0,
        }.items():
            if key in sample:
                sample[key] = value
        if "rollout_log_probs" in sample:
            sample["rollout_log_probs"] = torch.zeros_like(sample["rollout_log_probs"], dtype=torch.float32)

        extra_fields = dict(sample.get("extra_fields") or {})
        source_reqs_id = str(sample.get("reqs_id") or extra_fields.get("reqs_id") or "")
        padded_reqs_id = padded_reqs_ids.get((group_id, group_pad_index), "")
        source_parent_reqs_id = str(sample.get("parent_reqs_id") or extra_fields.get("parent_reqs_id") or "")
        parent_copies = copied_ids_by_source.get(source_parent_reqs_id, [])
        parent_reqs_id = parent_copies[group_pad_index % len(parent_copies)] if parent_copies else source_parent_reqs_id
        if source_reqs_id:
            sample["reqs_id"] = padded_reqs_id
        if source_parent_reqs_id:
            sample["parent_reqs_id"] = parent_reqs_id
        extra_fields.update(comlrl_padding_fields)
        extra_fields.update(
            {
                "is_padding": True,
                "agent_id": "__padding__",
                "traj_uid": pad_uid,
                "turn_id": -1,
                "raw_score": 0.0,
                "correctness": 0.0,
                "round_id": -1,
                "agent_index": -1,
                "finished_round": -1,
                "anchor_obs": f"__padding__:{pad_uid}",
                "next_obs": "__padding__",
                "step_reward": 0.0,
                "active_mask": 0.0,
                "discussion_id": -1,
                "interaction_id": f"__padding__:{pad_uid}",
                "comas_stage": "__padding__",
                "solver_agent_id": "__padding__",
                "evaluator_agent_id": "__padding__",
                "scorer_agent_id": "__padding__",
                "generated_score": -1,
                "score_valid": False,
                "normalized_score": -1.0,
                "intrinsic_reward": 0.0,
                "reward_source": "__padding__",
                "root_id": pad_uid,
                "node_id": pad_uid,
                "parent_node_id": "",
                "observation_group_id": pad_uid,
                "branch_index": -1,
                "selected_for_expansion": False,
                "local_score": 0.0,
                "c3_group_id": pad_uid,
                "c3_depth": -1,
                "c3_role_index": -1,
                "c3_parent_id": "",
                "c3_is_leaf": False,
                "c3_leaf_success": False,
                "c3_question": "",
                "c3_prefix_outputs": {},
                "c3_prefix_prompts": {},
                "c3_prefix_text": "",
                "c3_subtree_return": 0.0,
                "c3_leaf_count": 0,
            }
        )
        if source_reqs_id:
            extra_fields["reqs_id"] = padded_reqs_id
        if source_parent_reqs_id:
            extra_fields["parent_reqs_id"] = parent_reqs_id
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
