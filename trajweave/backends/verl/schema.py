from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any

import numpy as np
import torch
from tensordict import NonTensorData, NonTensorStack

DRMAS_AGENT_IDS = {
    "solver": "Solver Agent",
    "verifier": "Verifier Agent",
    "searcher": "Search Agent",
    "search": "Search Agent",
    "answer": "Answer Agent",
}

COMLRL_EXTRA_FIELDS = (
    "prompt_text",
    "response_text",
    "completion_id",
    "tree_node_id",
    "joint_action_ids",
    "joint_return_components",
    "projected_joint_return",
    "joint_advantage",
    "effective_projected_joint_return",
    "joint_transition_ids",
    "joint_reward",
    "joint_done",
    "joint_truncated",
    "joint_stop_reason",
    "joint_sampling_mode",
    "critic_group",
    "critic_type",
    "critic_input_ids",
    "critic_attention_mask",
    "critic_position_ids",
    "critic_loss_mask",
    "preference_pair_id",
    "preference_side",
    "chosen_reward",
    "rejected_reward",
    "candidate_mean",
    "preference_loss_mask",
    "raw_policy_reward",
    "raw_comparator_reward",
    "raw_candidate_rewards",
    "policy_provenance",
    "comparator_provenance",
    "winner_source",
    "loser_source",
)

COMLRL_ID_FIELDS = (
    "completion_id",
    "tree_node_id",
    "critic_group",
    "preference_pair_id",
)

COMLRL_FIELD_ALIASES = {
    "completion_id": "reqs_id",
    "tree_node_id": "node_id",
}

MAS_EXTRA_FIELDS = (
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
    "tool_observation",
    "verifier_decision",
    "memory_snapshot",
    "step_id",
    "reqs_id",
    "parent_reqs_id",
    "is_from_subagent_tool",
    "turn_count",
    "agent_type",
    "role_id",
    "shared_model_id",
    "matpo_tool_format_valid",
    "matpo_tool_call_count",
    "matpo_tool_format_reward",
    "mrlx_turn_role",
    "mrlx_training_mode",
    "mrlx_policy_lag",
    "mrlx_format_valid",
    "mrlx_parent_turn_id",
    "mrlx_outcome_reward",
    "mrlx_role_reward",
    "mrlx_explorer_format_bonus",
    "mrlx_adapter_format_bonus",
    "wideseek_turn_role",
    "wideseek_agent_instance",
    "wideseek_subtrajectory_id",
    "wideseek_agent_count",
    "wideseek_parallel_wave",
    "wideseek_format_valid",
    "wideseek_parent_turn_id",
    "wideseek_outcome_reward",
    "wideseek_trajectory_reward",
    "wideseek_all_formats_valid",
    "wideseek_used_access",
    "wideseek_length_penalty",
    "marshal_episode_id",
    "marshal_player_id",
    "marshal_player_turn",
    "marshal_action",
    "marshal_action_valid",
    "marshal_legal_actions",
    "marshal_format_valid",
    "marshal_turn_reward",
    "marshal_terminal",
    "marshal_terminal_payoff",
    "marshal_shared_policy",
    "policy_version",
    "plan_valid",
    "anchor_obs",
    "next_obs",
    "step_reward",
    "active_mask",
    "discussion_id",
    "interaction_id",
    "comas_stage",
    "solver_agent_id",
    "evaluator_agent_id",
    "scorer_agent_id",
    "generated_score",
    "score_valid",
    "normalized_score",
    "intrinsic_reward",
    "reward_source",
    "tree_id",
    "prompt_id",
    "node_id",
    "parent_idx",
    "path",
    "search_stage",
    "verifier_name",
    "verifier_success",
    "verifier_terminal",
    "search_stop",
    "verifier_feedback",
    "failure_type",
    "verification_mode",
    "verifier_metadata",
    "rollout_policy_step",
    "rollout_global_step",
    "policy_lag",
    "overlong_penalty",
    "root_id",
    "parent_node_id",
    "observation_group_id",
    "branch_index",
    "selected_for_expansion",
    "local_score",
    "c3_group_id",
    "c3_depth",
    "c3_role_index",
    "c3_parent_id",
    "c3_is_leaf",
    "c3_leaf_success",
    "c3_question",
    "c3_prefix_outputs",
    "c3_prefix_prompts",
    "c3_prefix_text",
    "c3_subtree_return",
    "c3_leaf_count",
    "marft_node_id",
    "marft_layer",
    "marft_role_index",
    "marft_transition_message",
    "marft_context_snapshot",
    "marft_credit_strategy",
    "marft_credit_discount",
    "marft_return_gamma",
    "marft_step_reward",
    "marft_projected_return",
) + COMLRL_EXTRA_FIELDS


def comlrl_extra_field_defaults(*, row_id: str, is_padding: bool = False) -> dict[str, Any]:
    prefix = "__padding__" if is_padding else "__missing__"
    unique_prefix = f"{prefix}:{row_id}"
    defaults: dict[str, Any] = {field: f"{unique_prefix}:{field}" for field in COMLRL_ID_FIELDS}
    defaults.update(
        {
            "prompt_text": "",
            "response_text": "",
            "joint_action_ids": [],
            "joint_return_components": [],
            "joint_transition_ids": [],
            "projected_joint_return": 0.0,
            "joint_advantage": 0.0,
            "effective_projected_joint_return": 0.0,
            "joint_reward": 0.0,
            "joint_done": bool(is_padding),
            "joint_truncated": bool(is_padding),
            "joint_stop_reason": "padding" if is_padding else "",
            "joint_sampling_mode": prefix,
            "critic_type": prefix,
            "critic_input_ids": [],
            "critic_attention_mask": [],
            "critic_position_ids": [],
            "critic_loss_mask": 0.0,
            "preference_side": prefix,
            "chosen_reward": 0.0,
            "rejected_reward": 0.0,
            "candidate_mean": 0.0,
            "preference_loss_mask": 0.0,
            "raw_policy_reward": 0.0,
            "raw_comparator_reward": 0.0,
            "raw_candidate_rewards": [],
            "policy_provenance": {},
            "comparator_provenance": {},
            "winner_source": prefix,
            "loser_source": prefix,
        }
    )
    return defaults


def resolve_comlrl_extra_fields(extra_fields: Mapping[str, Any], *, row_id: str) -> dict[str, Any]:
    resolved = comlrl_extra_field_defaults(row_id=row_id)
    for field in COMLRL_EXTRA_FIELDS:
        if field in extra_fields:
            value = to_python(extra_fields[field])
            if value is not None and (field not in COMLRL_ID_FIELDS or value != ""):
                resolved[field] = value
                continue
        alias = COMLRL_FIELD_ALIASES.get(field)
        if alias is not None and alias in extra_fields:
            alias_value = to_python(extra_fields[alias])
            if alias_value is not None and alias_value != "":
                resolved[field] = alias_value
    resolved["joint_action_ids"] = _string_list(resolved["joint_action_ids"], "joint_action_ids")
    resolved["joint_transition_ids"] = _string_list(resolved["joint_transition_ids"], "joint_transition_ids")
    resolved["joint_return_components"] = _finite_float_list(
        resolved["joint_return_components"], "joint_return_components"
    )
    if len(resolved["joint_action_ids"]) != len(set(resolved["joint_action_ids"])):
        raise ValueError("joint_action_ids must be unique")
    if len(resolved["joint_transition_ids"]) != len(set(resolved["joint_transition_ids"])):
        raise ValueError("joint_transition_ids must be unique")
    action_count = len(resolved["joint_action_ids"])
    if len(resolved["joint_transition_ids"]) != action_count:
        raise ValueError("joint_action_ids and joint_transition_ids must have equal lengths")
    if resolved["joint_return_components"] and len(resolved["joint_return_components"]) != action_count:
        raise ValueError("joint_action_ids and joint_return_components must have equal lengths")
    for field in (
        "projected_joint_return",
        "joint_advantage",
        "effective_projected_joint_return",
        "joint_reward",
        "critic_loss_mask",
        "chosen_reward",
        "rejected_reward",
        "candidate_mean",
        "preference_loss_mask",
        "raw_policy_reward",
        "raw_comparator_reward",
    ):
        value = float(resolved[field])
        if not isfinite(value):
            raise ValueError(f"{field} must be finite")
        resolved[field] = value
    resolved["raw_candidate_rewards"] = _finite_float_list(resolved["raw_candidate_rewards"], "raw_candidate_rewards")
    for field in ("policy_provenance", "comparator_provenance"):
        if not isinstance(resolved[field], Mapping):
            raise TypeError(f"{field} must be a mapping")
        resolved[field] = dict(resolved[field])
    return resolved


def _string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list | tuple):
        raise TypeError(f"{field} must be a sequence")
    normalized = [str(item) for item in value]
    if any(not item for item in normalized):
        raise ValueError(f"{field} must contain non-empty values")
    return normalized


def _finite_float_list(value: Any, field: str) -> list[float]:
    if not isinstance(value, list | tuple):
        raise TypeError(f"{field} must be a sequence")
    normalized = [float(item) for item in value]
    if any(not isfinite(item) for item in normalized):
        raise ValueError(f"{field} must contain only finite values")
    return normalized


def padded_rm_scores(response_mask: torch.Tensor, reward_score: float, response_len: int) -> torch.Tensor:
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


def pad_or_trim_1d(
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


def canonical_drmas_agent_id(agent_name: Any) -> str:
    name = str(to_python(agent_name))
    return DRMAS_AGENT_IDS.get(name, name)


def batch_item(value: Any, index: int) -> Any:
    if isinstance(value, torch.Tensor):
        return value[index]
    if isinstance(value, NonTensorStack):
        return value[index].data
    if isinstance(value, NonTensorData):
        return value.data
    if isinstance(value, list | tuple) and len(value) > index:
        return value[index]
    return value


def to_python(value: Any) -> Any:
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf

        if isinstance(value, DictConfig | ListConfig):
            return OmegaConf.to_container(value, resolve=True)
    except Exception:
        pass
    if isinstance(value, NonTensorData):
        return to_python(value.data)
    if isinstance(value, NonTensorStack):
        return [to_python(item.data if hasattr(item, "data") else item) for item in value]
    if isinstance(value, Mapping):
        return {key: to_python(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(to_python(item) for item in value)
    if isinstance(value, list):
        return [to_python(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def flatten_token_ids(token_ids: Any) -> list[int]:
    token_ids = to_python(token_ids)
    if isinstance(token_ids, Mapping):
        if "input_ids" not in token_ids:
            raise ValueError(f"Token mapping does not contain input_ids: {sorted(token_ids)}")
        token_ids = token_ids["input_ids"]
    if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def required_ground_truth(prompt: Mapping[str, Any]) -> str:
    reward_model = to_python(prompt.get("reward_model", {})) or {}
    if not isinstance(reward_model, Mapping):
        raise ValueError("prompt.reward_model must be a mapping containing ground_truth.")
    value = reward_model.get("ground_truth")
    if value is None or not str(value).strip():
        raise ValueError("prompt.reward_model.ground_truth is required and must be non-empty.")
    return str(value)
