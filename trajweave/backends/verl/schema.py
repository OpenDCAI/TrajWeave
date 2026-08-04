from __future__ import annotations

from collections.abc import Mapping
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
    "verifier_decision",
    "memory_snapshot",
    "step_id",
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
)


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
