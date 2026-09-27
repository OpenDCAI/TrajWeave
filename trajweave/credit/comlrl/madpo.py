from __future__ import annotations

from collections.abc import Mapping
from math import isfinite

import torch
import torch.nn.functional as F


def sequence_logprob_sum(log_probs: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Sum valid response-token log probabilities without length normalization."""

    if not isinstance(log_probs, torch.Tensor) or not isinstance(response_mask, torch.Tensor):
        raise TypeError("log_probs and response_mask must be torch tensors")
    if log_probs.ndim not in {1, 2} or response_mask.shape != log_probs.shape:
        raise ValueError("log_probs and response_mask must be aligned 1D or 2D tensors")
    if not log_probs.is_floating_point():
        raise TypeError("log_probs must be a floating-point tensor")
    if response_mask.dtype is not torch.bool:
        valid_mask_values = torch.logical_or(response_mask == 0, response_mask == 1)
        if not bool(valid_mask_values.all()):
            raise ValueError("response_mask must contain only boolean or 0/1 values")
    mask = response_mask.to(device=log_probs.device, dtype=torch.bool)
    row_mask = mask.unsqueeze(0) if mask.ndim == 1 else mask
    if bool((row_mask.sum(dim=-1) == 0).any()):
        raise ValueError("MADPO rejects empty completions")
    if not bool(torch.isfinite(log_probs[mask]).all()):
        raise FloatingPointError("active response log probabilities must be finite")
    sums = log_probs.masked_fill(~mask, 0).sum(dim=-1)
    if not bool(torch.isfinite(sums).all()):
        raise FloatingPointError("sequence log-probability sums must be finite")
    return sums


def preference_logprob_delta(
    chosen_log_probs: torch.Tensor,
    chosen_response_mask: torch.Tensor,
    rejected_log_probs: torch.Tensor,
    rejected_response_mask: torch.Tensor,
) -> torch.Tensor:
    """Return ``log pi(chosen) - log pi(rejected)`` for each pair."""

    chosen = sequence_logprob_sum(chosen_log_probs, chosen_response_mask)
    rejected = sequence_logprob_sum(rejected_log_probs, rejected_response_mask)
    if chosen.shape != rejected.shape:
        raise ValueError("chosen and rejected batches must contain the same number of pairs")
    delta = chosen - rejected
    _require_finite(delta, "MADPO preference deltas")
    return delta


def snapshot_detached_deltas(deltas_by_actor: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Validate and detach every actor delta from one common policy snapshot."""

    if not deltas_by_actor:
        raise ValueError("MADPO snapshot requires at least one actor")
    snapshot: dict[str, torch.Tensor] = {}
    expected_shape: torch.Size | None = None
    for actor_id, delta in deltas_by_actor.items():
        normalized_actor = str(actor_id)
        if not normalized_actor:
            raise ValueError("MADPO actor identifiers must be non-empty")
        if normalized_actor in snapshot:
            raise ValueError(f"duplicate MADPO actor identifier: {normalized_actor!r}")
        if not isinstance(delta, torch.Tensor) or delta.ndim != 1 or delta.numel() == 0:
            raise ValueError("each MADPO actor snapshot must be a non-empty 1D tensor")
        if expected_shape is None:
            expected_shape = delta.shape
        elif delta.shape != expected_shape:
            raise ValueError("all MADPO actor snapshots must cover the same preference pairs")
        _require_finite(delta, f"MADPO deltas for actor {normalized_actor!r}")
        snapshot[normalized_actor] = delta.detach().clone()
    return snapshot


def madpo_actor_loss(
    own_delta: torch.Tensor,
    snapshot_deltas: Mapping[str, torch.Tensor],
    *,
    actor_id: str,
    beta: float,
    reduction: str = "mean",
) -> torch.Tensor:
    """Reference-free joint logistic loss with gradients only through ``own_delta``."""

    beta_value = float(beta)
    if not isfinite(beta_value) or beta_value <= 0:
        raise ValueError("MADPO beta must be finite and greater than zero")
    if not isinstance(own_delta, torch.Tensor) or own_delta.ndim != 1 or own_delta.numel() == 0:
        raise ValueError("MADPO own_delta must be a non-empty 1D tensor")
    if actor_id not in snapshot_deltas:
        raise KeyError(f"MADPO snapshot is missing actor {actor_id!r}")
    expected_shape = own_delta.shape
    joint_delta = own_delta
    for other_actor, other_delta in snapshot_deltas.items():
        if other_actor == actor_id:
            continue
        if not isinstance(other_delta, torch.Tensor) or other_delta.shape != expected_shape:
            raise ValueError("all MADPO snapshot deltas must align with own_delta")
        joint_delta = joint_delta + other_delta.detach().to(device=own_delta.device, dtype=own_delta.dtype)

    losses = -F.logsigmoid(beta_value * joint_delta)
    if not bool(torch.isfinite(losses).all()):
        return torch.tensor(0.1, dtype=own_delta.dtype, device=own_delta.device, requires_grad=True)
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    if reduction == "mean":
        return losses.mean()
    raise ValueError("MADPO reduction must be one of: none, sum, mean")


def _require_finite(value: torch.Tensor, label: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{label} must be finite")


__all__ = [
    "madpo_actor_loss",
    "preference_logprob_delta",
    "sequence_logprob_sum",
    "snapshot_detached_deltas",
]
