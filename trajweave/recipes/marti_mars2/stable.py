"""MARS² stable-training math, kept separate from fidelity defaults.

The functions operate on PyTorch tensors but do not depend on VERL internals,
which makes the official GSPO/TIS/overlong formulas easy to audit on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from trajweave.credit.tree_grouping import importance_correction_weights

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class StableRecipeConfig:
    loss: Literal["gspo", "ppo"] = "gspo"
    tis_level: Literal["token", "sequence"] = "token"
    tis_mode: Literal["truncate", "mask"] = "truncate"
    tis_threshold: float = 2.0
    overlong_buffer_len: int | None = None
    overlong_penalty_factor: float = 1.0


def gspo_loss(
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    clip_eps: float = 0.2,
    tis_weights: torch.Tensor | None = None,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    """Sequence-level GSPO clipped surrogate.

    Active-token log-ratios form a geometric-mean sequence ratio, which is then
    clipped and reduced at sequence level.
    """
    torch = _torch()
    if old_log_probs.shape != new_log_probs.shape or old_log_probs.shape != response_mask.shape:
        raise ValueError("old_log_probs, new_log_probs, and response_mask must have identical shapes")
    if advantages.ndim not in {1, old_log_probs.ndim}:
        raise ValueError("advantages must be one value per sequence or token")
    if clip_eps < 0:
        raise ValueError("clip_eps must be non-negative")
    mask = response_mask.to(dtype=new_log_probs.dtype)
    # Official MARTI GSPO uses the geometric mean token ratio (mean log-ratio)
    # for the sequence-level surrogate; sequence TIS remains a product below.
    active_count = mask.sum(dim=-1).clamp_min(1.0)
    seq_log_ratio = ((new_log_probs - old_log_probs) * mask).sum(dim=-1) / active_count
    ratio = torch.exp(seq_log_ratio).clamp_min(torch.finfo(new_log_probs.dtype).tiny)
    if tis_weights is not None:
        tis = tis_weights.to(dtype=ratio.dtype)
        ratio = ratio * (tis.mean(dim=-1) if tis.ndim > 1 else tis)
    ratio = ratio.unsqueeze(-1)
    adv = advantages.to(dtype=ratio.dtype)
    if adv.ndim == 1:
        adv = adv.unsqueeze(-1).expand_as(new_log_probs)
    unclipped = ratio * adv
    clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * adv
    token_loss = -torch.minimum(unclipped, clipped)
    sequence_loss = (token_loss * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
    if reduction == "none":
        return sequence_loss
    if reduction == "sum":
        return sequence_loss.sum()
    if reduction != "mean":
        raise ValueError(f"Unsupported reduction: {reduction!r}")
    return sequence_loss.mean()


def compute_tis_weights(
    old_log_probs: torch.Tensor,
    rollout_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    level: Literal["token", "sequence"] = "token",
    mode: Literal["truncate", "mask"] = "truncate",
    threshold: float = 2.0,
) -> torch.Tensor:
    """Compute finite token/sequence truncated importance weights."""
    torch = _torch()
    if old_log_probs.shape != rollout_log_probs.shape or old_log_probs.shape != response_mask.shape:
        raise ValueError("TIS tensors must have identical shapes")
    rows = importance_correction_weights(
        old_logprobs=old_log_probs.detach().float().cpu().tolist(),
        rollout_logprobs=rollout_log_probs.detach().float().cpu().tolist(),
        action_mask=response_mask.detach().cpu().tolist(),
        level=level,
        mode=mode,
        upper_threshold=threshold,
    )
    return torch.tensor(rows, dtype=old_log_probs.dtype, device=old_log_probs.device)


def effective_sample_size(weights: torch.Tensor, response_mask: torch.Tensor | None = None) -> torch.Tensor:
    torch = _torch()
    values = weights if response_mask is None else weights[response_mask.to(dtype=torch.bool)]
    values = values.float()
    if values.numel() == 0:
        return values.new_zeros(())
    denominator = values.square().sum()
    return values.sum().square() / denominator.clamp_min(torch.finfo(values.dtype).tiny)


def apply_overlong_penalty(
    raw_scores: torch.Tensor,
    response_lengths: torch.Tensor,
    *,
    max_response_length: int,
    buffer_len: int,
    penalty_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the DAPO/MARS² overlong penalty before tree advantage shaping."""
    if buffer_len <= 0 or max_response_length < buffer_len:
        raise ValueError("max_response_length must be >= positive buffer_len")
    expected_len = max_response_length - buffer_len
    exceed = (response_lengths.to(dtype=raw_scores.dtype) - expected_len).clamp_min(0).clamp_max(buffer_len)
    penalty = -exceed / float(buffer_len) * float(penalty_factor)
    return raw_scores + penalty, penalty


def stable_acceptance(
    *,
    metrics: dict[str, float | int | bool],
    require_two_actors: bool = True,
    max_policy_lag: int = 1,
) -> dict[str, object]:
    """Evaluate mechanism-level stable-training acceptance without score claims."""
    checks = {
        "gspo_active": bool(metrics.get("gspo_active", False)),
        "tis_finite": bool(metrics.get("tis_finite", False)),
        "ess_nonzero": float(metrics.get("ess", 0.0)) > 0,
        "overlong_penalty_nonzero": abs(float(metrics.get("overlong_penalty", 0.0))) > 0,
        "policy_lag_within_threshold": int(metrics.get("max_policy_lag", 0)) <= max_policy_lag,
        "weight_sync_complete": int(metrics.get("pending", metrics.get("weight_sync_pending", 1))) == 0,
    }
    if require_two_actors:
        checks["both_actors_updated"] = int(metrics.get("actors_updated", 0)) >= 2
    passed = all(checks.values())
    return {"status": "passed" if passed else "failed", "checks": checks, "passed": passed}


__all__ = [
    "StableRecipeConfig",
    "apply_overlong_penalty",
    "compute_tis_weights",
    "effective_sample_size",
    "gspo_loss",
    "stable_acceptance",
    "build_stable_launch_overrides",
    "run_stable_cpu_fixture",
]


def _torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("stable MARS² tensor math requires PyTorch") from exc
    return torch


def build_stable_launch_overrides(config: dict, *, config_path: str | None = None) -> tuple[str, ...]:
    """Build an isolated stable recipe contract without changing fidelity defaults."""
    stable = config.get("stable", {}) or {}
    tis_level = str(stable.get("tis_level", "token"))
    if tis_level not in {"token", "sequence"}:
        raise ValueError("stable.tis_level must be token or sequence")
    loss = str(stable.get("loss", "gspo")).lower()
    if loss not in {"gspo", "ppo"}:
        raise ValueError("stable.loss must be gspo or ppo")
    overrides = [str(item) for item in (config.get("verl", {}) or {}).get("overrides", [])]
    required = [
        "algorithm.adv_estimator=grpo",
        f"actor_rollout_ref.actor.policy_loss.loss_mode={loss}",
        "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean",
        f"+trajweave.stable.loss={loss}",
        f"+trajweave.stable.tis_level={tis_level}",
        f"+trajweave.stable.tis_mode={str(stable.get('tis_mode', 'truncate'))}",
        f"+trajweave.stable.tis_threshold={float(stable.get('tis_threshold', 2.0))}",
        f"+trajweave.stable.overlong_buffer_len={stable.get('overlong_buffer_len', 1)}",
        f"+trajweave.stable.overlong_penalty_factor={float(stable.get('overlong_penalty_factor', 1.0))}",
        "++reward.reward_kwargs.overlong_buffer_cfg="
        f"{{enable:true,len:{int(stable.get('overlong_buffer_len', 1))},"
        f"penalty_factor:{float(stable.get('overlong_penalty_factor', 1.0))},log:true}}",
        "actor_rollout_ref.rollout.calculate_log_probs=true",
        "algorithm.rollout_correction.bypass_mode=false",
        f"algorithm.rollout_correction.rollout_is={tis_level}",
        f"algorithm.rollout_correction.rollout_is_threshold={float(stable.get('tis_threshold', 2.0))}",
    ]
    if config_path:
        required.append(f"+trajweave.config={config_path}")
    return tuple(overrides + required)


def run_stable_cpu_fixture(
    config: dict | None = None,
    *,
    require_two_actors: bool = False,
) -> dict[str, object]:
    """Run a deterministic mechanism smoke for GSPO, TIS and overlong penalty.

    The fixture deliberately uses tiny tensors and does not invoke VERL.  It
    is useful for validating the stable recipe contract in CI and for checking
    that a launch configuration selects the requested TIS level/mode.
    """

    torch = _torch()
    stable = (config or {}).get("stable", {}) or {}
    level = str(stable.get("tis_level", "token"))
    mode = str(stable.get("tis_mode", "truncate"))
    threshold = float(stable.get("tis_threshold", 2.0))
    old = torch.zeros((2, 3), dtype=torch.float32)
    new = torch.tensor([[0.1, -0.1, 0.05], [-0.2, 0.1, 0.0]], dtype=torch.float32)
    rollout = torch.tensor([[0.0, -0.2, 0.0], [-0.1, 0.0, 0.0]], dtype=torch.float32)
    mask = torch.ones_like(old)
    tis = compute_tis_weights(old, rollout, mask, level=level, mode=mode, threshold=threshold)
    loss = gspo_loss(old, new, torch.tensor([1.0, -1.0]), mask, tis_weights=tis)
    buffer_len = int(stable.get("overlong_buffer_len", 2))
    max_response_length = max(8, buffer_len * 2)
    shaped, penalty = apply_overlong_penalty(
        torch.tensor([0.5, 0.2]),
        torch.tensor([max_response_length // 2, max_response_length]),
        max_response_length=max_response_length,
        buffer_len=buffer_len,
        penalty_factor=float(stable.get("overlong_penalty_factor", 1.0)),
    )
    metrics = {
        "gspo_active": str(stable.get("loss", "gspo")).lower() == "gspo",
        "tis_finite": bool(torch.isfinite(tis).all().item()),
        "ess": float(effective_sample_size(tis, mask).item()),
        "overlong_penalty": float(penalty.mean().item()),
        "loss": float(loss.item()),
        "shaped_reward_mean": float(shaped.mean().item()),
        "actors_updated": 2,
        "max_policy_lag": 0,
        "pending": 0,
        "tis_level": level,
        "tis_mode": mode,
    }
    return {
        "metrics": metrics,
        "acceptance": stable_acceptance(
            metrics=metrics,
            require_two_actors=require_two_actors,
            max_policy_lag=int(stable.get("max_policy_lag", 1)),
        ),
    }
