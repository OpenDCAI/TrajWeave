from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from trajweave.backends.verl.extensions.common.hooks import PPOExtensionHooks


@dataclass(frozen=True)
class CoMASInteractionREINFORCEHooks(PPOExtensionHooks):
    """CoMAS 交互奖励 -> 按 Worker Group 归一化的 REINFORCE 优势。"""

    name: str = "comas_interaction_reinforce"

    def batch_schema_fields(self, stage: str, config: Any = None) -> tuple[str, ...]:
        if stage != "advantage":
            return ()
        return (
            "agent_id",
            "policy_group",
            "worker_group",
            "worker_group_model_path",
            "traj_uid",
            "turn_id",
            "round_id",
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
        )

    def tq_select_fields(
        self,
        stage: str,
        default_fields: tuple[str, ...],
        config: Any = None,
    ) -> tuple[str, ...]:
        fields = list(default_fields)
        if stage == "advantage":
            fields.extend(self.batch_schema_fields(stage, config=config))
        return tuple(dict.fromkeys(fields))

    def compute_advantage(
        self,
        data: Any,
        *,
        batch_keys: list[str],
        adv_estimator: Any,
        gamma: float,
        lam: float,
        num_repeat: int,
        norm_adv_by_std_in_grpo: bool,
        config: Any = None,
        fallback: Any = None,
    ) -> Any:
        del batch_keys, lam, num_repeat, norm_adv_by_std_in_grpo, fallback

        import torch

        from verl.trainer.ppo.core_algos import AdvantageEstimator
        from verl.trainer.ppo.ray_trainer import compute_response_mask

        if adv_estimator not in {AdvantageEstimator.REINFORCE_PLUS_PLUS, "reinforce_plus_plus"}:
            raise ValueError(
                "CoMAS uses its source-aligned REINFORCE return path and requires "
                f"algorithm.adv_estimator=reinforce_plus_plus as the VERL dispatch marker; got {adv_estimator!r}."
            )
        if "response_mask" not in data.batch:
            data.batch["response_mask"] = compute_response_mask(data)
        missing = [field for field in self.batch_schema_fields("advantage") if field not in data.non_tensor_batch]
        if missing:
            raise KeyError(f"CoMAS advantage requires non_tensor_batch fields: {missing}.")
        if "token_level_rewards" not in data.batch:
            raise KeyError("CoMAS advantage requires token_level_rewards.")

        response_mask = data.batch["response_mask"].to(dtype=torch.bool)
        rewards = data.batch["token_level_rewards"] * response_mask
        intrinsic = torch.tensor(
            [float(value) for value in _batch_values(data.non_tensor_batch["intrinsic_reward"])],
            dtype=rewards.dtype,
            device=rewards.device,
        )
        real_rows = response_mask.any(dim=-1)
        observed = rewards.sum(dim=-1)
        if bool(real_rows.any()) and not torch.allclose(observed[real_rows], intrinsic[real_rows], atol=1e-6, rtol=0):
            raise ValueError(
                "CoMAS token rewards differ from intrinsic interaction rewards. "
                "Ground-truth reward or another reward shaper may have contaminated the training signal."
            )

        returns = _cumulative_returns(rewards, response_mask=response_mask, gamma=float(gamma))
        settings = _config_get(config, "comas", {}) or {}
        normalize = bool(_config_get(settings, "normalize_advantages_by_worker_group", True))
        worker_groups = [str(value) for value in _batch_values(data.non_tensor_batch["worker_group"])]
        advantages, group_stats = _normalize_returns_by_worker_group(
            returns,
            response_mask=response_mask,
            worker_groups=worker_groups,
            enabled=normalize,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        data.meta_info["comas_worker_group_advantage_stats"] = group_stats
        data.meta_info["comas_normalize_advantages_by_worker_group"] = normalize
        return data

    def compute_extra_metrics(self, data: Any, metrics: dict[str, Any], stage: str) -> dict[str, Any]:
        del metrics
        if stage != "advantage":
            return {}
        import torch

        response_mask = data.batch["response_mask"].bool()
        real_rows = response_mask.any(dim=-1).detach().cpu().tolist()
        non_tensors = data.non_tensor_batch
        stages = [str(value) for value in _batch_values(non_tensors["comas_stage"])]
        rewards = [float(value) for value in _batch_values(non_tensors["intrinsic_reward"])]
        valid_scores = [bool(value) for value in _batch_values(non_tensors["score_valid"])]
        interactions = [str(value) for value in _batch_values(non_tensors["interaction_id"])]
        output: dict[str, Any] = {}
        valid_indexes = [index for index, valid in enumerate(real_rows) if valid]
        if valid_indexes:
            output["trajweave/comas/score_valid_rate"] = sum(valid_scores[index] for index in valid_indexes) / len(
                valid_indexes
            )
        for role in ("solver", "evaluator", "scorer"):
            role_rewards = [rewards[index] for index in valid_indexes if stages[index] == role]
            if role_rewards:
                output[f"trajweave/comas/reward/{role}_mean"] = sum(role_rewards) / len(role_rewards)

        grouped: dict[str, list[int]] = defaultdict(list)
        for index in valid_indexes:
            grouped[interactions[index]].append(index)
        violations = 0
        for indexes in grouped.values():
            role_map = {stages[index]: rewards[index] for index in indexes}
            if set(role_map) != {"solver", "evaluator", "scorer"}:
                violations += 1
                continue
            score_is_valid = all(valid_scores[index] for index in indexes)
            expected_sum = 1.0 if score_is_valid else -1.0
            if abs(sum(role_map.values()) - expected_sum) > 1e-6:
                violations += 1
        output["trajweave/comas/interaction_count"] = float(len(grouped))
        output["trajweave/comas/reward_invariant_violations"] = float(violations)

        for group_id, stats in data.meta_info.get("comas_worker_group_advantage_stats", {}).items():
            safe = _safe_metric_name(group_id)
            for name, value in stats.items():
                output[f"trajweave/comas/advantages/{safe}/{name}"] = float(value)
        if "advantages" in data.batch and bool(response_mask.any()):
            values = data.batch["advantages"][response_mask]
            output["trajweave/comas/advantage_abs_mean"] = float(values.abs().mean())
            output["trajweave/comas/advantage_finite"] = float(torch.isfinite(values).all())
        return output


def apply_comas_interaction_reinforce_patch(config: Any = None) -> None:
    """启用 CoMAS 所需的 TQ 字段与权重同步，不改写 VERL 默认算法。"""

    from trajweave.backends.verl.extensions.common.runtime import apply_hook_aware_tq_runtime

    apply_hook_aware_tq_runtime(config)


def _cumulative_returns(rewards: Any, *, response_mask: Any, gamma: float) -> Any:
    import torch

    returns = torch.zeros_like(rewards)
    running = torch.zeros(rewards.shape[0], dtype=rewards.dtype, device=rewards.device)
    for token_index in reversed(range(rewards.shape[1])):
        running = rewards[:, token_index] + gamma * running
        returns[:, token_index] = running
    return returns * response_mask


def _normalize_returns_by_worker_group(
    returns: Any,
    *,
    response_mask: Any,
    worker_groups: list[str],
    enabled: bool,
) -> tuple[Any, dict[str, dict[str, float]]]:
    import torch

    if len(worker_groups) != returns.shape[0]:
        raise ValueError("CoMAS worker_group metadata must have one value per batch row.")
    advantages = torch.zeros_like(returns)
    grouped_rows: dict[str, list[int]] = defaultdict(list)
    for row, group_id in enumerate(worker_groups):
        if bool(response_mask[row].any()):
            grouped_rows[group_id].append(row)

    stats: dict[str, dict[str, float]] = {}
    for group_id, rows in grouped_rows.items():
        row_index = torch.tensor(rows, dtype=torch.long, device=returns.device)
        group_returns = returns[row_index]
        group_mask = response_mask[row_index]
        values = group_returns[group_mask]
        if values.numel() == 0:
            continue
        mean = values.mean()
        std = values.var(unbiased=False).clamp(min=1e-8).sqrt()
        normalized = (group_returns - mean) / std if enabled else group_returns
        advantages[row_index] = normalized * group_mask
        normalized_values = advantages[row_index][group_mask]
        stats[group_id] = {
            "raw_mean": float(mean),
            "raw_std": float(std),
            "mean": float(normalized_values.mean()),
            "std": float(normalized_values.std(unbiased=False)),
            "token_count": float(values.numel()),
        }
    return advantages, stats


def _batch_values(values: Any) -> list[Any]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, list | tuple):
        return list(values)
    return [values]


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    try:
        return config.get(key, default)
    except (AttributeError, TypeError):
        return getattr(config, key, default)


def _safe_metric_name(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in str(value)).strip("_") or "group"
