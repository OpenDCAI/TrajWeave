from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from trajweave.backends.verl.extensions.common.hooks import PPOExtensionHooks


@dataclass(frozen=True)
class MARSHALHooks(PPOExtensionHooks):
    name: str = "marshal_turn_level_reinforce"

    def batch_schema_fields(self, stage: str, config: Any = None) -> tuple[str, ...]:
        if stage != "advantage":
            return ()
        return (
            "agent_id",
            "traj_uid",
            "turn_id",
            "marshal_episode_id",
            "marshal_player_id",
            "marshal_player_turn",
            "marshal_turn_reward",
            "active_mask",
        )

    def tq_select_fields(
        self,
        stage: str,
        default_fields: tuple[str, ...] | None = None,
        config: Any = None,
    ) -> tuple[str, ...]:
        fields = list(default_fields or ())
        if stage == "advantage":
            fields.extend(self.batch_schema_fields(stage, config=config))
        return tuple(dict.fromkeys(fields))

    def compute_advantage(self, data: Any, **kwargs: Any) -> Any:
        import numpy as np
        import torch

        from trajweave.credit.marshal import compute_marshal_scalar_advantages
        from verl.trainer.ppo import core_algos
        from verl.trainer.ppo.ray_trainer import compute_response_mask

        estimator = kwargs.get("adv_estimator")
        if estimator not in {
            core_algos.AdvantageEstimator.REINFORCE_PLUS_PLUS,
            core_algos.AdvantageEstimator.REINFORCE_PLUS_PLUS.value,
        }:
            raise ValueError(f"MARSHAL requires the critic-free REINFORCE++ trainer path, got {estimator!r}.")
        if "response_mask" not in data.batch:
            data.batch["response_mask"] = compute_response_mask(data)
        missing = [field for field in self.batch_schema_fields("advantage") if field not in data.non_tensor_batch]
        if missing:
            raise KeyError(f"MARSHAL advantage requires non_tensor_batch fields: {missing}.")

        response_mask = data.batch["response_mask"]
        row_count = response_mask.shape[0]
        non_tensors = data.non_tensor_batch
        active_values = [
            bool(float(value))
            for value in _batch_values(non_tensors["active_mask"], row_count=row_count, field="active_mask")
        ]
        valid_rows = response_mask.bool().any(dim=-1).detach().cpu().tolist()
        active_mask = [bool(active and valid) for active, valid in zip(active_values, valid_rows, strict=True)]
        marshal_config = _config_get(kwargs.get("config"), "marshal", {}) or {}
        result = compute_marshal_scalar_advantages(
            turn_rewards=torch.tensor(
                [
                    float(value)
                    for value in _batch_values(
                        non_tensors["marshal_turn_reward"], row_count=row_count, field="marshal_turn_reward"
                    )
                ],
                dtype=torch.float32,
                device=response_mask.device,
            ),
            episode_ids=_batch_values(
                non_tensors["marshal_episode_id"], row_count=row_count, field="marshal_episode_id"
            ),
            player_ids=_batch_values(non_tensors["marshal_player_id"], row_count=row_count, field="marshal_player_id"),
            player_turn_ids=[
                int(value)
                for value in _batch_values(
                    non_tensors["marshal_player_turn"], row_count=row_count, field="marshal_player_turn"
                )
            ],
            gamma=float(kwargs.get("gamma", 1.0)),
            reward_normalization=str(_config_get(marshal_config, "reward_normalization", "mean")),
            advantage_normalization=str(_config_get(marshal_config, "advantage_normalization", "mean")),
            whiten_rewards=bool(_config_get(marshal_config, "whiten_rewards", True)),
            whiten_advantages=bool(_config_get(marshal_config, "whiten_advantages", True)),
            active_mask=active_mask,
        )
        data.batch["advantages"] = result.advantages.unsqueeze(-1) * response_mask
        data.batch["returns"] = result.turn_returns.unsqueeze(-1) * response_mask
        data.batch["marshal_scalar_advantage"] = result.advantages
        data.batch["marshal_turn_return"] = result.turn_returns
        data.batch["marshal_normalized_reward"] = result.normalized_rewards
        data.non_tensor_batch["marshal_active_mask"] = np.asarray(active_mask, dtype=object)
        return data

    def compute_extra_metrics(self, data: Any, metrics: dict[str, Any], stage: str) -> dict[str, Any]:
        del metrics
        if stage != "advantage" or "marshal_scalar_advantage" not in data.batch:
            return {}
        import torch

        active = torch.tensor(
            [bool(value) for value in data.non_tensor_batch.get("marshal_active_mask", [])],
            dtype=torch.bool,
            device=data.batch["marshal_scalar_advantage"].device,
        )
        if active.numel() == 0 or not bool(active.any()):
            return {}
        advantages = data.batch["marshal_scalar_advantage"][active]
        returns = data.batch["marshal_turn_return"][active]
        players = {
            str(value)
            for value, is_active in zip(data.non_tensor_batch["marshal_player_id"], active.tolist(), strict=True)
            if is_active
        }
        return {
            "trajweave/marshal/active_players": float(len(players)),
            "trajweave/marshal/advantage_mean": float(advantages.mean()),
            "trajweave/marshal/advantage_std": float(advantages.std(unbiased=False)),
            "trajweave/marshal/nonzero_advantage_ratio": float((advantages.abs() > 1e-8).float().mean()),
            "trajweave/marshal/turn_return_mean": float(returns.mean()),
        }


def apply_marshal_turn_advantage_patch(config: Any = None) -> None:
    from trajweave.backends.verl.extensions.common.runtime import apply_hook_aware_tq_runtime

    apply_hook_aware_tq_runtime(config)


def _batch_values(values: Any, *, row_count: int, field: str) -> list[Any]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, list | tuple):
        values = [values]
    output = list(values)
    if len(output) != row_count:
        raise ValueError(f"MARSHAL field {field!r} has {len(output)} rows, expected {row_count}.")
    return output


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    try:
        return config.get(key, default)
    except (AttributeError, TypeError):
        return getattr(config, key, default)
