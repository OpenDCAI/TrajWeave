from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from trajweave.backends.verl.extensions.common.hooks import AgentWiseGRPOHooks


@dataclass(frozen=True)
class WideSeekR1GRPOHooks(AgentWiseGRPOHooks):
    """Trajectory GRPO plus WideSeek-R1 agent/token loss reweighting."""

    name: str = "wideseek_r1_multi_agent_grpo"

    def batch_schema_fields(self, stage: str, config: Any = None) -> tuple[str, ...]:
        if stage != "advantage":
            return ()
        return (
            "agent_id",
            "traj_uid",
            "turn_id",
            "wideseek_turn_role",
            "wideseek_agent_instance",
            "wideseek_subtrajectory_id",
            "wideseek_agent_count",
            "wideseek_parallel_wave",
            "wideseek_format_valid",
        )

    def tq_select_fields(
        self,
        stage: str,
        default_fields: tuple[str, ...] | None = None,
        config: Any = None,
    ) -> tuple[str, ...]:
        fields = list(super().tq_select_fields(stage, default_fields=default_fields, config=config))
        if stage == "advantage":
            fields.extend(self.batch_schema_fields(stage, config=config))
        return tuple(dict.fromkeys(fields))

    def compute_advantage(self, data: Any, **kwargs: Any) -> Any:
        # group_by_agent_id is intentionally disabled: WideSeek-R1 assigns one
        # trajectory advantage to the lead and every subagent.
        config = kwargs.get("config")
        algorithm = config
        if isinstance(config, dict):
            algorithm = {**config, "group_by_agent_id": False}
        kwargs["config"] = algorithm
        data = super().compute_advantage(data, **kwargs)
        self._apply_dual_level_reweighting(data)
        return data

    @staticmethod
    def _apply_dual_level_reweighting(data: Any) -> None:
        import numpy as np
        import torch

        required = ("traj_uid", "wideseek_agent_instance", "wideseek_agent_count")
        missing = [field for field in required if field not in data.non_tensor_batch]
        if missing:
            raise KeyError(f"WideSeek-R1 dual-level reweighting requires fields: {missing}.")
        response_mask = data.batch["response_mask"]
        advantages = data.batch["advantages"]
        row_count = response_mask.shape[0]
        trajectories = list(data.non_tensor_batch["traj_uid"])
        agents = list(data.non_tensor_batch["wideseek_agent_instance"])
        configured_counts = list(data.non_tensor_batch["wideseek_agent_count"])
        valid_rows = response_mask.bool().any(dim=-1).detach().cpu().tolist()
        real_rows = [row for row, valid in enumerate(valid_rows) if valid]
        trajectory_ids = tuple(dict.fromkeys(str(trajectories[row]) for row in real_rows))
        if not trajectory_ids:
            raise ValueError("WideSeek-R1 advantage batch has no trainable response rows.")

        agent_tokens: dict[tuple[str, str], float] = defaultdict(float)
        agents_by_trajectory: dict[str, set[str]] = defaultdict(set)
        total_tokens = 0.0
        for row in real_rows:
            trajectory_id = str(trajectories[row])
            agent_id = str(agents[row])
            tokens = float(response_mask[row].sum().item())
            total_tokens += tokens
            agent_tokens[(trajectory_id, agent_id)] += tokens
            agents_by_trajectory[trajectory_id].add(agent_id)

        scales = torch.zeros(row_count, dtype=advantages.dtype, device=advantages.device)
        trajectory_count = len(trajectory_ids)
        for row in real_rows:
            trajectory_id = str(trajectories[row])
            agent_id = str(agents[row])
            observed_count = len(agents_by_trajectory[trajectory_id])
            configured_count = int(configured_counts[row])
            if observed_count != configured_count:
                raise ValueError(
                    f"WideSeek-R1 trajectory {trajectory_id!r} declares {configured_count} agents "
                    f"but emitted {observed_count}."
                )
            denominator = trajectory_count * observed_count * agent_tokens[(trajectory_id, agent_id)]
            scales[row] = total_tokens / denominator
        data.batch["advantages"] = advantages * scales.unsqueeze(-1)
        data.batch["returns"] = data.batch["returns"] * scales.unsqueeze(-1)
        data.non_tensor_batch["wideseek_loss_scale"] = np.asarray(scales.detach().cpu(), dtype=np.float32)
        data.meta_info["wideseek_dual_level_reweighting"] = True

    def compute_extra_metrics(self, data: Any, metrics: dict[str, Any], stage: str) -> dict[str, Any]:
        del metrics
        if stage != "advantage" or "wideseek_loss_scale" not in data.non_tensor_batch:
            return {}
        values = [float(value) for value in data.non_tensor_batch["wideseek_loss_scale"] if float(value) > 0]
        agents = data.non_tensor_batch.get("wideseek_agent_instance", [])
        return {
            "trajweave/wideseek_r1/active_agent_instances": float(len(set(map(str, agents)))),
            "trajweave/wideseek_r1/loss_scale_mean": sum(values) / len(values) if values else 0.0,
        }


def apply_wideseek_r1_grpo_patch(config: Any = None) -> None:
    from trajweave.backends.verl.extensions.common.runtime import apply_hook_aware_tq_runtime

    apply_hook_aware_tq_runtime(config)
