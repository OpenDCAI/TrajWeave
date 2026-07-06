from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PPOExtensionHooks:
    name: str = "default"

    def batch_schema_fields(self, stage: str) -> tuple[str, ...]:
        return ()

    def on_rollout_output(self, data: Any) -> Any:
        return data

    def process_rewards(self, data: Any) -> Any:
        return data

    def build_advantage_groups(self, data: Any) -> Any:
        return data.non_tensor_batch["uid"]

    def compute_extra_metrics(self, data: Any, metrics: dict[str, Any], stage: str) -> dict[str, Any]:
        return {}


@dataclass(frozen=True)
class AgentWiseGRPOHooks(PPOExtensionHooks):
    name: str = "agent_wise_grpo"

    def batch_schema_fields(self, stage: str) -> tuple[str, ...]:
        return ("agent_id", "traj_uid", "turn_id")

    def build_advantage_groups(self, data: Any) -> Any:
        import numpy as np

        if "agent_id" not in data.non_tensor_batch:
            raise KeyError("agent-wise GRPO requires non_tensor_batch['agent_id'].")
        return np.array(
            [
                f"{uid}_{agent_id}"
                for uid, agent_id in zip(data.non_tensor_batch["uid"], data.non_tensor_batch["agent_id"], strict=True)
            ],
            dtype=object,
        )
