from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from trajweave.backends.verl.emitters.registry import supported_emitter_recipes


@dataclass(frozen=True)
class TrajWeaveAgentLoopRuntimeConfig:
    recipe: str | None = None
    config_path: str | None = None
    coordination_protocol: str | None = None
    trajectory_schema: str | None = None
    credit_allocator: str | None = None
    agent_loop_backend: str = "verl_tq"
    run_id: str | None = None
    run_dir: str | None = None
    capture_online_turns: bool = False
    max_policy_lag: int = 1
    buffer_min_batch_size: int = 1
    dynamic_filter_reward_range: tuple[float, float] | None = None

    @classmethod
    def from_verl_config(cls, config: Any) -> "TrajWeaveAgentLoopRuntimeConfig":
        trajweave = config_get(config, "trajweave", default={})
        return cls(
            recipe=config_get(trajweave, "recipe"),
            config_path=config_get(trajweave, "config"),
            coordination_protocol=config_get(trajweave, "coordination_protocol"),
            trajectory_schema=config_get(trajweave, "trajectory_schema"),
            credit_allocator=config_get(trajweave, "credit_allocator"),
            agent_loop_backend=str(config_get(trajweave, "agent_loop_backend", "verl_tq")),
            run_id=config_get(trajweave, "run_id"),
            run_dir=config_get(trajweave, "run_dir"),
            capture_online_turns=as_bool(config_get(trajweave, "capture_online_turns", False)),
            max_policy_lag=int(config_get(trajweave, "max_policy_lag", 1)),
            buffer_min_batch_size=int(config_get(trajweave, "buffer_min_batch_size", 1)),
            dynamic_filter_reward_range=_reward_range(config_get(trajweave, "dynamic_filter_reward_range", None)),
        )

    def as_overrides(self) -> dict[str, str]:
        values = {
            "trajweave.recipe": self.recipe,
            "trajweave.config": self.config_path,
            "trajweave.coordination_protocol": self.coordination_protocol,
            "trajweave.trajectory_schema": self.trajectory_schema,
            "trajweave.credit_allocator": self.credit_allocator,
            "trajweave.agent_loop_backend": self.agent_loop_backend,
            "trajweave.run_id": self.run_id,
            "trajweave.run_dir": self.run_dir,
            "trajweave.capture_online_turns": str(self.capture_online_turns).lower(),
            "trajweave.max_policy_lag": str(self.max_policy_lag),
            "trajweave.buffer_min_batch_size": str(self.buffer_min_batch_size),
        }
        if self.dynamic_filter_reward_range is not None:
            values["trajweave.dynamic_filter_reward_range"] = str(list(self.dynamic_filter_reward_range))
        return {key: value for key, value in values.items() if value is not None}


def validate_agent_loop_backend(recipe: str | None, backend: str) -> None:
    supported_recipes = supported_emitter_recipes()
    if recipe and recipe not in supported_recipes:
        raise ValueError(f"Unsupported TrajWeave recipe for VERL AgentLoopManager: {recipe}")
    if backend not in {"verl_tq", "synthetic_tq", "hf_local_tq", "vllm_marti_tq"}:
        raise ValueError(f"Unsupported TrajWeave AgentLoop backend: {backend}")
    if backend == "vllm_marti_tq" and recipe != "marti_mars2_single_mcts":
        raise ValueError("vllm_marti_tq is currently implemented only for marti_mars2_single_mcts.")
    if recipe and backend == "verl_tq":
        raise ValueError(
            "TrajWeave MASRL recipes require agent_loop_backend in {'synthetic_tq', 'hf_local_tq'}; "
            "VERL native verl_tq does not emit agent_id/traj_uid/turn_id metadata required by agent-wise credit."
        )


def config_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    try:
        return config.get(key, default)
    except (AttributeError, TypeError):
        return getattr(config, key, default)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _reward_range(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = [part.strip() for part in value.split(",")]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("trajweave.dynamic_filter_reward_range must contain [lower, upper]")
    result = (float(value[0]), float(value[1]))
    if result[0] >= result[1]:
        raise ValueError("dynamic_filter_reward_range lower must be smaller than upper")
    return result
