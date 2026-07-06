from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
        }
        return {key: value for key, value in values.items() if value is not None}


def validate_agent_loop_backend(recipe: str | None, backend: str) -> None:
    supported_recipes = {"doctor_mas_math", "doctor_mas_search", "maporl_debate_math", "agentflow_planner_tool"}
    if recipe and recipe not in supported_recipes:
        raise ValueError(f"Unsupported TrajWeave recipe for VERL AgentLoopManager: {recipe}")
    if backend not in {"verl_tq", "synthetic_tq", "hf_local_tq"}:
        raise ValueError(f"Unsupported TrajWeave AgentLoop backend: {backend}")
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
