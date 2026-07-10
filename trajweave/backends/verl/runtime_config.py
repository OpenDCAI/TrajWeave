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
    hf_local_dtype: str = "fp32"
    hf_local_model_cache_size: int = 0
    run_id: str | None = None
    run_dir: str | None = None
    capture_online_turns: bool = False
    turn_padding_multiple: int = 1

    @classmethod
    def from_verl_config(cls, config: Any) -> TrajWeaveAgentLoopRuntimeConfig:
        trajweave = config_get(config, "trajweave", default={})
        recipe = config_get(trajweave, "recipe")
        hf_local_dtype = normalize_hf_local_dtype(config_get(trajweave, "hf_local_dtype", "fp32"))
        hf_local_model_cache_size = int(config_get(trajweave, "hf_local_model_cache_size", 0))
        if hf_local_model_cache_size < 0:
            raise ValueError("trajweave.hf_local_model_cache_size must be zero or a positive integer.")
        return cls(
            recipe=recipe,
            config_path=config_get(trajweave, "config"),
            coordination_protocol=config_get(trajweave, "coordination_protocol"),
            trajectory_schema=config_get(trajweave, "trajectory_schema"),
            credit_allocator=config_get(trajweave, "credit_allocator"),
            agent_loop_backend=str(config_get(trajweave, "agent_loop_backend", "verl_tq")),
            hf_local_dtype=hf_local_dtype,
            hf_local_model_cache_size=hf_local_model_cache_size,
            run_id=config_get(trajweave, "run_id"),
            run_dir=config_get(trajweave, "run_dir"),
            capture_online_turns=as_bool(config_get(trajweave, "capture_online_turns", False)),
            turn_padding_multiple=max(
                1,
                int(
                    config_get(
                        trajweave,
                        "turn_padding_multiple",
                        _default_turn_padding_multiple(recipe),
                    )
                ),
            ),
        )

    def as_overrides(self) -> dict[str, str]:
        values = {
            "trajweave.recipe": self.recipe,
            "trajweave.config": self.config_path,
            "trajweave.coordination_protocol": self.coordination_protocol,
            "trajweave.trajectory_schema": self.trajectory_schema,
            "trajweave.credit_allocator": self.credit_allocator,
            "trajweave.agent_loop_backend": self.agent_loop_backend,
            "trajweave.hf_local_dtype": self.hf_local_dtype,
            "trajweave.hf_local_model_cache_size": str(self.hf_local_model_cache_size),
            "trajweave.run_id": self.run_id,
            "trajweave.run_dir": self.run_dir,
            "trajweave.capture_online_turns": str(self.capture_online_turns).lower(),
            "trajweave.turn_padding_multiple": str(self.turn_padding_multiple),
        }
        return {key: value for key, value in values.items() if value is not None}


def validate_agent_loop_backend(recipe: str | None, backend: str) -> None:
    from trajweave.backends.verl.emitters.registry import supported_emitter_recipes

    supported_recipes = supported_emitter_recipes()
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


def normalize_hf_local_dtype(value: Any) -> str:
    aliases = {
        "fp32": "fp32",
        "float32": "fp32",
        "fp16": "fp16",
        "float16": "fp16",
        "bf16": "bf16",
        "bfloat16": "bf16",
    }
    normalized = aliases.get(str(value).strip().lower())
    if normalized is None:
        raise ValueError(f"trajweave.hf_local_dtype must be one of fp32, fp16, or bf16; got {value!r}.")
    return normalized


def _default_turn_padding_multiple(recipe: str | None) -> int:
    return {
        "doctor_mas_math": 2,
        "doctor_mas_search": 4,
        "maporl_debate_math": 2,
        "agentflow_planner_tool": 4,
    }.get(str(recipe), 1)
