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
    allow_plain_text_prompt_fallback: bool = False
    run_id: str | None = None
    run_dir: str | None = None
    capture_online_turns: bool = False
    turn_padding_multiple: int = 1
    max_policy_lag: int = 1
    buffer_min_batch_size: int = 1
    dynamic_filter_reward_range: tuple[float, float] | None = None

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
            allow_plain_text_prompt_fallback=as_bool(config_get(trajweave, "allow_plain_text_prompt_fallback", False)),
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
            "trajweave.hf_local_dtype": self.hf_local_dtype,
            "trajweave.hf_local_model_cache_size": str(self.hf_local_model_cache_size),
            "trajweave.allow_plain_text_prompt_fallback": str(self.allow_plain_text_prompt_fallback).lower(),
            "trajweave.run_id": self.run_id,
            "trajweave.run_dir": self.run_dir,
            "trajweave.capture_online_turns": str(self.capture_online_turns).lower(),
            "trajweave.turn_padding_multiple": str(self.turn_padding_multiple),
            "trajweave.max_policy_lag": str(self.max_policy_lag),
            "trajweave.buffer_min_batch_size": str(self.buffer_min_batch_size),
        }
        if self.dynamic_filter_reward_range is not None:
            values["trajweave.dynamic_filter_reward_range"] = str(list(self.dynamic_filter_reward_range))
        return {key: value for key, value in values.items() if value is not None}


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
        "gigpo_solver_verifier_math": 2,
    }.get(str(recipe), 1)


def _reward_range(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = [part.strip() for part in value.split(",")]
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError("trajweave.dynamic_filter_reward_range must contain [lower, upper]")
    result = (float(value[0]), float(value[1]))
    if result[0] >= result[1]:
        raise ValueError("dynamic_filter_reward_range lower must be smaller than upper")
    return result
