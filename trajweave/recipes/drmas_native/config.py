from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

TRAJWEAVE_AGENT_LOOP_MANAGER_FQN = "trajweave.backends.verl.agent_loop.TrajWeaveAgentLoopManager"


@dataclass(frozen=True)
class DrMASNativeRecipeSpec:
    name: str
    task: str
    runtime_recipe: str
    agent_ids: tuple[str, ...]
    orchestra_type: str
    coordination_protocol: str
    default_max_loop_num: int | None = None


DRMAS_NATIVE_MATH = DrMASNativeRecipeSpec(
    name="drmas_native_math",
    task="math",
    runtime_recipe="doctor_mas_math",
    agent_ids=("Solver Agent", "Verifier Agent"),
    orchestra_type="math",
    coordination_protocol="solver_verifier_loop",
    default_max_loop_num=3,
)

DRMAS_NATIVE_SEARCH = DrMASNativeRecipeSpec(
    name="drmas_native_search",
    task="search",
    runtime_recipe="doctor_mas_search",
    agent_ids=("Verifier Agent", "Search Agent", "Answer Agent"),
    orchestra_type="search",
    coordination_protocol="search_answer",
)


def recipe_spec(name: str) -> DrMASNativeRecipeSpec:
    aliases = {
        "drmas.math.verl_tiny": DRMAS_NATIVE_MATH,
        "drmas_native_math": DRMAS_NATIVE_MATH,
        "doctor_mas_native_math": DRMAS_NATIVE_MATH,
        "drmas.search.verl_tiny": DRMAS_NATIVE_SEARCH,
        "drmas_native_search": DRMAS_NATIVE_SEARCH,
        "doctor_mas_native_search": DRMAS_NATIVE_SEARCH,
    }
    try:
        return aliases[name]
    except KeyError as exc:
        raise ValueError(f"Unknown DrMAS-native recipe: {name}") from exc


def build_drmas_native_launch_overrides(
    config: dict[str, Any],
    *,
    config_path: str | None,
) -> tuple[str, ...]:
    spec = recipe_spec(str(config.get("recipe", config.get("run", {}).get("recipe", "drmas_native_math"))))
    native_cfg = config.get("drmas_native", {})
    verl_cfg = config.get("verl", {})
    agent_ids = tuple(native_cfg.get("agent_ids", spec.agent_ids))
    model_ids = tuple(native_cfg.get("model_ids", native_cfg.get("models", ("default",) * len(agent_ids))))
    if len(model_ids) != len(agent_ids):
        raise ValueError("drmas_native.model_ids must have the same length as drmas_native.agent_ids.")

    model_sharing = str(bool(native_cfg.get("model_sharing", len(set(model_ids)) == 1))).lower()
    agent_loop_backend = str(native_cfg.get("agent_loop_backend", native_cfg.get("rollout_backend", "hf_local_tq")))
    if agent_loop_backend == "verl_tq":
        raise ValueError("DrMAS native integration requires agent_loop_backend to be synthetic_tq or hf_local_tq, not verl_tq.")
    max_loop_num = native_cfg.get("max_loop_num", spec.default_max_loop_num)
    source_config = config_path or str(Path.cwd())

    required = [
        "algorithm.adv_estimator=grpo",
        "++algorithm.group_by_agent_id=true",
        "algorithm.norm_adv_by_std_in_grpo=true",
        f"+agent.agent_ids={_hydra_list(agent_ids)}",
        f"+agent.model_ids={_hydra_list(model_ids)}",
        f"+agent.model_sharing={model_sharing}",
        f"+agent.orchestra_type={spec.orchestra_type}",
        f"+trajweave.recipe={spec.runtime_recipe}",
        f"+trajweave.config={source_config}",
        f"+trajweave.coordination_protocol={spec.coordination_protocol}",
        "+trajweave.trajectory_schema=multi_agent_turn_v1",
        "+trajweave.credit_allocator=drmas_agent_wise_grpo",
        "+trajweave.verl_extensions=[drmas_agent_wise_grpo]",
        f"+trajweave.agent_loop_backend={agent_loop_backend}",
        f"+actor_rollout_ref.rollout.agent.agent_loop_manager_class={TRAJWEAVE_AGENT_LOOP_MANAGER_FQN}",
    ]
    if max_loop_num is not None and spec.task == "math":
        required.append(f"+agent.orchestra.math.max_loop_num={int(max_loop_num)}")
    if spec.task == "search":
        search_url = native_cfg.get("search_url")
        if search_url:
            required.append(f"+env.search.search_url={search_url}")

    return tuple(str(item) for item in verl_cfg.get("overrides", [])) + tuple(required)


def _hydra_list(values: tuple[str, ...]) -> str:
    return "[" + ",".join(_quote(value) for value in values) + "]"


def _quote(value: str) -> str:
    escaped = str(value).replace('"', '\\"')
    return f'"{escaped}"'
