from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Any

from trajweave.pipeline.config import hydra_path

TRAJWEAVE_AGENT_LOOP_MANAGER_FQN = "trajweave.backends.verl.agent_loop.TrajWeaveAgentLoopManager"
WIDESEEK_HOOKS_FQN = "trajweave.backends.verl.extensions.wideseek_r1.WideSeekR1GRPOHooks"


def resolve_wideseek_r1_settings(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("wideseek_r1", {}) or {}
    if not isinstance(raw, dict):
        raise ValueError("WideSeek-R1 settings must be a mapping.")
    backend = str(raw.get("agent_loop_backend", "hf_local_tq")).strip()
    if backend not in {"synthetic_tq", "hf_local_tq"}:
        raise ValueError("WideSeek-R1 agent_loop_backend must be synthetic_tq or hf_local_tq.")
    max_subagents = _positive_int(raw.get("max_parallel_subagents", 3), "max_parallel_subagents")
    if max_subagents > 32:
        raise ValueError("WideSeek-R1 max_parallel_subagents must not exceed 32.")
    lead_agent = _nonempty(raw.get("lead_agent", "lead_agent"), "lead_agent")
    subagent_prefix = _nonempty(raw.get("subagent_prefix", "subagent_"), "subagent_prefix")
    shared_model_id = _nonempty(raw.get("shared_model_id", "shared_policy"), "shared_model_id")
    return {
        "agent_loop_backend": backend,
        "max_parallel_subagents": max_subagents,
        "lead_agent": lead_agent,
        "subagent_prefix": subagent_prefix,
        "shared_model_id": shared_model_id,
        "format_reward": _nonnegative(raw.get("format_reward", 0.1), "format_reward"),
        "search_reward": _nonnegative(raw.get("search_reward", 0.05), "search_reward"),
        "length_limit": _positive_int(raw.get("length_limit", 3000), "length_limit"),
        "max_length_limit": _positive_int(raw.get("max_length_limit", 5000), "max_length_limit"),
        "length_penalty": _nonnegative(raw.get("length_penalty", 0.1), "length_penalty"),
    }


def build_wideseek_r1_launch_overrides(config: dict[str, Any], *, config_path: str | None) -> tuple[str, ...]:
    settings = resolve_wideseek_r1_settings(config)
    if settings["max_length_limit"] <= settings["length_limit"]:
        raise ValueError("WideSeek-R1 max_length_limit must be greater than length_limit.")
    verl = config.get("verl", {}) or {}
    if not isinstance(verl, dict):
        raise ValueError("WideSeek-R1 verl settings must be a mapping.")
    lead = settings["lead_agent"]
    workers = [f"{settings['subagent_prefix']}{index}" for index in range(settings["max_parallel_subagents"])]
    agent_ids = [lead, *workers]
    model_ids = [settings["shared_model_id"]] * len(agent_ids)
    source_config = config_path or str(Path.cwd())
    required = (
        "trainer.use_v1=true",
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=false",
        "critic.enable=false",
        "actor_rollout_ref.actor.policy_loss.loss_mode=vanilla",
        "actor_rollout_ref.actor.loss_agg_mode=token-mean",
        "actor_rollout_ref.actor.ppo_epochs=1",
        "actor_rollout_ref.actor.use_kl_loss=false",
        "++algorithm.group_by_agent_id=false",
        f"++algorithm.extension_hooks_class={WIDESEEK_HOOKS_FQN}",
        f"+agent.agent_ids={_hydra_list(agent_ids)}",
        f"+agent.model_ids={_hydra_list(model_ids)}",
        "+agent.model_sharing=true",
        "+agent.orchestra_type=wideseek_r1",
        f"+agent.orchestra.wideseek_r1.lead_agent={_quote(lead)}",
        f"+agent.orchestra.wideseek_r1.subagent_prefix={_quote(settings['subagent_prefix'])}",
        f"+agent.orchestra.wideseek_r1.max_parallel_subagents={settings['max_parallel_subagents']}",
        f"+agent.orchestra.wideseek_r1.shared_model_id={_quote(settings['shared_model_id'])}",
        f"+agent.orchestra.wideseek_r1.format_reward={settings['format_reward']}",
        f"+agent.orchestra.wideseek_r1.search_reward={settings['search_reward']}",
        f"+agent.orchestra.wideseek_r1.length_limit={settings['length_limit']}",
        f"+agent.orchestra.wideseek_r1.max_length_limit={settings['max_length_limit']}",
        f"+agent.orchestra.wideseek_r1.length_penalty={settings['length_penalty']}",
        "+trajweave.recipe=wideseek_r1_broad_search",
        f"+trajweave.config={hydra_path(source_config)}",
        "+trajweave.coordination_protocol=wideseek_r1_width_search",
        "+trajweave.trajectory_schema=wideseek_r1_isolated_subtrajectory_v1",
        "+trajweave.credit_allocator=wideseek_r1_multi_agent_grpo",
        "+trajweave.verl_extensions=[trajweave_wideseek_r1_grpo]",
        f"+trajweave.agent_loop_backend={settings['agent_loop_backend']}",
        "+trajweave.turn_padding_multiple=4",
        f"+actor_rollout_ref.rollout.agent.agent_loop_manager_class={TRAJWEAVE_AGENT_LOOP_MANAGER_FQN}",
    )
    configured = tuple(str(item) for item in verl.get("overrides", []))
    required_keys = {_override_key(item) for item in required}
    retained = tuple(item for item in configured if _override_key(item) not in required_keys)
    return retained + required


def _hydra_list(values: list[str]) -> str:
    return "[" + ",".join(_quote(value) for value in values) + "]"


def _quote(value: Any) -> str:
    return '"' + str(value).replace('"', '\\"') + '"'


def _override_key(value: str) -> str:
    return value.split("=", 1)[0].lstrip("+")


def _nonempty(value: Any, field: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"WideSeek-R1 {field} must be non-empty.")
    return result


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"WideSeek-R1 {field} must be a positive integer.")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"WideSeek-R1 {field} must be a positive integer.") from exc
    if result < 1 or float(value) != result:
        raise ValueError(f"WideSeek-R1 {field} must be a positive integer.")
    return result


def _nonnegative(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"WideSeek-R1 {field} must be finite and non-negative.") from exc
    if not isfinite(result) or result < 0:
        raise ValueError(f"WideSeek-R1 {field} must be finite and non-negative.")
    return result
