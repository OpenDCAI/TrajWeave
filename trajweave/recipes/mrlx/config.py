from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Any

TRAJWEAVE_AGENT_LOOP_MANAGER_FQN = "trajweave.backends.verl.agent_loop.TrajWeaveAgentLoopManager"
MRLX_HOOKS_FQN = "trajweave.backends.verl.extensions.mrlx.MrlXMGRPOHooks"
MRLX_TRAINER_MODE = "trajweave_mrlx_async"
_ALLOWED_TOOLS = {"search", "browse", "web_search", "search_and_browse"}


def resolve_mrlx_settings(config: dict[str, Any], *, require_worker_assets: bool = False) -> dict[str, Any]:
    raw = config.get("mrlx", {}) or {}
    if not isinstance(raw, dict):
        raise ValueError("MrlX settings must be a mapping.")
    explorer_agent = _nonempty(raw, "explorer_agent", "main_explorer")
    adapter_agent = _nonempty(raw, "adapter_agent", "sub_adapter")
    if explorer_agent == adapter_agent:
        raise ValueError("MrlX explorer_agent and adapter_agent must be different.")
    explorer_model_id = _nonempty(raw, "explorer_model_id", "explorer_policy")
    adapter_model_id = _nonempty(raw, "adapter_model_id", "adapter_policy")
    if explorer_model_id == adapter_model_id:
        raise ValueError("MrlX requires distinct explorer_model_id and adapter_model_id values.")
    tool_name = _nonempty(raw, "tool_name", "search_and_browse")
    if tool_name not in _ALLOWED_TOOLS:
        raise ValueError(f"MrlX tool_name must be one of: {', '.join(sorted(_ALLOWED_TOOLS))}.")
    research_rounds = _positive_int(raw.get("research_rounds", 1), field="research_rounds")
    if research_rounds != 1:
        raise ValueError(
            "The current MrlX training bridge requires research_rounds=1 so each Adapter invocation "
            "maps to one delayed training unit."
        )
    delay_steps = _positive_int(raw.get("adapter_delay_steps", 1), field="adapter_delay_steps")
    if delay_steps != 1:
        raise ValueError("The MrlX integration currently requires adapter_delay_steps=1.")
    explorer_format_bonus = _unit_float(
        raw.get("explorer_format_bonus", 0.1), field="explorer_format_bonus"
    )
    adapter_format_bonus = _unit_float(raw.get("adapter_format_bonus", 0.1), field="adapter_format_bonus")
    backend = str(raw.get("agent_loop_backend", "hf_local_tq")).strip().lower()
    if backend not in {"synthetic_tq", "hf_local_tq"}:
        raise ValueError("MrlX agent_loop_backend must be synthetic_tq or hf_local_tq.")
    if str(config.get("mode", "")) == "verl_train" and backend != "hf_local_tq":
        raise ValueError("MrlX verl_train requires agent_loop_backend=hf_local_tq for updated rollout weights.")
    tokenizer_mode = str(raw.get("tokenizer_mode", "shared")).strip().lower()
    if tokenizer_mode != "shared":
        raise ValueError(
            "MrlX currently requires tokenizer_mode=shared so rollout and training use identical prompt IDs."
        )

    raw_groups = raw.get("worker_groups", {}) or {}
    if not isinstance(raw_groups, dict):
        raise ValueError("MrlX worker_groups must be a mapping keyed by model id.")
    worker_groups: dict[str, dict[str, Any]] = {}
    for model_id in (explorer_model_id, adapter_model_id):
        group = raw_groups.get(model_id, {}) or {}
        if not isinstance(group, dict):
            raise ValueError(f"MrlX worker_groups.{model_id} must be a mapping.")
        worker_groups[model_id] = {**group, "trainable": True}
    unexpected = sorted(set(raw_groups) - set(worker_groups))
    if unexpected:
        raise ValueError(f"MrlX worker_groups contains unused groups: {unexpected}.")

    if require_worker_assets:
        missing = [
            f"{model_id}.{field}"
            for model_id, group in worker_groups.items()
            for field in ("model_path", "tokenizer_path")
            if not group.get(field)
        ]
        if missing:
            raise ValueError(f"MrlX training requires explicit worker group assets; missing: {missing}.")
        tokenizer_paths = {str(group["tokenizer_path"]) for group in worker_groups.values()}
        if tokenizer_mode == "shared" and len(tokenizer_paths) != 1:
            raise ValueError("MrlX shared tokenizer_mode requires identical tokenizer_path values.")

    return {
        "explorer_agent": explorer_agent,
        "adapter_agent": adapter_agent,
        "explorer_model_id": explorer_model_id,
        "adapter_model_id": adapter_model_id,
        "tool_name": tool_name,
        "research_rounds": research_rounds,
        "adapter_delay_steps": delay_steps,
        "explorer_format_bonus": explorer_format_bonus,
        "adapter_format_bonus": adapter_format_bonus,
        "agent_loop_backend": backend,
        "tokenizer_mode": tokenizer_mode,
        "worker_groups": worker_groups,
    }


def build_mrlx_launch_overrides(config: dict[str, Any], *, config_path: str | None) -> tuple[str, ...]:
    settings = resolve_mrlx_settings(config, require_worker_assets=True)
    verl = config.get("verl", {}) or {}
    if not isinstance(verl, dict):
        raise ValueError("MrlX verl settings must be a mapping.")
    explorer_agent = settings["explorer_agent"]
    adapter_agent = settings["adapter_agent"]
    explorer_group = settings["explorer_model_id"]
    adapter_group = settings["adapter_model_id"]
    worker_groups = settings["worker_groups"]
    source_config = config_path or str(Path.cwd())
    required = (
        "trainer.use_v1=true",
        f"trainer.v1.trainer_mode={MRLX_TRAINER_MODE}",
        "trainer.critic_warmup=0",
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=false",
        "critic.enable=false",
        "actor_rollout_ref.actor.policy_loss.loss_mode=vanilla",
        "actor_rollout_ref.actor.loss_agg_mode=token-mean",
        "actor_rollout_ref.actor.clip_ratio=0.2",
        "actor_rollout_ref.actor.clip_ratio_low=0.2",
        "actor_rollout_ref.actor.clip_ratio_high=0.28",
        "actor_rollout_ref.actor.ppo_epochs=1",
        "actor_rollout_ref.actor.use_kl_loss=false",
        "++algorithm.group_by_agent_id=true",
        f"++algorithm.extension_hooks_class={MRLX_HOOKS_FQN}",
        f"+agent.agent_ids={_hydra_list((explorer_agent, adapter_agent))}",
        f"+agent.model_ids={_hydra_list((explorer_group, adapter_group))}",
        "+agent.model_sharing=false",
        f"+agent.worker_group_ids={_hydra_list((explorer_group, adapter_group))}",
        f"+agent.worker_groups={_hydra_worker_groups(worker_groups)}",
        "+agent.orchestra_type=mrlx",
        f"+agent.orchestra.mrlx.explorer_agent={_quote(explorer_agent)}",
        f"+agent.orchestra.mrlx.adapter_agent={_quote(adapter_agent)}",
        f"+agent.orchestra.mrlx.tool_name={_quote(settings['tool_name'])}",
        f"+agent.orchestra.mrlx.research_rounds={settings['research_rounds']}",
        f"+agent.orchestra.mrlx.explorer_format_bonus={settings['explorer_format_bonus']}",
        f"+agent.orchestra.mrlx.adapter_format_bonus={settings['adapter_format_bonus']}",
        "+trajweave.recipe=mrlx_research_qa",
        f"+trajweave.config={source_config}",
        "+trajweave.coordination_protocol=mrlx_async_research",
        "+trajweave.trajectory_schema=mrlx_role_turn_v1",
        "+trajweave.credit_allocator=mrlx_mgrpo",
        "+trajweave.verl_extensions=[trajweave_mrlx_mgrpo]",
        f"+trajweave.agent_loop_backend={settings['agent_loop_backend']}",
        "+trajweave.turn_padding_multiple=2",
        "+trajweave.multi_actor.enabled=true",
        "+trajweave.multi_actor.routing_field=worker_group",
        f"+trajweave.multi_actor.tokenizer_mode={settings['tokenizer_mode']}",
        "+trajweave.multi_actor.metric_namespace=mrlx",
        f"+trajweave.mrlx.explorer_group={_quote(explorer_group)}",
        f"+trajweave.mrlx.adapter_group={_quote(adapter_group)}",
        f"+trajweave.mrlx.adapter_delay_steps={settings['adapter_delay_steps']}",
        "+trajweave.mrlx.drain_adapter_replay=true",
        f"+actor_rollout_ref.rollout.agent.agent_loop_manager_class={TRAJWEAVE_AGENT_LOOP_MANAGER_FQN}",
    )
    configured = tuple(str(item) for item in verl.get("overrides", []))
    required_keys = {_override_key(item) for item in required}
    retained = tuple(item for item in configured if _override_key(item) not in required_keys)
    return retained + required


def _hydra_worker_groups(groups: dict[str, dict[str, Any]]) -> str:
    entries = []
    for group_id, group in groups.items():
        fields = [f"id:{_quote(group_id)}", "trainable:true"]
        for field in ("model_path", "tokenizer_path", "gpus"):
            if field not in group:
                continue
            value = group[field]
            fields.append(f"{field}:{value if isinstance(value, int | float) else _quote(value)}")
        entries.append("{" + ",".join(fields) + "}")
    return "[" + ",".join(entries) + "]"


def _hydra_list(values: tuple[str, ...]) -> str:
    return "[" + ",".join(_quote(value) for value in values) + "]"


def _quote(value: Any) -> str:
    return '"' + str(value).replace('"', '\\"') + '"'


def _override_key(value: str) -> str:
    return value.split("=", 1)[0].lstrip("+")


def _nonempty(config: dict[str, Any], field: str, default: str) -> str:
    value = str(config.get(field, default)).strip()
    if not value:
        raise ValueError(f"MrlX {field} must be non-empty.")
    return value


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"MrlX {field} must be a positive integer.")
    try:
        parsed = int(value)
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"MrlX {field} must be a positive integer.") from exc
    if parsed < 1 or not isfinite(numeric) or numeric != parsed:
        raise ValueError(f"MrlX {field} must be a positive integer.")
    return parsed


def _unit_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"MrlX {field} must be a finite number between 0 and 1.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"MrlX {field} must be a finite number between 0 and 1.") from exc
    if not isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"MrlX {field} must be a finite number between 0 and 1.")
    return parsed
