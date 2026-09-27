from __future__ import annotations

from math import isfinite, lcm
from pathlib import Path
from typing import Any

from trajweave.backends.verl.multi_actor.critic_config import normalize_critic_route_specs

TRAJWEAVE_AGENT_LOOP_MANAGER_FQN = "trajweave.backends.verl.agent_loop.TrajWeaveAgentLoopManager"
C3_HOOKS_FQN = "trajweave.backends.verl.extensions.c3.C3ContextualCounterfactualHooks"


def resolve_c3_settings(config: dict[str, Any], *, mode: str | None = None) -> dict[str, Any]:
    c3 = config.get("c3", {}) or {}
    fanout_raw = c3.get("fanout", [2, 2])
    if not isinstance(fanout_raw, list | tuple) or len(fanout_raw) != 2:
        raise ValueError("c3.fanout must contain [reasoner_fanout, actor_fanout].")
    fanout = tuple(
        _integer_at_least(value, field=f"c3.fanout[{index}]", minimum=2) for index, value in enumerate(fanout_raw)
    )
    variant = str(c3.get("credit_variant", "value_assisted")).strip().lower()
    if variant not in {"reward_only", "value_only", "value_assisted"}:
        raise ValueError("c3.credit_variant must be reward_only, value_only, or value_assisted.")
    baseline_mode = str(c3.get("baseline_mode", "loo")).strip().lower()
    if baseline_mode not in {"loo", "full_mean"}:
        raise ValueError("c3.baseline_mode must be loo or full_mean.")
    alpha = float(c3.get("value_assisted_alpha", c3.get("va_alpha", 1.0)))
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("c3.value_assisted_alpha must be in [0, 1].")
    role_order = tuple(str(value).strip() for value in c3.get("role_order", ["reasoner", "actor"]))
    if role_order != ("reasoner", "actor"):
        raise ValueError("The paper-facing C3 integration currently requires role_order=[reasoner, actor].")
    agent_loop_backend = str(c3.get("agent_loop_backend", "hf_local_tq")).strip().lower()
    if agent_loop_backend not in {"synthetic_tq", "hf_local_tq"}:
        raise ValueError("c3.agent_loop_backend must be synthetic_tq or hf_local_tq.")
    resolved_mode = _resolve_mode(config, explicit=mode)
    if resolved_mode == "verl_train" and agent_loop_backend != "hf_local_tq":
        raise ValueError(
            "C3 mode='verl_train' requires c3.agent_loop_backend=hf_local_tq so updated Actor weights "
            "are reloaded for the next rollout; synthetic_tq is diagnostic-only."
        )
    cache_size = _integer_at_least(
        c3.get("hf_local_model_cache_size", 2),
        field="c3.hf_local_model_cache_size",
        minimum=0,
    )
    settings = {
        "agent_loop_backend": agent_loop_backend,
        "hf_local_model_cache_size": cache_size,
        "fanout": fanout,
        "role_order": role_order,
        "credit_variant": variant,
        "baseline_mode": baseline_mode,
        "value_assisted_alpha": alpha,
        "normalize_advantages": bool(c3.get("normalize_advantages", True)),
        "critic": dict(c3.get("critic", {}) or {}),
    }
    _validate_critic_settings(settings)
    return settings


def build_c3_launch_overrides(
    config: dict[str, Any],
    *,
    config_path: str | None,
    mode: str | None = None,
) -> tuple[str, ...]:
    settings = resolve_c3_settings(config, mode=mode)
    c3 = config.get("c3", {}) or {}
    source_config = config_path or str(Path.cwd())
    actor_groups = tuple(str(value).strip() for value in c3.get("model_ids", ["reasoner", "actor"]))
    if len(actor_groups) != 2 or len(set(actor_groups)) != 2 or any(not value for value in actor_groups):
        raise ValueError("c3.model_ids must contain two distinct non-empty trainable worker groups.")
    agent_ids = tuple(str(value) for value in c3.get("agent_ids", ["reasoner", "actor"]))
    if agent_ids != ("reasoner", "actor"):
        raise ValueError("c3.agent_ids must be [reasoner, actor].")
    worker_groups = _normalize_worker_groups(c3, actor_groups=actor_groups)
    padding_multiple = lcm(
        2, *(_worker_group_gpus(group, group_id=group_id) for group_id, group in worker_groups.items())
    )
    required = [
        "algorithm.adv_estimator=grpo",
        "++algorithm.group_by_agent_id=false",
        f"algorithm.norm_adv_by_std_in_grpo={str(settings['normalize_advantages']).lower()}",
        f"++algorithm.c3.credit_variant={settings['credit_variant']}",
        f"++algorithm.c3.baseline_mode={settings['baseline_mode']}",
        f"++algorithm.c3.value_assisted_alpha={settings['value_assisted_alpha']}",
        f"++algorithm.c3.normalize_advantages={str(settings['normalize_advantages']).lower()}",
        "++actor_rollout_ref.actor.policy_loss.loss_mode=vanilla_no_dual_clip",
        "actor_rollout_ref.rollout.n=1",
        "actor_rollout_ref.rollout.val_kwargs.n=1",
        f"++algorithm.extension_hooks_class={C3_HOOKS_FQN}",
        f"+agent.agent_ids={_hydra_list(agent_ids)}",
        f"+agent.model_ids={_hydra_list(actor_groups)}",
        "+agent.model_sharing=false",
        f"+agent.worker_groups={_hydra_dict_list(tuple(worker_groups.items()))}",
        "+agent.orchestra_type=c3_reasoner_actor",
        f"+agent.orchestra.c3.fanout={_hydra_int_list(settings['fanout'])}",
        "+trajweave.recipe=c3_reasoner_actor_math",
        f"+trajweave.config={_quote(source_config)}",
        "+trajweave.coordination_protocol=contextual_counterfactual_prefix_replay",
        "+trajweave.trajectory_schema=c3_prefix_tree_v1",
        "+trajweave.credit_allocator=c3_contextual_counterfactual",
        "+trajweave.verl_extensions=[trajweave_c3_contextual_counterfactual]",
        f"+trajweave.agent_loop_backend={settings['agent_loop_backend']}",
        f"+trajweave.hf_local_model_cache_size={settings['hf_local_model_cache_size']}",
        f"+trajweave.turn_padding_multiple={padding_multiple}",
        "+trajweave.multi_actor.enabled=true",
        "+trajweave.multi_actor.routing_field=worker_group",
        "+trajweave.multi_actor.tokenizer_mode=shared",
        f"+trajweave.c3.credit_variant={settings['credit_variant']}",
        f"+trajweave.c3.baseline_mode={settings['baseline_mode']}",
        f"+trajweave.c3.value_assisted_alpha={settings['value_assisted_alpha']}",
        f"+trajweave.c3.normalize_advantages={str(settings['normalize_advantages']).lower()}",
        f"+trajweave.c3.fanout={_hydra_int_list(settings['fanout'])}",
        f"+actor_rollout_ref.rollout.agent.agent_loop_manager_class={TRAJWEAVE_AGENT_LOOP_MANAGER_FQN}",
    ]
    if settings["credit_variant"] == "reward_only":
        required.extend(["trainer.v1.trainer_mode=trajweave_multi_actor_sync", "critic.enable=false"])
    else:
        critic = settings["critic"]
        route = {
            "critic_group": str(critic.get("critic_group", "c3-q-critic")),
            "actor_groups": list(actor_groups),
            "topology": "centralized",
            "critic_type": "q",
            "model_path": critic.get("model_path"),
            "tokenizer_path": critic.get("tokenizer_path"),
            "gpus": _integer_at_least(critic.get("gpus", 1), field="c3.critic.gpus", minimum=1),
            "max_length": _integer_at_least(
                critic.get("max_length", 2048),
                field="c3.critic.max_length",
                minimum=1,
            ),
        }
        normalize_critic_route_specs(
            [route],
            trainable_actor_groups=actor_groups,
            topology="centralized",
            critic_type="q",
        )
        actor_critic = {
            "topology": "centralized",
            "critic_type": "q",
            "max_length": route["max_length"],
            "value_loss_coef": float(critic.get("value_loss_coef", 1.0)),
            "critic_routes": [route],
        }
        required.extend(
            [
                "trainer.v1.trainer_mode=trajweave_c3_critic_sync",
                "critic.enable=true",
                f"critic.optim.lr={float(critic.get('learning_rate', 5.0e-6))}",
                f"+trajweave.actor_critic={_hydra_mapping(actor_critic)}",
            ]
        )
    overrides = tuple(str(item) for item in config.get("verl", {}).get("overrides", []))
    for item in required:
        overrides = _set_override(overrides, item)
    return overrides


def _validate_critic_settings(settings: dict[str, Any]) -> None:
    critic = settings["critic"]
    if settings["credit_variant"] == "reward_only":
        return
    missing = [field for field in ("model_path", "tokenizer_path") if not str(critic.get(field, "")).strip()]
    if missing:
        raise ValueError(f"C3 {settings['credit_variant']} requires c3.critic fields: {missing}.")
    critic_gpus = _integer_at_least(critic.get("gpus", 1), field="c3.critic.gpus", minimum=1)
    if critic_gpus != 1:
        raise ValueError("The current C3 Q critic runtime requires c3.critic.gpus=1.")
    _integer_at_least(critic.get("max_length", 2048), field="c3.critic.max_length", minimum=1)
    loss_coefficient = float(critic.get("value_loss_coef", 1.0))
    if not isfinite(loss_coefficient) or loss_coefficient <= 0.0:
        raise ValueError("c3.critic.value_loss_coef must be positive and finite.")


def _normalize_worker_groups(c3: dict[str, Any], *, actor_groups: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    raw = c3.get("worker_groups", {}) or {}
    if not isinstance(raw, dict):
        raise ValueError("c3.worker_groups must be a mapping keyed by model id.")
    groups = {}
    for group_id in actor_groups:
        value = raw.get(group_id, {}) or {}
        if not isinstance(value, dict):
            raise ValueError(f"c3.worker_groups.{group_id} must be a mapping.")
        group = dict(value)
        group.setdefault("trainable", True)
        if group["trainable"] is not True:
            raise ValueError(f"C3 requires c3.worker_groups.{group_id}.trainable=true.")
        group["gpus"] = _worker_group_gpus(group, group_id=group_id)
        groups[group_id] = group
    return groups


def _worker_group_gpus(group: dict[str, Any], *, group_id: str) -> int:
    return _integer_at_least(
        group.get("gpus", 1),
        field=f"c3.worker_groups.{group_id}.gpus",
        minimum=1,
    )


def _integer_at_least(value: Any, *, field: str, minimum: int) -> int:
    message = f"{field} must be an integer greater than or equal to {minimum}."
    if isinstance(value, bool):
        raise ValueError(message)
    if isinstance(value, int):
        result = value
    elif isinstance(value, str):
        normalized = value.strip()
        digits = normalized[1:] if normalized[:1] in {"+", "-"} else normalized
        if not digits or not digits.isascii() or not digits.isdigit():
            raise ValueError(message)
        result = int(normalized)
    else:
        raise ValueError(message)
    if result < minimum:
        raise ValueError(message)
    return result


def _resolve_mode(config: dict[str, Any], *, explicit: str | None) -> str:
    if explicit is not None:
        return str(explicit).strip().lower()
    run = config.get("run", {}) or {}
    nested = run.get("mode") if isinstance(run, dict) else None
    return str(config.get("mode", nested) or "").strip().lower()


def _set_override(overrides: tuple[str, ...], replacement: str) -> tuple[str, ...]:
    key = replacement.split("=", 1)[0].lstrip("+")
    return (*(item for item in overrides if item.split("=", 1)[0].lstrip("+") != key), replacement)


def _quote(value: str) -> str:
    return '"' + str(value).replace('"', '\\"') + '"'


def _hydra_list(values: tuple[str, ...]) -> str:
    return "[" + ",".join(_quote(value) for value in values) + "]"


def _hydra_int_list(values: tuple[int, ...]) -> str:
    return "[" + ",".join(str(value) for value in values) + "]"


def _hydra_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int | float):
        return str(value)
    return _quote(str(value))


def _hydra_mapping(value: dict[str, Any]) -> str:
    fields = []
    for key, item in value.items():
        if isinstance(item, dict):
            rendered = _hydra_mapping(item)
        elif isinstance(item, list):
            rendered = (
                "["
                + ",".join(_hydra_mapping(entry) if isinstance(entry, dict) else _hydra_value(entry) for entry in item)
                + "]"
            )
        else:
            rendered = _hydra_value(item)
        fields.append(f"{key}:{rendered}")
    return "{" + ",".join(fields) + "}"


def _hydra_dict_list(groups: tuple[tuple[str, dict[str, Any]], ...]) -> str:
    return "[" + ",".join(_hydra_mapping({"id": group_id, **group}) for group_id, group in groups) + "]"
