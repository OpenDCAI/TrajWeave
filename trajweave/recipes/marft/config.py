from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from math import isfinite
from pathlib import Path
from typing import Any

from trajweave.backends.verl.multi_actor import (
    hydra_string_list,
    hydra_worker_group_list,
    normalize_worker_groups,
    replace_override,
    validate_multi_actor_worker_groups,
)
from trajweave.backends.verl.multi_actor.critic_config import normalize_critic_route_specs
from trajweave.backends.verl.runtime_config import normalize_hf_local_dtype
from trajweave.orchestration.marft import MARFTWorkflowGraph
from trajweave.recipes.marft.math_workflow import DEFAULT_ROLE_PROMPTS

TRAJWEAVE_AGENT_LOOP_MANAGER_FQN = "trajweave.backends.verl.agent_loop.TrajWeaveAgentLoopManager"
MARFT_HOOK_FQN = "trajweave.backends.verl.extensions.marft.MARFTPPOHooks"


def resolve_marft_settings(
    config: dict[str, Any],
    *,
    require_worker_assets: bool = False,
) -> dict[str, Any]:
    marft = config.get("marft", {}) or {}
    if not isinstance(marft, Mapping):
        raise ValueError("MARFT config must be a mapping.")
    role_names = tuple(str(value).strip() for value in marft.get("role_names", ["planner", "solver"]))
    if len(role_names) < 2 or len(role_names) != len(set(role_names)) or any(not role for role in role_names):
        raise ValueError("MARFT requires at least two unique non-empty role_names.")
    raw_role_configs = marft.get("role_configs", {}) or {}
    if not isinstance(raw_role_configs, Mapping):
        raise ValueError("MARFT role_configs must be a mapping keyed by role name.")
    role_configs: dict[str, dict[str, Any]] = {}
    for role in role_names:
        raw = raw_role_configs.get(role, {}) or {}
        if not isinstance(raw, Mapping):
            raise ValueError(f"MARFT role_configs.{role} must be a mapping.")
        role_config = dict(raw)
        role_config.setdefault("system_prompt", DEFAULT_ROLE_PROMPTS.get(role, f"Act as the {role} role."))
        if not str(role_config["system_prompt"]).strip():
            raise ValueError(f"MARFT role_configs.{role}.system_prompt must be non-empty.")
        role_configs[role] = role_config
    extra_roles = sorted(set(raw_role_configs) - set(role_names))
    if extra_roles:
        raise ValueError(f"MARFT role_configs contains unused roles: {extra_roles}.")

    graph_type = str(marft.get("graph_type", "sequential")).strip().lower()
    transition_messages = marft.get("transition_messages")
    if graph_type == "sequential":
        graph = MARFTWorkflowGraph.sequential(role_names, transition_messages)
    elif graph_type == "custom":
        graph = MARFTWorkflowGraph.from_config(dict(marft.get("graph_config", {}) or {}))
    else:
        raise ValueError("MARFT graph_type must be sequential or custom.")
    graph_roles = {node.role_name for node in graph.nodes}
    unknown_graph_roles = sorted(graph_roles - set(role_names))
    if unknown_graph_roles:
        raise ValueError(f"MARFT graph references unknown roles: {unknown_graph_roles}.")

    strategy = str(marft.get("credit_strategy", "equal")).strip().lower()
    if strategy not in {"equal", "step_discount", "per_step"}:
        raise ValueError("MARFT credit_strategy must be equal, step_discount, or per_step.")
    credit_discount = _unit_float(marft.get("credit_discount", 1.0), field="credit_discount")
    return_gamma = _unit_float(marft.get("return_gamma", marft.get("gamma", 1.0)), field="return_gamma")
    step_reward_fn = marft.get("step_reward_fn")
    raw_per_agent_reward_fns = marft.get("per_agent_reward_fns", {}) or {}
    if not isinstance(raw_per_agent_reward_fns, Mapping):
        raise ValueError("MARFT per_agent_reward_fns must be a mapping.")
    per_agent_reward_fns = dict(raw_per_agent_reward_fns)
    for role, role_config in role_configs.items():
        if role_config.get("reward_fn") is not None:
            per_agent_reward_fns.setdefault(role, role_config["reward_fn"])
    unknown_reward_roles = sorted(set(per_agent_reward_fns) - set(role_names))
    if unknown_reward_roles:
        raise ValueError(f"MARFT per_agent_reward_fns references unknown roles: {unknown_reward_roles}.")
    if step_reward_fn is not None and not str(step_reward_fn).strip():
        raise ValueError("MARFT step_reward_fn must be a non-empty import path.")
    if any(not str(value).strip() for value in per_agent_reward_fns.values()):
        raise ValueError("MARFT per_agent_reward_fns values must be non-empty import paths.")
    if strategy == "per_step" and not step_reward_fn and not per_agent_reward_fns:
        raise ValueError("MARFT per_step credit requires step_reward_fn or per_agent_reward_fns.")

    shared_policy = bool(marft.get("shared_policy", True))
    use_multi_lora = bool(marft.get("use_multi_lora", True))
    shared_lora = bool(marft.get("shared_lora", True))
    lora_rank = int(marft.get("lora_rank", 32))
    lora_alpha = int(marft.get("lora_alpha", 16))
    lora_target_modules = str(marft.get("lora_target_modules", "all-linear")).strip()
    if use_multi_lora and (lora_rank <= 0 or lora_alpha <= 0 or not lora_target_modules):
        raise ValueError("MARFT LoRA mode requires positive lora_rank/lora_alpha and lora_target_modules.")
    kl_coef = _nonnegative_float(marft.get("kl_coef", 0.1), field="kl_coef")
    kl_penalty = str(marft.get("kl_penalty", "low_var_kl")).strip().lower()
    if kl_penalty not in {"kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"}:
        raise ValueError("MARFT kl_penalty must be a VERL token KL estimator (k1, k2, k3, abs, or low_var_kl).")
    if kl_coef > 0.0 and not use_multi_lora:
        raise ValueError(
            "MARFT kl_coef>0 requires use_multi_lora=true so the frozen base model can serve as reference policy."
        )
    ref_log_prob_micro_batch_size_per_gpu = _positive_int(
        marft.get("ref_log_prob_micro_batch_size_per_gpu", 1),
        field="ref_log_prob_micro_batch_size_per_gpu",
    )
    default_model_ids = (
        ("shared",) * len(role_names)
        if shared_policy and (not use_multi_lora or shared_lora)
        else tuple(f"policy_{role}" for role in role_names)
    )
    model_ids = tuple(str(value).strip() for value in marft.get("model_ids", default_model_ids))
    if len(model_ids) != len(role_names) or any(not model_id for model_id in model_ids):
        raise ValueError("MARFT model_ids must align with role_names and be non-empty.")
    if shared_policy and (not use_multi_lora or shared_lora) and len(set(model_ids)) != 1:
        raise ValueError("MARFT shared policy/shared LoRA mode requires one shared model_id.")
    if not shared_policy and len(set(model_ids)) != len(model_ids):
        raise ValueError("MARFT shared_policy=false requires one unique model_id per role.")
    worker_groups = normalize_worker_groups(
        marft.get("worker_groups", {}),
        model_ids=model_ids,
        family="MARFT",
    )
    multi_actor_training = len(set(model_ids)) > 1
    if require_worker_assets:
        _validate_worker_assets(
            marft,
            worker_groups=worker_groups,
            model_ids=model_ids,
            multi_actor_training=multi_actor_training,
        )

    independent_critic = marft.get("independent_critic")
    if independent_critic is not None:
        independent_critic = str(independent_critic).strip().lower()
        if independent_critic not in {"lora", "separate"}:
            raise ValueError(
                "MARFT independent_critic must be null, lora, or separate; multi_head is not supported "
                "by TrajWeave's role-routed critic worker API."
            )
        if independent_critic == "lora" and not use_multi_lora:
            raise ValueError("MARFT independent_critic=lora requires use_multi_lora=true.")
        if not multi_actor_training:
            raise ValueError(
                "TrajWeave maps MARFT independent critics to role-routed critic worker groups; "
                "configure distinct model_ids (shared_policy=false or shared_lora=false)."
            )
        if kl_coef > 0.0:
            raise ValueError(
                "MARFT independent critics currently require kl_coef=0 because their row-level TD path does not "
                "yet incorporate token-level reference-policy rewards."
            )
    critic_mode = "independent" if independent_critic is not None else "shared"
    agent_loop_backend = str(marft.get("agent_loop_backend", "hf_local_tq"))
    if agent_loop_backend not in {"synthetic_tq", "hf_local_tq"}:
        raise ValueError("MARFT agent_loop_backend must be synthetic_tq or hf_local_tq.")
    dtype = normalize_hf_local_dtype(marft.get("hf_local_dtype", "fp32"))
    return {
        "role_names": role_names,
        "role_configs": role_configs,
        "graph_type": graph_type,
        "graph": graph,
        "credit_strategy": strategy,
        "credit_discount": credit_discount,
        "return_gamma": return_gamma,
        "step_reward_fn": None if step_reward_fn is None else str(step_reward_fn),
        "per_agent_reward_fns": {str(key): str(value) for key, value in per_agent_reward_fns.items()},
        "shared_policy": shared_policy,
        "use_multi_lora": use_multi_lora,
        "shared_lora": shared_lora,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_target_modules": lora_target_modules,
        "kl_coef": kl_coef,
        "kl_penalty": kl_penalty,
        "ref_log_prob_micro_batch_size_per_gpu": ref_log_prob_micro_batch_size_per_gpu,
        "model_ids": model_ids,
        "worker_groups": worker_groups,
        "multi_actor_training": multi_actor_training,
        "independent_critic": independent_critic,
        "critic_mode": critic_mode,
        "tokenizer_mode": str(marft.get("tokenizer_mode", "shared")),
        "agent_loop_backend": agent_loop_backend,
        "hf_local_dtype": dtype,
    }


def build_marft_launch_overrides(config: dict[str, Any], *, config_path: str | None) -> tuple[str, ...]:
    marft = config.get("marft", {}) or {}
    verl = config.get("verl", {}) or {}
    settings = resolve_marft_settings(config, require_worker_assets=True)
    if settings["return_gamma"] != 1.0:
        raise ValueError(
            "MARFT VERL training currently requires return_gamma=1.0. Role-row projection cannot faithfully "
            "represent upstream token-distance discounting when gamma<1."
        )
    trainer_mode = "trajweave_marft_shared_critic_sync"
    if settings["multi_actor_training"]:
        trainer_mode = (
            "trajweave_marft_independent_critic_sync"
            if settings["critic_mode"] == "independent"
            else "trajweave_marft_multi_actor_sync"
        )
    overrides = tuple(str(item) for item in verl.get("overrides", []))
    _reject_value_critic_hf_export(overrides)
    overrides = replace_override(
        overrides,
        "trainer.v1.trainer_mode",
        f"trainer.v1.trainer_mode={trainer_mode}",
    )
    overrides = replace_override(
        overrides,
        "actor_rollout_ref.rollout.calculate_log_probs",
        "actor_rollout_ref.rollout.calculate_log_probs=false",
    )
    controlled_overrides = {
        "algorithm.gamma": "algorithm.gamma=1.0",
        "algorithm.lam": "algorithm.lam=1.0",
        "algorithm.use_kl_in_reward": (f"algorithm.use_kl_in_reward={str(settings['kl_coef'] > 0.0).lower()}"),
        "algorithm.kl_penalty": f"algorithm.kl_penalty={settings['kl_penalty']}",
        "algorithm.kl_ctrl.type": "algorithm.kl_ctrl.type=fixed",
        "algorithm.kl_ctrl.kl_coef": f"algorithm.kl_ctrl.kl_coef={settings['kl_coef']}",
        "actor_rollout_ref.actor.use_kl_loss": "actor_rollout_ref.actor.use_kl_loss=false",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": (
            "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="
            f"{settings['ref_log_prob_micro_batch_size_per_gpu']}"
        ),
    }
    for key, value in controlled_overrides.items():
        overrides = replace_override(overrides, key, value)
    primary_group = settings["worker_groups"][settings["model_ids"][0]]
    for key, field in (
        ("actor_rollout_ref.model.path", "model_path"),
        ("actor_rollout_ref.model.tokenizer_path", "tokenizer_path"),
    ):
        overrides = replace_override(overrides, key, f"{key}={_hydra_value(primary_group[field])}")

    graph = settings["graph"]
    graph_config = {
        "nodes": [
            {
                "id": node.node_id,
                "role_name": node.role_name,
                **({"transition_message": node.transition_message} if node.transition_message is not None else {}),
            }
            for node in graph.nodes
        ],
        "edges": [list(edge) for edge in graph.edges],
    }
    source_config = config_path or str(Path.cwd())
    required = [
        "trainer.use_v1=true",
        "algorithm.adv_estimator=gae",
        "++algorithm.group_by_agent_id=false",
        f"++algorithm.extension_hooks_class={MARFT_HOOK_FQN}",
        "critic.enable=true",
        "actor_rollout_ref.actor.policy_loss.loss_mode=vanilla",
        f"+agent.agent_ids={hydra_string_list(settings['role_names'])}",
        f"+agent.model_ids={hydra_string_list(settings['model_ids'])}",
        f"+agent.model_sharing={str(len(set(settings['model_ids'])) == 1).lower()}",
        f"+agent.worker_group_ids={hydra_string_list(tuple(settings['worker_groups']))}",
        f"+agent.worker_groups={hydra_worker_group_list(tuple(settings['worker_groups'].items()))}",
        "+agent.orchestra_type=marft",
        f"+agent.orchestra.marft.role_names={hydra_string_list(settings['role_names'])}",
        f"+agent.orchestra.marft.role_configs={_hydra_value(settings['role_configs'])}",
        f"+agent.orchestra.marft.graph_config={_hydra_value(graph_config)}",
        f"+agent.orchestra.marft.credit_strategy={settings['credit_strategy']}",
        f"+agent.orchestra.marft.credit_discount={settings['credit_discount']}",
        f"+agent.orchestra.marft.return_gamma={settings['return_gamma']}",
        f"+agent.orchestra.marft.step_reward_fn={_hydra_value(settings['step_reward_fn'] or '')}",
        f"+agent.orchestra.marft.per_agent_reward_fns={_hydra_value(settings['per_agent_reward_fns'])}",
        "+trajweave.recipe=marft_math_workflow",
        f"+trajweave.config={_hydra_value(source_config)}",
        "+trajweave.coordination_protocol=marft_static_dag",
        "+trajweave.trajectory_schema=marft_role_turn_v1",
        "+trajweave.credit_allocator=marft_ctde",
        "+trajweave.verl_extensions=[trajweave_marft_ctde]",
        f"+trajweave.agent_loop_backend={settings['agent_loop_backend']}",
        f"+trajweave.hf_local_dtype={settings['hf_local_dtype']}",
        f"+trajweave.multi_actor.enabled={str(settings['multi_actor_training']).lower()}",
        "+trajweave.multi_actor.routing_field=worker_group",
        f"+trajweave.multi_actor.tokenizer_mode={settings['tokenizer_mode']}",
        "+trajweave.multi_actor.metric_namespace=marft",
        f"+actor_rollout_ref.rollout.agent.agent_loop_manager_class={TRAJWEAVE_AGENT_LOOP_MANAGER_FQN}",
    ]
    if settings["use_multi_lora"]:
        required.extend(
            (
                f"actor_rollout_ref.model.lora_rank={settings['lora_rank']}",
                f"actor_rollout_ref.model.lora_alpha={settings['lora_alpha']}",
                f"actor_rollout_ref.model.target_modules={_hydra_value(settings['lora_target_modules'])}",
            )
        )
        if settings["critic_mode"] == "shared":
            required.extend(
                (
                    f"critic.model.lora_rank={settings['lora_rank']}",
                    f"critic.model.lora_alpha={settings['lora_alpha']}",
                    f"critic.model.target_modules={_hydra_value(settings['lora_target_modules'])}",
                )
            )
    if settings["critic_mode"] == "independent":
        actor_critic = _independent_critic_config(marft, settings=settings)
        required.append(f"+trajweave.actor_critic={_hydra_value(actor_critic)}")
        if settings["independent_critic"] == "lora":
            required.extend(
                (
                    f"critic.model.lora_rank={settings['lora_rank']}",
                    f"critic.model.lora_alpha={settings['lora_alpha']}",
                    f"critic.model.target_modules={_hydra_value(settings['lora_target_modules'])}",
                )
            )
        else:
            overrides = replace_override(overrides, "critic.model.lora_rank", "critic.model.lora_rank=0")
    return overrides + tuple(required)


def _reject_value_critic_hf_export(overrides: tuple[str, ...]) -> None:
    for override in overrides:
        key, separator, raw_value = override.partition("=")
        if not separator or key.strip().lstrip("+") != "critic.checkpoint.save_contents":
            continue
        values = raw_value.strip().removeprefix("[").removesuffix("]").split(",")
        normalized = {value.strip().strip("'\"") for value in values}
        if "hf_model" in normalized:
            raise ValueError(
                "MARFT value critic checkpoints do not support critic.checkpoint.save_contents=hf_model. "
                "Use the default model/optimizer/extra checkpoint contents for resumable training."
            )


def load_reward_callable(path: str) -> Callable[..., float]:
    module_name, separator, attribute = str(path).strip().partition(":")
    if not separator:
        module_name, separator, attribute = str(path).strip().rpartition(".")
    if not module_name or not separator or not attribute:
        raise ValueError(f"Invalid MARFT reward function import path: {path!r}.")
    value = getattr(importlib.import_module(module_name), attribute)
    if not callable(value):
        raise TypeError(f"MARFT reward function {path!r} is not callable.")
    return value


def _validate_worker_assets(
    marft: Mapping[str, Any],
    *,
    worker_groups: Mapping[str, Mapping[str, Any]],
    model_ids: tuple[str, ...],
    multi_actor_training: bool,
) -> None:
    if multi_actor_training:
        validate_multi_actor_worker_groups(
            marft,
            worker_groups=worker_groups,
            model_ids=model_ids,
            enabled=True,
            family="MARFT",
        )
        return
    group_id = model_ids[0]
    group = worker_groups[group_id]
    missing = [field for field in ("model_path", "tokenizer_path") if not group.get(field)]
    if missing:
        raise ValueError(f"MARFT shared worker group {group_id!r} requires {missing}.")


def _independent_critic_config(marft: Mapping[str, Any], *, settings: dict[str, Any]) -> dict[str, Any]:
    raw = dict(marft.get("actor_critic", {}) or {})
    raw.setdefault("topology", "independent")
    raw.setdefault("critic_type", "v")
    raw.setdefault("gamma", settings["return_gamma"])
    raw.setdefault("normalize_advantages", True)
    raw.setdefault("value_loss_coef", 0.6)
    if "critic_routes" not in raw:
        raw["critic_routes"] = [
            {
                "critic_group": f"critic_{group_id}",
                "actor_groups": [group_id],
                "topology": "independent",
                "critic_type": raw["critic_type"],
                "model_path": group["model_path"],
                "tokenizer_path": group["tokenizer_path"],
                "gpus": 1,
            }
            for group_id, group in settings["worker_groups"].items()
            if bool(group.get("trainable", True))
        ]
    normalize_critic_route_specs(
        raw["critic_routes"],
        trainable_actor_groups=tuple(dict.fromkeys(settings["model_ids"])),
        topology="independent",
        critic_type=str(raw["critic_type"]),
        max_length=int(raw.get("max_length", 2048)),
    )
    return raw


def _hydra_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, Mapping):
        return "{" + ",".join(f"{key}:{_hydra_value(item)}" for key, item in value.items()) + "}"
    if isinstance(value, list | tuple):
        return "[" + ",".join(_hydra_value(item) for item in value) + "]"
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _unit_float(value: Any, *, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"MARFT {field} must be a finite number in [0, 1].") from exc
    if not isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"MARFT {field} must be a finite number in [0, 1].")
    return parsed


def _nonnegative_float(value: Any, *, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"MARFT {field} must be a finite non-negative number.") from exc
    if not isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"MARFT {field} must be a finite non-negative number.")
    return parsed


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"MARFT {field} must be a positive integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"MARFT {field} must be a positive integer.") from exc
    if parsed <= 0:
        raise ValueError(f"MARFT {field} must be a positive integer.")
    return parsed
