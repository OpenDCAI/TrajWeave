from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from trajweave.backends.verl.multi_actor.critic_config import (
    normalize_critic_route_specs,
    resolve_actor_critic_settings,
)
from trajweave.backends.verl.tokenizer_compat import assert_compatible_tokenizers


def validate_trajweave_config(config: Any, *, use_reference_policy: bool, use_critic: bool) -> None:
    """按每个模型实际占用的 GPU 数验证批量大小，避免把独立角色误当成数据并行。"""
    from copy import deepcopy

    from omegaconf import OmegaConf
    from verl.utils.config import omega_conf_to_dataclass, validate_config

    if not OmegaConf.select(config, "trajweave.multi_actor.enabled", default=False):
        validate_config(config, use_reference_policy, use_critic)
        return
    raw_groups = OmegaConf.to_container(config.agent.worker_groups, resolve=True)
    groups = ([{"id": key, **value} for key, value in raw_groups.items()]
              if isinstance(raw_groups, Mapping) else raw_groups)
    trainable = [group for group in groups if group.get("trainable", True)]
    if not trainable:
        raise ValueError("Multi-actor training requires at least one trainable worker group.")
    for group in trainable:
        _validate_gpu_count(group.get("gpus", 1), family="multi-actor", group_id=group["id"])
        actor_config = deepcopy(config)
        actor_config.trainer.nnodes = 1
        actor_config.trainer.n_gpus_per_node = int(group.get("gpus", 1))
        validate_config(actor_config, use_reference_policy, use_critic=False)
    if use_critic:
        settings = resolve_actor_critic_settings(config)
        if settings is None:
            critic_gpu_counts = [config.trainer.nnodes * config.trainer.n_gpus_per_node]
        else:
            routes = normalize_critic_route_specs(
                settings.get("critic_routes", settings.get("critic_groups", settings.get("critics"))),
                trainable_actor_groups=[group["id"] for group in trainable],
                topology=settings.get("topology"), critic_type=settings.get("critic_type"),
                max_length=int(settings.get("max_length", 2048)),
            )
            critic_gpu_counts = [route.gpus for route in routes]
        for n_gpus in critic_gpu_counts:
            omega_conf_to_dataclass(config.critic).validate(n_gpus, config.data.train_batch_size)


def reconcile_multi_actor_global_assets(config: Any) -> dict[str, str] | None:
    """Align VERL's singleton actor config with the primary trainable worker group."""

    from omegaconf import DictConfig, OmegaConf, open_dict

    if isinstance(config, DictConfig):
        enabled = bool(OmegaConf.select(config, "trajweave.multi_actor.enabled", default=False))
        raw_groups = OmegaConf.select(config, "agent.worker_groups", default=[])
        raw_groups = OmegaConf.to_container(raw_groups, resolve=True) if OmegaConf.is_config(raw_groups) else raw_groups
    elif isinstance(config, dict):
        trajweave = config.get("trajweave", {}) or {}
        multi_actor = trajweave.get("multi_actor", {}) if isinstance(trajweave, Mapping) else {}
        enabled = bool(multi_actor.get("enabled", False)) if isinstance(multi_actor, Mapping) else False
        agent = config.get("agent", {}) or {}
        raw_groups = agent.get("worker_groups", []) if isinstance(agent, Mapping) else []
    else:
        raise TypeError(f"multi-actor global asset reconciliation requires DictConfig or dict, got {type(config)!r}.")

    if not enabled:
        return None
    if isinstance(raw_groups, Mapping):
        groups = [{"id": str(group_id), **dict(group or {})} for group_id, group in raw_groups.items()]
    elif isinstance(raw_groups, Sequence) and not isinstance(raw_groups, str | bytes):
        groups = [dict(group) for group in raw_groups]
    else:
        raise ValueError("agent.worker_groups must be a list or mapping for multi-actor training.")
    primary = next((group for group in groups if bool(group.get("trainable", True))), None)
    if primary is None:
        raise ValueError("Multi-actor training requires a trainable primary worker group.")
    model_path = str(primary.get("model_path") or "").strip()
    tokenizer_path = str(primary.get("tokenizer_path") or "").strip()
    if not model_path or not tokenizer_path:
        raise ValueError("Primary multi-actor worker group requires model_path and tokenizer_path.")
    assets = {
        "worker_group": str(primary.get("id") or ""),
        "model_path": model_path,
        "tokenizer_path": tokenizer_path,
    }
    trainable_group_ids = [
        str(group.get("id") or "")
        for group in groups
        if bool(group.get("trainable", True)) and str(group.get("id") or "")
    ]
    if isinstance(config, DictConfig):
        with open_dict(config):
            OmegaConf.update(config, "actor_rollout_ref.model.path", model_path, merge=False, force_add=True)
            OmegaConf.update(
                config,
                "actor_rollout_ref.model.tokenizer_path",
                tokenizer_path,
                merge=False,
                force_add=True,
            )
        _reconcile_multi_actor_global_critic_assets(config, trainable_actor_groups=trainable_group_ids)
        _disable_unavailable_comlrl_rollout_logprobs(config)
        return assets
    actor_model = config.setdefault("actor_rollout_ref", {}).setdefault("model", {})
    actor_model["path"] = model_path
    actor_model["tokenizer_path"] = tokenizer_path
    _reconcile_multi_actor_global_critic_assets(config, trainable_actor_groups=trainable_group_ids)
    _disable_unavailable_comlrl_rollout_logprobs(config)
    return assets


def _reconcile_multi_actor_global_critic_assets(
    config: Any,
    *,
    trainable_actor_groups: Sequence[str],
) -> None:
    """Align VERL's validation-only singleton critic with the first routed critic."""

    from omegaconf import DictConfig, OmegaConf, open_dict

    settings = resolve_actor_critic_settings(config)
    if settings is None:
        return
    routes = normalize_critic_route_specs(
        settings.get("critic_routes", settings.get("critic_groups", settings.get("critics"))),
        trainable_actor_groups=trainable_actor_groups,
        topology=settings.get("topology"),
        critic_type=settings.get("critic_type"),
        max_length=int(settings.get("max_length", 2048)),
    )
    primary = routes[0]
    if not primary.model_path or not primary.tokenizer_path:
        raise ValueError("Primary routed critic requires model_path and tokenizer_path.")
    if isinstance(config, DictConfig):
        with open_dict(config):
            OmegaConf.update(config, "critic.model.path", primary.model_path, merge=False, force_add=True)
            OmegaConf.update(
                config,
                "critic.model.tokenizer_path",
                primary.tokenizer_path,
                merge=False,
                force_add=True,
            )
        return
    critic_model = config.setdefault("critic", {}).setdefault("model", {})
    critic_model["path"] = primary.model_path
    critic_model["tokenizer_path"] = primary.tokenizer_path


def _disable_unavailable_comlrl_rollout_logprobs(config: Any) -> None:
    """CoMLRL HF-local outputs do not carry sampling-policy log probabilities."""

    from omegaconf import DictConfig, OmegaConf, open_dict

    if isinstance(config, DictConfig):
        recipe = OmegaConf.select(config, "trajweave.recipe", default=None)
        comlrl = OmegaConf.select(config, "trajweave.comlrl", default=None)
        if recipe != "comlrl_joint_math" and comlrl is None:
            return
        with open_dict(config):
            OmegaConf.update(
                config,
                "actor_rollout_ref.rollout.calculate_log_probs",
                False,
                merge=False,
                force_add=True,
            )
        return
    trajweave = config.get("trajweave", {}) or {}
    if not isinstance(trajweave, Mapping):
        return
    if trajweave.get("recipe") != "comlrl_joint_math" and "comlrl" not in trajweave:
        return
    rollout = config.setdefault("actor_rollout_ref", {}).setdefault("rollout", {})
    rollout["calculate_log_probs"] = False


def normalize_worker_groups(
    raw_groups: Any,
    *,
    model_ids: Sequence[str],
    family: str,
) -> dict[str, dict[str, Any]]:
    """把 recipe 中的 Worker Group 映射规范化为稳定、可路由的顺序映射。"""

    if raw_groups is None:
        raw_groups = {}
    if not isinstance(raw_groups, Mapping):
        raise ValueError(f"{family}.worker_groups must be a mapping keyed by model id.")

    normalized_model_ids = tuple(str(model_id) for model_id in model_ids)
    groups: dict[str, dict[str, Any]] = {}
    for model_id in dict.fromkeys(normalized_model_ids):
        raw_group = raw_groups.get(model_id, {})
        if raw_group is None:
            raw_group = {}
        if not isinstance(raw_group, Mapping):
            raise ValueError(f"{family}.worker_groups.{model_id} must be a mapping.")
        group = dict(raw_group)
        group.setdefault("trainable", True)
        groups[model_id] = group

    for raw_group_id, raw_group in raw_groups.items():
        group_id = str(raw_group_id)
        if group_id in groups:
            continue
        if not isinstance(raw_group, Mapping):
            raise ValueError(f"{family}.worker_groups.{group_id} must be a mapping.")
        group = dict(raw_group)
        group.setdefault("trainable", False)
        groups[group_id] = group
    return groups


def validate_multi_actor_worker_groups(
    settings: Mapping[str, Any],
    *,
    worker_groups: Mapping[str, Mapping[str, Any]],
    model_ids: Sequence[str],
    enabled: bool,
    family: str,
) -> dict[str, Any]:
    """校验多 Actor 公共合约，不包含任何论文特有逻辑。"""

    tokenizer_mode = str(settings.get("tokenizer_mode", "shared"))
    allowed_tokenizer_modes = {"shared", "compatible"}
    if tokenizer_mode not in allowed_tokenizer_modes:
        raise ValueError(
            f"{family} multi_actor_training supports tokenizer_mode in "
            f"{sorted(allowed_tokenizer_modes)}; got {tokenizer_mode!r}."
        )

    trainable_groups = tuple(
        str(group_id) for group_id, group in worker_groups.items() if bool(group.get("trainable", True))
    )
    result: dict[str, Any] = {
        "status": "disabled" if not enabled else "passed",
        "tokenizer_mode": tokenizer_mode,
        "trainable_worker_groups": trainable_groups,
    }
    if not enabled:
        return result
    if len(trainable_groups) < 2:
        raise ValueError(f"{family} multi_actor_training requires at least two trainable worker groups.")

    normalized_model_ids = tuple(str(model_id) for model_id in model_ids)
    missing_groups = [model_id for model_id in dict.fromkeys(normalized_model_ids) if model_id not in worker_groups]
    if missing_groups:
        raise ValueError(f"{family} model_ids missing worker_groups entries: {missing_groups}.")
    frozen_model_ids = sorted(set(normalized_model_ids) - set(trainable_groups))
    if frozen_model_ids:
        raise ValueError(
            f"{family} multi_actor_training requires each referenced model_id to be trainable; "
            f"frozen model_ids: {frozen_model_ids}."
        )

    missing_fields: list[str] = []
    tokenizer_paths: list[str] = []
    for group_id in trainable_groups:
        group = worker_groups[group_id]
        if not group.get("model_path"):
            missing_fields.append(f"{group_id}.model_path")
        if not group.get("tokenizer_path"):
            missing_fields.append(f"{group_id}.tokenizer_path")
        else:
            tokenizer_paths.append(str(group["tokenizer_path"]))
        _validate_gpu_count(group.get("gpus", 1), family=family, group_id=group_id)
    if missing_fields:
        raise ValueError(
            f"{family} multi_actor_training requires explicit model_path and tokenizer_path for every "
            f"trainable worker group; missing: {missing_fields}."
        )

    unique_tokenizer_paths = sorted(set(tokenizer_paths))
    if tokenizer_mode == "shared":
        if len(unique_tokenizer_paths) != 1:
            raise ValueError(
                f"{family} shared-tokenizer multi_actor_training requires identical tokenizer_path across "
                f"trainable worker groups. Got tokenizer paths: {unique_tokenizer_paths}."
            )
        result["shared_tokenizer_path"] = unique_tokenizer_paths[0]
        return result

    fingerprints = assert_compatible_tokenizers(unique_tokenizer_paths)
    if fingerprints:
        result["compatible_tokenizer_digest"] = next(iter(fingerprints.values())).digest
        result["compatible_tokenizer_paths"] = unique_tokenizer_paths
    return result


def hydra_string_list(values: Sequence[str]) -> str:
    return "[" + ",".join(_quote(value) for value in values) + "]"


def hydra_float_list(values: Sequence[float]) -> str:
    return "[" + ",".join(str(float(value)) for value in values) + "]"


def hydra_worker_group_list(groups: Sequence[tuple[str, Mapping[str, Any]]]) -> str:
    items: list[str] = []
    for group_id, group in groups:
        fields = [f"id:{_quote(group_id)}", f"trainable:{str(bool(group.get('trainable', True))).lower()}"]
        for key in ("model_path", "tokenizer_path", "gpus"):
            if key in group:
                fields.append(f"{key}:{_hydra_value(group[key])}")
        items.append("{" + ",".join(fields) + "}")
    return "[" + ",".join(items) + "]"


def replace_override(overrides: Sequence[str], key: str, value: str) -> tuple[str, ...]:
    output: list[str] = []
    replaced = False
    for item in overrides:
        item_key = str(item).split("=", 1)[0].lstrip("+")
        if item_key == key:
            if not replaced:
                output.append(value)
                replaced = True
            continue
        output.append(str(item))
    if not replaced:
        output.append(value)
    return tuple(output)


def _quote(value: Any) -> str:
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _hydra_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int | float):
        return str(value)
    return _quote(value)


def _validate_gpu_count(value: Any, *, family: str, group_id: str) -> None:
    if isinstance(value, bool):
        raise ValueError(f"{family} worker group {group_id!r} gpus must be a positive integer.")
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{family} worker group {group_id!r} gpus must be a positive integer.") from exc
    if count <= 0:
        raise ValueError(f"{family} worker group {group_id!r} gpus must be a positive integer.")
