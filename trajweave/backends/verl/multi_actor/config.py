from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from trajweave.backends.verl.tokenizer_compat import assert_compatible_tokenizers


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
