from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from trajweave.backends.verl.schema import to_python


@dataclass(frozen=True)
class RoutedBatch:
    group_id: str
    batch: Any


def split_batch_by_group_values(batch: Any, group_values: list[str]) -> list[RoutedBatch]:
    if len(group_values) != len(batch.keys):
        raise ValueError(
            "worker_group value count must match batch keys, "
            f"got {len(group_values)} values for {len(batch.keys)} keys."
        )

    grouped_keys: OrderedDict[str, list[str]] = OrderedDict()
    for key, group_id in zip(batch.keys, group_values, strict=True):
        normalized_group = str(group_id)
        if not normalized_group:
            raise ValueError(f"Empty worker_group for batch key {key!r}.")
        grouped_keys.setdefault(normalized_group, []).append(key)

    return [RoutedBatch(group_id=group_id, batch=batch.select_keys(keys)) for group_id, keys in grouped_keys.items()]


def read_tq_group_values(batch: Any, *, field: str = "worker_group") -> list[str]:
    import transfer_queue as tq

    data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=[field])
    if field not in data:
        raise KeyError(f"TransferQueue batch is missing routing field {field!r}.")
    return normalize_group_values(data[field])


def split_tq_batch_by_field(batch: Any, *, field: str = "worker_group") -> list[RoutedBatch]:
    return split_batch_by_group_values(batch, read_tq_group_values(batch, field=field))


def normalize_group_values(values: Any) -> list[str]:
    values = to_python(values)
    if hasattr(values, "tolist"):
        values = values.tolist()
    elif hasattr(values, "data"):
        values = values.data

    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list | tuple):
        try:
            values = list(values)
        except TypeError as exc:
            raise TypeError(f"Cannot normalize worker_group values from {type(values)!r}.") from exc

    normalized = []
    for value in values:
        item = to_python(value)
        if isinstance(item, bytes):
            item = item.decode("utf-8")
        normalized.append(str(item))
    return normalized


def safe_actor_role_key(group_id: str) -> str:
    safe = []
    for char in str(group_id):
        if char.isalnum():
            safe.append(char.lower())
        else:
            safe.append("_")
    key = "".join(safe).strip("_")
    return f"trajweave_actor_{key or 'group'}"


def safe_critic_role_key(group_id: str) -> str:
    return safe_actor_role_key(group_id).replace("trajweave_actor_", "trajweave_critic_", 1)


def safe_worker_role_key(group_id: str) -> str:
    """MAPoRL 旧测试和外部调用的兼容别名。"""

    return safe_actor_role_key(group_id).replace("trajweave_actor_", "maporl_actor_", 1)
