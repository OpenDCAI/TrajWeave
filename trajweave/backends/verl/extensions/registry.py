from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

RuntimeExtension = Callable[[Any], None]


def apply_verl_runtime_extensions(config: Any) -> tuple[str, ...]:
    """Apply TrajWeave runtime patches around VERL without editing VERL files."""

    names = _extension_names(config)
    applied: list[str] = []
    for name in names:
        extension = _registry()[name]
        extension(config)
        applied.append(name)
    return tuple(applied)


def _extension_names(config: Any) -> tuple[str, ...]:
    trajweave = _get(config, "trajweave", default={}) or {}
    explicit = _get(trajweave, "verl_extensions", default=None)
    if explicit:
        return _normalize_names(explicit)

    credit_allocator = _get(trajweave, "credit_allocator", default=None)
    recipe = _get(trajweave, "recipe", default=None)
    if credit_allocator == "drmas_agent_wise_grpo" or recipe in {"doctor_mas_math", "doctor_mas_search"}:
        return ("drmas_agent_wise_grpo",)
    return ()


def _normalize_names(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        names = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, Iterable):
        names = [str(part).strip() for part in value if str(part).strip()]
    else:
        raise TypeError(f"trajweave.verl_extensions must be a string or sequence, got {type(value)!r}.")

    unknown = sorted(set(names) - set(_registry()))
    if unknown:
        raise ValueError(f"Unknown TrajWeave VERL runtime extension(s): {unknown}.")
    return tuple(dict.fromkeys(names))


def _registry() -> dict[str, RuntimeExtension]:
    from trajweave.backends.verl.extensions.drmas import apply_drmas_agent_wise_grpo_patch
    from trajweave.backends.verl.extensions.nested_compat import apply_tq_nested_compat_patch

    return {
        "drmas_agent_wise_grpo": apply_drmas_agent_wise_grpo_patch,
        "tq_nested_compat": apply_tq_nested_compat_patch,
    }


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    try:
        return obj.get(key, default)
    except (AttributeError, TypeError):
        return getattr(obj, key, default)
