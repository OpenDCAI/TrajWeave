from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("YAML config loading requires PyYAML.") from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def recipe_name(config: dict[str, Any]) -> str:
    value = _required_config_value(config, "recipe")
    return str(value)


def mode_name(config: dict[str, Any]) -> str:
    value = _required_config_value(config, "mode")
    return str(value)


def _required_config_value(config: dict[str, Any], key: str) -> Any:
    run_config = config.get("run", {})
    if run_config is None:
        run_config = {}
    if not isinstance(run_config, dict):
        raise ValueError("run config must be a mapping.")
    value = config.get(key, run_config.get(key))
    if value is None or not str(value).strip():
        raise ValueError(f"Config must explicitly define a non-empty {key!r}.")
    return value
