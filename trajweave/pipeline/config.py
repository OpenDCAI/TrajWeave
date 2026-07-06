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
    return str(config.get("recipe", config.get("run", {}).get("recipe", "doctor_mas_math")))


def mode_name(config: dict[str, Any]) -> str:
    return str(config.get("mode", config.get("run", {}).get("mode", "smoke")))
