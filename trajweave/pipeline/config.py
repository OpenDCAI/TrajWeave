from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def hydra_path(value: str) -> str:
    """保留普通路径写法，并转义中文、空格及 Hydra 语法字符。"""
    if re.fullmatch(r"[A-Za-z0-9_./-]+", value):
        return value
    return json.dumps(value, ensure_ascii=False)


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


def resolve_model_references(value: Any) -> Any:
    """在角色校验前解析模型环境变量，保留由训练子进程解析的 Hydra overrides。"""
    if isinstance(value, dict):
        resolved = {}
        for key, item in value.items():
            if key in {"model_path", "tokenizer_path", "reward_model_name"} and isinstance(item, str):
                if "${oc.env:" in item:
                    from omegaconf import OmegaConf

                    item = OmegaConf.create({"value": item}).value
            resolved[key] = resolve_model_references(item)
        return resolved
    if isinstance(value, list):
        return [resolve_model_references(item) for item in value]
    return value


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
