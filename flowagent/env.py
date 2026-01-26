"""Lightweight .env loader for FlowAgent.

This is intentionally dependency-free (no python-dotenv required).
It loads key/value pairs from the project root `.env` file into `os.environ`.

Supported syntax:
- KEY=VALUE
- export KEY=VALUE
- Quotes: KEY="VALUE" or KEY='VALUE'
- Comments starting with #

Security:
- Never commit your real `.env` file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


_LOADED = False


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and ((value[0] == value[-1] == '"') or (value[0] == value[-1] == "'")):
        return value[1:-1]
    return value


def load_project_env(*, override: bool = False, env_path: Optional[Path] = None) -> bool:
    """Load project root `.env` into environment.

    Returns True if a `.env` file existed and was processed.
    """
    global _LOADED
    if _LOADED:
        return False

    path = env_path or (_project_root() / ".env")
    if not path.exists() or not path.is_file():
        _LOADED = True
        return False

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8-sig")

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue

        value = _strip_quotes(value)

        if not override and key in os.environ:
            continue

        os.environ[key] = value

    _LOADED = True
    return True
