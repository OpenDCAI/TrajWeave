from __future__ import annotations

import math
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

REDACTED = "[REDACTED]"
_SENSITIVE_KEY_SUFFIXES = (
    "password",
    "passwd",
    "pwd",
    "secret",
    "token",
    "credential",
    "credentials",
    "authorization",
    "cookie",
    "apikey",
    "api_key",
    "accesskey",
    "access_key",
    "privatekey",
    "private_key",
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)(?P<prefix>(?<![a-z0-9_])['\"]?(?:[a-z0-9_.-]*(?:password|passwd|pwd|secret|credential|"
    r"authorization|cookie|api[_-]?key|access[_-]?key|private[_-]?key|secret[_-]?key|auth[_-]?token|"
    r"access[_-]?token|refresh[_-]?token|id[_-]?token|[_.-]token)|token)['\"]?\s*[:=]\s*)"
    r"(?P<value>(?:(?:bearer|basic|token)\s+)?(?:\"[^\"\r\n]*\"|'[^'\r\n]*'|[^\s,;]+))"
)
_SECRET_ARGUMENT_RE = re.compile(
    r"(?i)(?P<prefix>--?(?:password|passwd|pwd|secret|credential|authorization|cookie|api[_-]?key|"
    r"access[_-]?key|private[_-]?key|auth[_-]?token|access[_-]?token|refresh[_-]?token|id[_-]?token|"
    r"token)\s+)(?P<value>\S+)"
)
_BEARER_RE = re.compile(r"(?i)(\bbearer\s+)[a-z0-9._~+/=-]+")
_URL_CREDENTIAL_RE = re.compile(r"(?i)(\b[a-z][a-z0-9+.-]*://[^:/\s]+:)[^@\s]+(@)")


def json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [json_safe(item) for item in value]
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        tensor = value.detach().cpu()
        if hasattr(tensor, "tolist"):
            return tensor.tolist()
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    return value


def is_sensitive_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
    compact = normalized.replace("_", "")
    sensitive_segments = {"password", "passwd", "pwd", "secret", "credential", "credentials", "authorization"}
    return (
        bool(sensitive_segments.intersection(normalized.split("_")))
        or normalized.endswith(_SENSITIVE_KEY_SUFFIXES)
        or compact.endswith(_SENSITIVE_KEY_SUFFIXES)
    )


def redact_text(value: str) -> str:
    value = _URL_CREDENTIAL_RE.sub(rf"\1{REDACTED}\2", value)
    value = _BEARER_RE.sub(rf"\1{REDACTED}", value)
    value = _SECRET_ARGUMENT_RE.sub(rf"\g<prefix>{REDACTED}", value)
    return _SECRET_ASSIGNMENT_RE.sub(rf"\g<prefix>{REDACTED}", value)


def redact_secrets(value: Any) -> Any:
    return _redact_json_value(json_safe(value))


def _redact_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): REDACTED if is_sensitive_key(str(key)) else _redact_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_json_value(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value
