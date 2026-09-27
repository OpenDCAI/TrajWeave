# SPDX-License-Identifier: Apache-2.0
"""Math-solution cleanup adapted from EIT-EAST-Lab/C3 commit 628185be.

See ``licenses/C3-Apache-2.0.txt`` and ``Notice.txt``.
"""

from __future__ import annotations

import re

_FENCED_BLOCK = re.compile(
    r"```(?P<lang>[A-Za-z0-9_-]*)\s*\n(?P<body>[\s\S]*?)```",
    re.MULTILINE,
)
_ERROR_LINE = re.compile(r"\b(?:Traceback|Exception|[A-Za-z_]*Error)\b")
_LATEX_TEXT_BLOCK = re.compile(r"\\(?:text|mathrm|mathbf|textbf|textit|operatorname)\s*\{\s*([^{}]*?)\s*\}")
_SPECIAL_TOKENS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|assistant|>",
    "<|user|>",
    "<|system|>",
    "<|tool|>",
    "<|endoftext|>",
    "</s>",
    "<s>",
)
_MARKDOWN_ROLE_HEADER = re.compile(r"\*\*\s*(actor|reasoner|assistant|system|user)\s*\*\*", re.IGNORECASE)
_ROLE_PREFIX = re.compile(r"^\s*(actor|reasoner|assistant|system|user)\s*:\s*", re.IGNORECASE)
_ANSWER_PREFIX = re.compile(r"^\s*(final\s+answer|answer|final)\s*:\s*", re.IGNORECASE)
_SEPARATOR_LINE = re.compile(r"^\s*(?:={3,}|-{3,}|_{3,}|\*{3,})\s*$")


def sanitize_math_solution_text(text: str | None, *, strip_answer_prefix: bool = True) -> str:
    """Remove leaked chat markers and tool/error transcripts from math text."""

    sanitized = "" if text is None else str(text)
    sanitized = _LATEX_TEXT_BLOCK.sub(lambda match: f" {match.group(1)} ", sanitized)

    for token in _SPECIAL_TOKENS:
        sanitized = sanitized.replace(token, " ")
    sanitized = re.sub(r"\bim_start\b", " ", sanitized)
    sanitized = re.sub(r"\bim_end\b", " ", sanitized)
    sanitized = _MARKDOWN_ROLE_HEADER.sub(" ", sanitized)

    def replace_fenced_block(match: re.Match[str]) -> str:
        language = (match.group("lang") or "").strip().lower()
        body = match.group("body") or ""
        if language in {"python", "py", "output", "bash", "sh"} or _ERROR_LINE.search(body):
            return "\n"
        return match.group(0)

    sanitized = _FENCED_BLOCK.sub(replace_fenced_block, sanitized)

    kept_lines: list[str] = []
    for raw_line in sanitized.splitlines():
        if _SEPARATOR_LINE.match(raw_line) or _ERROR_LINE.search(raw_line):
            continue
        line = _ROLE_PREFIX.sub("", raw_line)
        if strip_answer_prefix:
            line = _ANSWER_PREFIX.sub("", line)
        kept_lines.append(line)

    sanitized = "\n".join(kept_lines)
    sanitized = re.sub(r"[ \t]{2,}", " ", sanitized)
    return re.sub(r"\n{3,}", "\n\n", sanitized).strip()


__all__ = ["sanitize_math_solution_text"]
