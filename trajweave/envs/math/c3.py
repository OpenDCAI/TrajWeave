# SPDX-License-Identifier: Apache-2.0
"""C3-compatible math answer parsing and deterministic reward.

The parsing rules in this module are adapted from EIT-EAST-Lab/C3 commit
628185becc70732771393be28d087e88f0a4a5e8, in particular
``c3/envs/math/parsing.py`` and ``c3/envs/math/reward.py``. C3 is licensed
under Apache-2.0; see ``licenses/C3-Apache-2.0.txt`` and ``Notice.txt``.

``SolverVerifierMathEnvironment`` reuses this judge so math workflows can
consume the original string labels used by MATH, CMATH, and GSM8K-style
datasets without maintaining a second answer parser.
"""

from __future__ import annotations

import re
import json
import subprocess
import sys
import threading
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from functools import lru_cache

from trajweave.envs.math.c3_text_sanitize import sanitize_math_solution_text

_TRAILING_PUNCTUATION = " \t\r\n.。；;，,"
_MINUS_CHARS = "\u2212\u2012\u2013\u2014\u2010"
_HASH_LINE_RE = re.compile(r"(?m)^\s*####\s*(.+?)\s*$")
_ANSWER_LINE_RE = re.compile(
    r"(?im)^\s*(?:final answer|the answer is|answer|答案是|最后答案|最终答案)\s*[:：=]\s*(.+?)\s*$"
)
_ANSWER_INLINE_RE = re.compile(
    r"(?i)(?:final answer|the answer is|answer|答案是|最后答案|最终答案)\s*[:：=]\s*([^\r\n]+)"
)
_LATEX_FRACTION_RE = re.compile(
    r"\\(?:d?frac|tfrac)\s*\{\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\}"
    r"\s*\{\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\}"
)
_PAREN_FRACTION_RE = re.compile(
    r"\(\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\)\s*/\s*"
    r"\(\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\)"
)
_THOUSANDS_RE = re.compile(r"^[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?$")
_DECIMAL_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
_FRACTION_RE = re.compile(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*/\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))$")
_MIXED_NUMBER_RE = re.compile(r"^([+-]?\d+)\s+(\d+)\s*/\s*(\d+)$")
_MATH_VERIFY_CANDIDATE_RE = re.compile(r"^[0-9A-Za-z\\{}()[\].,+\-*/^=_$|<>\s]+$")


@dataclass(frozen=True)
class C3MathTask:
    task_id: str
    question: str
    answer: str


@dataclass(frozen=True)
class C3MathEnvironment:
    """Score the Actor leaf against the unmodified C3 dataset label."""

    use_math_verify: bool = True
    name: str = "c3_math"

    def initial_observation(self, task: C3MathTask) -> str:
        return task.question

    def evaluate(self, task: C3MathTask, final_answer: str) -> tuple[float, bool]:
        # Match upstream C3 judging without mutating the stored model output or label.
        predicted, prediction_method = parse_c3_math_answer(
            sanitize_math_solution_text(final_answer, strip_answer_prefix=False)
        )
        expected, _ = parse_c3_math_answer(sanitize_math_solution_text(task.answer, strip_answer_prefix=False))
        if predicted is None or expected is None:
            return 0.0, False

        success = c3_math_answers_equal(predicted, expected)
        if not success and self.use_math_verify and _eligible_for_math_verify(predicted, prediction_method):
            verified = _math_verify_equal(predicted, expected)
            success = bool(verified) if verified is not None else False
        return (1.0 if success else 0.0), success


def parse_c3_math_answer(text: str) -> tuple[str | None, str]:
    """Extract C3's final answer, preferring the last explicit answer marker."""

    if not text:
        return None, "empty"
    value = str(text)

    explicit: list[tuple[int, str, str]] = []
    for match in _HASH_LINE_RE.finditer(value):
        answer = _nonempty_answer(match.group(1))
        if answer is not None:
            explicit.append((match.start(), answer, "hash"))

    boxed = _extract_last_boxed(value)
    if boxed is not None:
        position, answer = boxed
        explicit.append((position, answer, "boxed"))

    for method, pattern in (("anchor", _ANSWER_LINE_RE), ("anchor_inline", _ANSWER_INLINE_RE)):
        for match in pattern.finditer(value):
            answer = _nonempty_answer(match.group(1))
            if answer is not None:
                explicit.append((match.start(), answer, method))

    if explicit:
        _, answer, method = max(explicit, key=lambda candidate: candidate[0])
        return answer, method

    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if not lines:
        return None, "empty_lines"
    return _nonempty_answer(lines[-1]), "last_line"


def c3_math_answers_equal(predicted: str, expected: str) -> bool:
    """Compare numeric answers exactly, then fall back to normalized text."""

    predicted_norm, predicted_fraction = _normalize_math_answer(predicted)
    expected_norm, expected_fraction = _normalize_math_answer(expected)
    if predicted_fraction is not None and expected_fraction is not None:
        return predicted_fraction == expected_fraction
    return predicted_norm == expected_norm


def _extract_last_boxed(text: str) -> tuple[int, str] | None:
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None

    depth = 1
    output: list[str] = []
    for char in text[start + len(marker) :]:
        if char == "{":
            depth += 1
            output.append(char)
        elif char == "}":
            depth -= 1
            if depth == 0:
                answer = _nonempty_answer("".join(output))
                return (start, answer) if answer is not None else None
            output.append(char)
        else:
            output.append(char)
    return None


def _nonempty_answer(value: str) -> str | None:
    answer = _strip_trailing_junk(value)
    return answer if answer else None


def _strip_trailing_junk(value: str) -> str:
    answer = (value or "").strip().rstrip(_TRAILING_PUNCTUATION).strip()
    while answer.endswith("}") and "{" not in answer:
        answer = answer[:-1].rstrip(_TRAILING_PUNCTUATION).strip()
    while answer.endswith("]") and "[" not in answer and "(" not in answer:
        answer = answer[:-1].rstrip(_TRAILING_PUNCTUATION).strip()
    while answer.endswith(")") and "(" not in answer and "[" not in answer:
        answer = answer[:-1].rstrip(_TRAILING_PUNCTUATION).strip()
    return answer


def _normalize_math_answer(answer: str) -> tuple[str, Fraction | None]:
    value = (answer or "").strip().strip("$ ").strip()
    value = value.rstrip(_TRAILING_PUNCTUATION).strip()
    for minus in _MINUS_CHARS:
        value = value.replace(minus, "-")
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = _PAREN_FRACTION_RE.sub(r"\1/\2", value)
    value = _LATEX_FRACTION_RE.sub(r"\1/\2", value)
    value = re.sub(r"^\s*-\s+", "-", value)

    option = re.fullmatch(r"\(?\s*([A-Za-z])\s*\)?", value)
    if option:
        return option.group(1).lower(), None

    numeric_value = value.replace(",", "") if _THOUSANDS_RE.fullmatch(value) else value
    fraction = _to_fraction(numeric_value)
    if fraction is not None:
        normalized = str(fraction.numerator)
        if fraction.denominator != 1:
            normalized += f"/{fraction.denominator}"
        return normalized, fraction

    normalized = re.sub(r"\s+", "", value).lower()
    return normalized, None


def _to_fraction(value: str) -> Fraction | None:
    mixed = _MIXED_NUMBER_RE.fullmatch(value)
    if mixed:
        whole, numerator, denominator = (int(part) for part in mixed.groups())
        sign = -1 if value.lstrip().startswith("-") else 1
        try:
            return sign * (abs(whole) + Fraction(numerator, denominator))
        except ZeroDivisionError:
            return None

    fraction = _FRACTION_RE.fullmatch(value)
    if fraction:
        try:
            numerator = Fraction(Decimal(fraction.group(1)))
            denominator = Fraction(Decimal(fraction.group(2)))
            return numerator / denominator
        except (InvalidOperation, ValueError, ZeroDivisionError):
            return None

    if _DECIMAL_RE.fullmatch(value):
        try:
            return Fraction(Decimal(value))
        except (InvalidOperation, ValueError):
            return None
    return None


def _eligible_for_math_verify(answer: str, method: str) -> bool:
    if method in {"hash", "boxed", "anchor", "anchor_inline"}:
        return True
    return bool(_MATH_VERIFY_CANDIDATE_RE.fullmatch(answer)) and not bool(re.search(r"\s+[A-Za-z]{2,}\s+", answer))


def _math_verify_equal(predicted_answer: str, ground_truth: str) -> bool | None:
    """Use math-verify when installed; ``None`` selects the deterministic result."""

    # Ray 异步 Actor 在非主线程调用环境；math-verify 的 signal 超时在此不可用。
    # 将符号判分移到有超时限制的独立进程，避免正确答案被静默判为错误。
    if threading.current_thread() is not threading.main_thread():
        return _thread_math_verify_equal(predicted_answer, ground_truth)
    return _math_verify_equal_inline(predicted_answer, ground_truth)


@lru_cache(maxsize=4096)
def _thread_math_verify_equal(predicted_answer: str, ground_truth: str) -> bool | None:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "trajweave.envs.math.verify_worker"],
            input=json.dumps([predicted_answer, ground_truth]),
            text=True, capture_output=True, timeout=12, check=True,
        )
        value = json.loads(result.stdout)
        return value if isinstance(value, bool) else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _math_verify_equal_inline(predicted_answer: str, ground_truth: str) -> bool | None:

    try:
        from math_verify.errors import TimeoutException
        from math_verify.metric import math_metric
        from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
    except (ImportError, ModuleNotFoundError):
        return None

    try:
        verify = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        )
        score, _ = verify(
            [rf"\boxed{{{ground_truth}}}"],
            [rf"\boxed{{{predicted_answer}}}"],
        )
        return float(score or 0.0) >= 0.999
    except TimeoutException:
        return False
    except Exception:
        return None
