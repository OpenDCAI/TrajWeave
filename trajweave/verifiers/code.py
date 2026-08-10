from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

from trajweave.verifiers.base import VerifierRequest, VerifierResult


@dataclass
class CodeVerifierAdapter:
    """Small local code verifier used by the MARTI rollout bridge.

    When a task carries PRIME-style ``inputs``/``outputs`` test cases this
    delegates to VERL's subprocess-based code checker.  Older smoke records
    only carry a textual ground truth; those use a deliberately explicit
    fallback matcher and report that mode in metadata rather than pretending
    to be sandbox execution.
    """

    name: str = "verl_prime_code"
    timeout: float = 5.0

    def verify(self, request: VerifierRequest) -> VerifierResult:
        metadata = dict(request.metadata)
        reward_model = metadata.get("reward_model") or {}
        extra_info = metadata.get("extra_info") or {}
        test_cases = reward_model.get("test_cases") or extra_info.get("test_cases")
        if test_cases:
            return self._verify_test_cases(request, test_cases)

        ground_truth = reward_model.get("ground_truth")
        if isinstance(ground_truth, str) and ground_truth.strip():
            expected = _normalize_code(ground_truth)
            candidate = _normalize_code(request.candidate)
            success = expected in candidate or candidate in expected
            return VerifierResult(
                score=1.0 if success else 0.0,
                success=success,
                terminal=success,
                feedback="ground-truth code matched" if success else "ground-truth code did not match",
                metadata={
                    "verification_mode": "ground_truth_match",
                    "failure_type": None if success else "wrong_answer",
                },
            )

        return VerifierResult(
            score=0.0,
            success=False,
            terminal=False,
            feedback="no code test cases or ground truth were provided",
            metadata={"verification_mode": "missing_tests", "failure_type": "missing_tests"},
        )

    def _verify_test_cases(self, request: VerifierRequest, test_cases: Any) -> VerifierResult:
        try:
            from verl.utils.reward_score.prime_code import compute_score

            score, details = compute_score(request.candidate, test_cases, continuous=True)
            numeric_score = float(score)
            success = numeric_score >= 1.0
            failure_type = None if success else _failure_type(details)
            return VerifierResult(
                score=numeric_score,
                success=success,
                terminal=success,
                feedback="all code tests passed" if success else "code tests failed",
                metadata={
                    "verification_mode": "prime_code_subprocess",
                    "failure_type": failure_type,
                    "test_details": details,
                },
            )
        except ModuleNotFoundError as exc:
            # Some environments omit VERL or PRIME's optional ``pyext``
            # runtime extension. Keep small call-based smoke tasks usable,
            # but label this path explicitly; it is not PRIME's sandbox.
            return self._verify_test_cases_local(request, test_cases, reason=str(exc))
        except Exception as exc:  # verifier errors are trajectory data, not worker crashes
            return self._verifier_error(exc)

    def _verify_test_cases_local(self, request: VerifierRequest, test_cases: Any, *, reason: str) -> VerifierResult:
        if not isinstance(test_cases, dict) or not test_cases.get("fn_name"):
            return self._verifier_error(
                RuntimeError("local fallback supports only call-based test_cases with fn_name"),
                mode="local_subprocess_fallback",
                reason=reason,
            )
        script = """
import json, sys
candidate = json.loads(sys.stdin.readline())
cases = json.loads(sys.stdin.readline())
try:
    namespace = {}
    exec(candidate, namespace, namespace)
    obj = namespace.get("Solution")
    target = obj() if isinstance(obj, type) else namespace
    fn = getattr(target, cases["fn_name"], None) if obj is not None else namespace.get(cases["fn_name"])
    if fn is None:
        raise AttributeError(f"function {cases['fn_name']} not found")
    results = []
    for raw_inputs, raw_output in zip(cases["inputs"], cases["outputs"]):
        args = [json.loads(line) for line in raw_inputs.split("\\n")]
        expected = json.loads(raw_output)
        try:
            actual = fn(*args)
            results.append({"ok": actual == expected, "actual": repr(actual), "expected": repr(expected)})
        except Exception as error:
            results.append({"ok": False, "error": f"{type(error).__name__}: {error}", "expected": repr(expected)})
    print(json.dumps(results))
except Exception as error:
    print(json.dumps({"error": f"{type(error).__name__}: {error}"}))
"""
        try:
            proc = subprocess.run(
                [sys.executable, "-c", script],
                input=json.dumps(_extract_executable_code(request.candidate)) + "\n" + json.dumps(test_cases) + "\n",
                text=True,
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )
            payload = (
                json.loads(proc.stdout.strip().splitlines()[-1])
                if proc.stdout.strip()
                else {"error": proc.stderr.strip()}
            )
        except subprocess.TimeoutExpired:
            return VerifierResult(
                score=0.0, success=False, terminal=False,
                feedback="local code verifier timed out",
                metadata={
                    "verification_mode": "local_subprocess_fallback",
                    "failure_type": "timeout",
                    "fallback_reason": reason,
                },
            )
        except Exception as exc:
            return self._verifier_error(exc, mode="local_subprocess_fallback", reason=reason)
        if isinstance(payload, dict) and "error" in payload:
            error_text = str(payload["error"])
            return VerifierResult(
                score=0.0, success=False, terminal=False,
                feedback=f"local code tests failed: {error_text}",
                metadata={
                    "verification_mode": "local_subprocess_fallback",
                    "failure_type": _local_failure_type(error_text),
                    "fallback_reason": reason,
                },
            )
        passed = sum(1 for item in payload if item.get("ok"))
        total = max(len(payload), 1)
        success = passed == total
        return VerifierResult(
            score=passed / total,
            success=success,
            terminal=success,
            feedback="all code tests passed" if success else f"code tests passed {passed}/{total}",
            metadata={
                "verification_mode": "local_subprocess_fallback",
                "failure_type": None if success else "wrong_answer",
                "fallback_reason": reason,
                "test_details": payload,
            },
        )

    def _verifier_error(
        self,
        exc: Exception,
        *,
        mode: str = "prime_code_subprocess",
        reason: str | None = None,
    ) -> VerifierResult:
        metadata = {"verification_mode": mode, "failure_type": "verifier_error", "verifier_error": repr(exc)}
        if reason:
            metadata["fallback_reason"] = reason
        return VerifierResult(
            score=0.0,
            success=False,
            terminal=False,
            feedback=f"code verifier error: {type(exc).__name__}: {exc}",
            metadata=metadata,
        )


@dataclass
class StandardInputCodeVerifierAdapter(CodeVerifierAdapter):
    """Verifier for LiveCodeBench-style stdin/stdout examples.

    ``test_cases`` may be a list of ``{"input": ..., "output": ...}`` rows or
    a mapping containing an ``examples`` list.  The candidate is executed in a
    subprocess with a timeout and never in the trainer process.
    """

    name: str = "livecodebench_standard_input"

    def _verify_test_cases(self, request: VerifierRequest, test_cases: Any) -> VerifierResult:
        examples = test_cases.get("examples", test_cases) if isinstance(test_cases, dict) else test_cases
        if not isinstance(examples, list) or not examples:
            return self._verifier_error(ValueError("standard-input verifier requires non-empty examples"), mode="standard_input")
        passed = 0
        details = []
        for example in examples:
            if not isinstance(example, dict):
                details.append({"ok": False, "error": "example must be a mapping"})
                continue
            raw_input = str(example.get("input", ""))
            expected = _normalize_output(str(example.get("output", "")))
            script = "import sys\n" + _extract_executable_code(request.candidate)
            try:
                proc = subprocess.run([sys.executable, "-c", script], input=raw_input, text=True,
                                      capture_output=True, timeout=self.timeout, check=False)
                actual = _normalize_output(proc.stdout)
                ok = proc.returncode == 0 and actual == expected
                detail = {"ok": ok, "actual": actual, "expected": expected, "returncode": proc.returncode}
                if proc.returncode != 0:
                    detail["stderr"] = proc.stderr[-500:]
            except subprocess.TimeoutExpired:
                ok = False
                detail = {"ok": False, "error": "timeout", "expected": expected}
            passed += int(ok)
            details.append(detail)
        score = passed / len(examples)
        success = passed == len(examples)
        return VerifierResult(score=score, success=success, terminal=success,
                              feedback=f"standard-input tests passed {passed}/{len(examples)}",
                              metadata={"verification_mode": "standard_input_subprocess", "failure_type": None if success else "wrong_answer", "test_details": details})


def _normalize_code(value: str) -> str:
    value = _extract_executable_code(value)
    return re.sub(r"\s+", " ", value).strip().lower()


def _extract_executable_code(value: str) -> str:
    python_fence = re.search(r"```(?:python|py)\s*(.*?)```", value, flags=re.IGNORECASE | re.DOTALL)
    if python_fence:
        return python_fence.group(1).strip()
    generic_fence = re.search(r"```\s*(.*?)```", value, flags=re.DOTALL)
    if generic_fence:
        return generic_fence.group(1).strip()
    return value.strip()


def _failure_type(details: Any) -> str:
    text = repr(details).lower()
    if "timeout" in text:
        return "timeout"
    if "syntax" in text or "compile" in text:
        return "compile_error"
    if "error" in text or "traceback" in text:
        return "runtime_error"
    return "wrong_answer"


def _local_failure_type(error_text: str) -> str:
    lowered = error_text.lower()
    if "syntaxerror" in lowered or "indentationerror" in lowered:
        return "compile_error"
    if "not found" in lowered or "attributeerror" in lowered:
        return "wrong_answer"
    if "timeout" in lowered:
        return "timeout"
    return "runtime_error"


def _normalize_output(value: str) -> str:
    return " ".join(value.strip().split())


def build_code_verifier(kind: str = "auto") -> CodeVerifierAdapter:
    normalized = str(kind).lower()
    if normalized in {"standard", "standard_input", "livecodebench"}:
        return StandardInputCodeVerifierAdapter()
    if normalized in {"call", "call_based", "prime_code", "auto"}:
        return CodeVerifierAdapter()
    raise ValueError(f"Unsupported code verifier kind: {kind!r}")
