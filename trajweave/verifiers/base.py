from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass(frozen=True)
class VerifierRequest:
    task_id: str
    prompt: str
    candidate: str
    node_id: int
    parent_idx: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerifierResult:
    score: float
    success: bool = False
    terminal: bool = False
    feedback: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class VerifierAdapter(Protocol):
    name: str

    def verify(self, request: VerifierRequest) -> VerifierResult: ...


@dataclass
class CallableVerifierAdapter:
    verifier: Callable[[VerifierRequest], VerifierResult | float | int | bool | dict[str, Any]]
    name: str = "callable_verifier"
    success_threshold: float = 1.0

    def verify(self, request: VerifierRequest) -> VerifierResult:
        value = self.verifier(request)
        if isinstance(value, VerifierResult):
            return value
        if isinstance(value, dict):
            score = float(value.get("score", value.get("reward", 0.0)))
            success = bool(value.get("success", score >= self.success_threshold))
            return VerifierResult(
                score=score,
                success=success,
                terminal=bool(value.get("terminal", success)),
                feedback=str(value.get("feedback", "")),
                metadata={
                    key: item
                    for key, item in value.items()
                    if key not in {"score", "reward", "success", "terminal", "feedback"}
                },
            )
        score = float(value)
        success = score >= self.success_threshold
        return VerifierResult(score=score, success=success, terminal=success)
