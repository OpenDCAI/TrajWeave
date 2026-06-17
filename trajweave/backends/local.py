from __future__ import annotations

import re
from dataclasses import dataclass

from trajweave.backends.policy import PolicyRequest, PolicyResponse, StableByteTokenizer


_FINAL_RE = re.compile(r"final\s*answer\s*[:=]\s*(-?\d+)", re.IGNORECASE)


def _parse_arithmetic(text: str) -> int | None:
    match = re.search(r"(-?\d+)\s*([+\-*x])\s*(-?\d+)", text)
    if match is None:
        return None
    left = int(match.group(1))
    op = match.group(2)
    right = int(match.group(3))
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    return left * right


def extract_final_int(text: str) -> int | None:
    match = _FINAL_RE.search(text)
    if match:
        return int(match.group(1))
    numbers = re.findall(r"-?\d+", text)
    return int(numbers[-1]) if numbers else None


@dataclass
class RuleBasedMathPolicyBackend:
    tokenizer: StableByteTokenizer = StableByteTokenizer()

    def generate(self, request: PolicyRequest) -> PolicyResponse:
        if request.agent.role.lower() == "solver":
            answer = _parse_arithmetic(request.observation)
            text = f"Reasoning: compute the expression. Final answer: {answer if answer is not None else 0}"
        elif request.agent.role.lower() == "verifier":
            predicted = extract_final_int(request.team_context)
            expected = request.metadata.get("answer")
            if predicted is not None and expected is not None and int(predicted) == int(expected):
                text = "APPROVED: the solver final answer is correct."
            else:
                text = "REVISE: the solver final answer is not verified."
        else:
            text = "PASS"
        token_ids = self.tokenizer.encode(text)
        return PolicyResponse(text=text, token_ids=token_ids, logprobs=[0.0] * len(token_ids))


@dataclass
class TinyTorchPolicyBackend(RuleBasedMathPolicyBackend):
    device: str = "cpu"
    hidden_size: int = 16

    def __post_init__(self) -> None:
        try:
            import torch
        except Exception:
            self._torch = None
            self._model = None
            return
        self._torch = torch
        self._requested_device = self.device
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"
        self._model = torch.nn.Sequential(
            torch.nn.Embedding(257, self.hidden_size),
            torch.nn.Linear(self.hidden_size, 1),
        ).to(self.device)
        self._model.eval()

    def generate(self, request: PolicyRequest) -> PolicyResponse:
        response = super().generate(request)
        if self._torch is None or self._model is None:
            return response
        ids = self._torch.tensor(response.token_ids, dtype=self._torch.long, device=self.device).clamp(max=256)
        with self._torch.no_grad():
            scores = self._model(ids).squeeze(-1)
            logprobs = (-scores.abs()).detach().cpu().tolist()
        response.logprobs = [float(value) for value in logprobs]
        response.metadata["model_backend"] = "tiny-torch"
        response.metadata["device"] = self.device
        response.metadata["requested_device"] = self._requested_device
        return response
