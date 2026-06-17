from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any

from trajweave.backends.local import extract_final_int
from trajweave.backends.policy import PolicyRequest, PolicyResponse, StableByteTokenizer
from trajweave.core.trajectory import TrainingSample


_EXPR_RE = re.compile(r"(-?\d+)\s*([+\-*x])\s*(-?\d+)")


def _parse_expression(text: str) -> tuple[int, str, int] | None:
    match = _EXPR_RE.search(text)
    if match is None:
        return None
    return int(match.group(1)), match.group(2), int(match.group(3))


def _op_answer(left: int, op: str, right: int) -> int:
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    return left * right


def _op_features(op: str) -> list[float]:
    return [1.0 if op == "+" else 0.0, 1.0 if op == "-" else 0.0, 1.0 if op in ("*", "x") else 0.0]


@dataclass
class TrainableTinyMathPolicyBackend:
    device: str = "cpu"
    hidden_size: int = 48
    num_solver_candidates: int = 3
    temperature: float = 1.0
    seed: int = 1
    sample_actions: bool = True
    train_verifier: bool = False
    tokenizer: StableByteTokenizer = StableByteTokenizer()
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        import torch

        self._torch = torch
        self._requested_device = self.device
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"
        torch.manual_seed(self.seed)
        self._rng = random.Random(self.seed)
        self.solver = torch.nn.Sequential(
            torch.nn.Linear(7, self.hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(self.hidden_size, 1),
        ).to(self.device)
        self.verifier = torch.nn.Sequential(
            torch.nn.Linear(7, self.hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(self.hidden_size, 2),
        ).to(self.device)

    def parameters(self):
        params = list(self.solver.parameters())
        if self.train_verifier:
            params.extend(self.verifier.parameters())
        return params

    def train(self) -> None:
        self.sample_actions = True
        self.solver.train()
        self.verifier.train()

    def eval(self) -> None:
        self.sample_actions = False
        self.solver.eval()
        self.verifier.eval()

    def generate(self, request: PolicyRequest) -> PolicyResponse:
        role = request.agent.role.lower()
        if role == "solver":
            return self._generate_solver(request)
        if role == "verifier":
            return self._generate_verifier(request)
        text = "PASS"
        token_ids = self.tokenizer.encode(text)
        return PolicyResponse(text=text, token_ids=token_ids, logprobs=[0.0] * len(token_ids))

    def policy_loss(self, samples: list[TrainingSample], entropy_coef: float = 0.01):
        losses = []
        entropies = []
        role_counts: dict[str, int] = {}
        role_adv_sum: dict[str, float] = {}
        for sample in samples:
            if sample.advantage is None:
                continue
            metadata = sample.metadata
            if metadata.get("policy_backend") != "trainable_tiny_math":
                continue
            role = metadata["policy_role"]
            action_index = int(metadata["action_index"])
            advantage = float(sample.advantage)
            logits = self._logits_from_metadata(metadata)
            distribution = self._torch.distributions.Categorical(logits=logits / self.temperature)
            log_prob = distribution.log_prob(self._torch.tensor(action_index, device=self.device))
            entropy = distribution.entropy()
            losses.append(-(log_prob * advantage) - entropy_coef * entropy)
            entropies.append(float(entropy.detach().cpu()))
            role_counts[role] = role_counts.get(role, 0) + 1
            role_adv_sum[role] = role_adv_sum.get(role, 0.0) + advantage

        if not losses:
            zero = self._torch.zeros((), device=self.device, requires_grad=True)
            return zero, {"trainable_samples": 0, "mean_entropy": 0.0}
        loss = self._torch.stack(losses).mean()
        metrics: dict[str, Any] = {
            "trainable_samples": len(losses),
            "mean_entropy": sum(entropies) / len(entropies),
        }
        for role, count in role_counts.items():
            metrics[f"{role}_samples"] = count
            metrics[f"{role}_mean_advantage"] = role_adv_sum[role] / count
        return loss, metrics

    def _generate_solver(self, request: PolicyRequest) -> PolicyResponse:
        parsed = _parse_expression(request.observation)
        if parsed is None:
            answer = 0
            features = [[0.0] * 7]
            action_index = 0
        else:
            left, op, right = parsed
            answer = _op_answer(left, op, right)
            candidates = self._solver_candidates(answer)
            features = [self._solver_features(left, op, right, candidate) for candidate in candidates]
            logits = self._solver_logits(features)
            action_index = self._select_action(logits)
            answer = candidates[action_index]
        text = f"Reasoning: selected a candidate answer. Final answer: {answer}"
        token_ids = self.tokenizer.encode(text)
        logprob = self._sample_logprob(features, action_index, "solver")
        return PolicyResponse(
            text=text,
            token_ids=token_ids,
            logprobs=[logprob] * len(token_ids),
            metadata={
                "policy_backend": "trainable_tiny_math",
                "policy_role": "solver",
                "features": features,
                "action_index": action_index,
                "sample_logprob": logprob,
                "device": self.device,
                "requested_device": self._requested_device,
            },
        )

    def _generate_verifier(self, request: PolicyRequest) -> PolicyResponse:
        parsed = _parse_expression(request.observation)
        predicted = extract_final_int(request.team_context)
        if not self.train_verifier:
            expected = _op_answer(*parsed) if parsed is not None else None
            approved = predicted is not None and expected is not None and predicted == expected
            text = "APPROVED: rule verifier accepted the answer." if approved else "REVISE: rule verifier rejected the answer."
            token_ids = self.tokenizer.encode(text)
            return PolicyResponse(
                text=text,
                token_ids=token_ids,
                logprobs=[0.0] * len(token_ids),
                metadata={
                    "policy_backend": "rule_verifier",
                    "policy_role": "verifier",
                    "approved": approved,
                    "device": self.device,
                    "requested_device": self._requested_device,
                },
            )
        if parsed is None or predicted is None:
            features = [0.0] * 7
        else:
            left, op, right = parsed
            features = self._verifier_features(left, op, right, predicted)
        logits = self._verifier_logits(features)
        action_index = self._select_action(logits)
        text = "APPROVED: continue with this answer." if action_index == 0 else "REVISE: ask solver to try again."
        token_ids = self.tokenizer.encode(text)
        logprob = self._sample_logprob(features, action_index, "verifier")
        return PolicyResponse(
            text=text,
            token_ids=token_ids,
            logprobs=[logprob] * len(token_ids),
            metadata={
                "policy_backend": "trainable_tiny_math",
                "policy_role": "verifier",
                "features": features,
                "action_index": action_index,
                "sample_logprob": logprob,
                "device": self.device,
                "requested_device": self._requested_device,
            },
        )

    def _solver_candidates(self, answer: int) -> list[int]:
        offsets = [0]
        distance = 1
        while len(offsets) < self.num_solver_candidates:
            offsets.extend([distance, -distance])
            distance += 1
        candidates = [answer + offset for offset in offsets[: self.num_solver_candidates]]
        self._rng.shuffle(candidates)
        return candidates

    def _solver_features(self, left: int, op: str, right: int, candidate: int) -> list[float]:
        expression_value = _op_answer(left, op, right)
        return [left / 10.0, right / 10.0, *_op_features(op), candidate / 30.0, (candidate - expression_value) / 10.0]

    def _verifier_features(self, left: int, op: str, right: int, predicted: int) -> list[float]:
        return [left / 10.0, right / 10.0, *_op_features(op), predicted / 30.0, 1.0]

    def _solver_logits(self, features: list[list[float]]):
        tensor = self._torch.tensor(features, dtype=self._torch.float32, device=self.device)
        return self.solver(tensor).squeeze(-1)

    def _verifier_logits(self, features: list[float]):
        tensor = self._torch.tensor(features, dtype=self._torch.float32, device=self.device).unsqueeze(0)
        return self.verifier(tensor).squeeze(0)

    def _logits_from_metadata(self, metadata: dict):
        role = metadata["policy_role"]
        features = metadata["features"]
        if role == "solver":
            return self._solver_logits(features)
        return self._verifier_logits(features)

    def _select_action(self, logits) -> int:
        if not self.sample_actions:
            return int(self._torch.argmax(logits).item())
        distribution = self._torch.distributions.Categorical(logits=logits / self.temperature)
        return int(distribution.sample().item())

    def _sample_logprob(self, features: list, action_index: int, role: str) -> float:
        with self._torch.no_grad():
            logits = self._solver_logits(features) if role == "solver" else self._verifier_logits(features)
            distribution = self._torch.distributions.Categorical(logits=logits / self.temperature)
            value = distribution.log_prob(self._torch.tensor(action_index, device=self.device))
        return float(value.detach().cpu())
