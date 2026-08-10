"""Standalone test-time search evaluation for MARTI-MARS²."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal

from trajweave.backends.policy import PolicyBackend, PolicyRequest
from trajweave.core import TeamSpec
from trajweave.orchestration.tree_search import TreeSearchProtocol
from trajweave.verifiers import VerifierAdapter, VerifierRequest, VerifierResult


@dataclass(frozen=True)
class SearchEvalCase:
    task_id: str
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchBudget:
    greedy_nodes: int = 1
    best_of_n_nodes: int = 8
    mcts_nodes: int = 8
    initial_candidates: int = 2
    refinement_concurrency: int = 2


@dataclass
class SearchEvalResult:
    strategy: str
    task_ids: list[str]
    passed_task_ids: list[str]
    any_pass: int
    pass_at_1: float
    final_pass: float
    tokens: int
    latency_seconds: float
    refinement_recovery: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy, "task_count": len(self.task_ids),
            "passed_task_ids": list(self.passed_task_ids), "any_pass": self.any_pass,
            "pass_at_1": self.pass_at_1, "mcts_final_pass": self.final_pass,
            "tokens": self.tokens, "latency_seconds": self.latency_seconds,
            "refinement_recovery": list(self.refinement_recovery),
        }


class StandalonePolicyGroupEndpoints:
    """Lifecycle contract for evaluation-only native vLLM endpoints."""

    def __init__(self, model_paths: dict[str, str], *, endpoint_factory: Callable[[str, str], Any] | None = None):
        if not model_paths or any(not str(key) for key in model_paths):
            raise ValueError("standalone eval requires explicit policy-group model paths")
        missing = [group for group, path in model_paths.items() if not path]
        if missing:
            raise ValueError(f"Missing model path for policy groups: {missing}")
        self.model_paths = {str(group): str(path) for group, path in model_paths.items()}
        self.endpoint_factory = endpoint_factory
        self.endpoints: dict[str, Any] = {}

    def start(self) -> dict[str, Any]:
        if self.endpoint_factory is None:
            raise RuntimeError("No endpoint_factory configured; refusing to enter optimizer/training mode")
        self.endpoints = {group: self.endpoint_factory(group, path) for group, path in self.model_paths.items()}
        return {"groups": list(self.endpoints), "mode": "standalone_native_vllm"}

    def stop(self) -> None:
        for endpoint in self.endpoints.values():
            close = getattr(endpoint, "close", None) or getattr(endpoint, "stop", None)
            if close:
                close()
        self.endpoints.clear()


class VerifierBestSelector:
    name = "verifier-best"

    def select(self, candidates: Iterable[tuple[str, VerifierResult]]) -> tuple[str, VerifierResult]:
        rows = list(candidates)
        if not rows:
            raise ValueError("cannot select from empty candidate set")
        return max(enumerate(rows), key=lambda item: (item[1][1].score, -item[0]))[1]


class RewardModelSelector:
    name = "reward-model"

    def __init__(self, selector: Callable[[list[tuple[str, VerifierResult]]], tuple[str, VerifierResult]] | None = None):
        if selector is None:
            raise RuntimeError("Reward Model selector requested but no model is configured")
        self.selector = selector

    def select(self, candidates):
        return self.selector(list(candidates))


def evaluate_search(
    cases: Iterable[SearchEvalCase], *, strategy: Literal["greedy", "best_of_n", "mcts"],
    policy_backend: PolicyBackend, verifier: VerifierAdapter, team: TeamSpec,
    nodes: int = 8, initial_candidates: int = 2, refinement_concurrency: int = 2,
    selector: VerifierBestSelector | RewardModelSelector | None = None,
) -> SearchEvalResult:
    rows = list(cases)
    if nodes <= 0:
        raise ValueError("nodes must be positive")
    selector = selector or VerifierBestSelector()
    passed: list[str] = []
    recovered: list[str] = []
    token_count = 0
    started = time.perf_counter()
    for case in rows:
        candidates: list[tuple[str, VerifierResult]] = []
        if strategy == "greedy":
            response = policy_backend.generate(PolicyRequest(agent=team.agents[0], task_id=case.task_id,
                observation=case.prompt, prompt=case.prompt, metadata={"temperature": 0.0, "node_id": 0}))
            result = verifier.verify(VerifierRequest(task_id=case.task_id, prompt=case.prompt, candidate=response.text, node_id=0, metadata=case.metadata))
            candidates.append((response.text, result)); token_count += len(response.token_ids)
        else:
            protocol = TreeSearchProtocol(max_num_nodes=nodes, initial_candidates=(nodes if strategy == "best_of_n" else initial_candidates),
                refinement_concurrency=refinement_concurrency, min_num_nodes=min(initial_candidates, nodes))
            trajectory = protocol.run(tree_id=case.task_id, prompt_id=case.task_id, task=case, team=team,
                                      observation=case.prompt, policy_backend=policy_backend, verifier=verifier)
            candidates = [(node.action_text, VerifierResult(score=node.effective_reward,
                success=bool(node.metadata.get("verifier_success")), terminal=node.is_terminal,
                feedback=str(node.metadata.get("verifier_feedback", "")), metadata=node.metadata)) for node in trajectory.nodes]
            token_count += sum(len(node.action_token_ids) for node in trajectory.nodes)
            root_success = any(result.success for node, (_, result) in zip(trajectory.nodes, candidates, strict=True) if node.parent_idx is None)
            refinement_success = any(result.success for node, (_, result) in zip(trajectory.nodes, candidates, strict=True) if node.parent_idx is not None)
            if refinement_success and not root_success:
                recovered.append(case.task_id)
        _chosen_text, chosen = selector.select(candidates)
        if chosen.success:
            passed.append(case.task_id)
    elapsed = time.perf_counter() - started
    count = max(1, len(rows))
    return SearchEvalResult(strategy=strategy, task_ids=[case.task_id for case in rows], passed_task_ids=passed,
        any_pass=len(passed), pass_at_1=len(passed) / count, final_pass=len(passed) / count,
        tokens=token_count, latency_seconds=elapsed, refinement_recovery=recovered)


def run_livecodebench_subset(*, cases: Iterable[SearchEvalCase], policy_backend: PolicyBackend,
                             verifier: VerifierAdapter, team: TeamSpec, nodes: int = 8) -> dict[str, Any]:
    """Run the fixed-budget Greedy/Best-of-N/MCTS comparison."""
    fixed_cases = list(cases)
    return {strategy: evaluate_search(fixed_cases, strategy=strategy, policy_backend=policy_backend,
        verifier=verifier, team=team, nodes=nodes, initial_candidates=2).as_dict()
        for strategy in ("greedy", "best_of_n", "mcts")}


__all__ = ["SearchEvalCase", "SearchBudget", "SearchEvalResult", "StandalonePolicyGroupEndpoints", "VerifierBestSelector", "RewardModelSelector", "evaluate_search", "run_livecodebench_subset"]
