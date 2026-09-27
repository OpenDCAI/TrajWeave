from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from trajweave.verifiers import CodeVerifierAdapter, VerifierAdapter, VerifierRequest, VerifierResult


@dataclass(frozen=True)
class CodeTask:
    task_id: str
    prompt: str
    reward_model: dict[str, Any] = field(default_factory=dict)
    extra_info: dict[str, Any] = field(default_factory=dict)
    node_rewards: tuple[float, ...] | None = None


class CodeExecutionEnvironment:
    """Code-task observation and verifier boundary shared by tree-search recipes."""

    name = "code_execution"

    def __init__(self, verifier: VerifierAdapter | None = None) -> None:
        self.verifier = verifier or CodeVerifierAdapter()

    def initial_observation(self, task: CodeTask) -> str:
        return task.prompt

    def verify_candidate(
        self,
        task: CodeTask,
        candidate: str,
        *,
        node_id: int,
        parent_idx: int | None = None,
    ) -> VerifierResult:
        if task.node_rewards is not None:
            if not 0 <= node_id < len(task.node_rewards):
                raise IndexError(f"CodeTask {task.task_id!r} has no reward for node {node_id}.")
            score = float(task.node_rewards[node_id])
            return VerifierResult(
                score=score,
                success=score > 0.0,
                terminal=False,
                feedback="passes deterministic tests" if score > 0.0 else "fails deterministic tests",
                metadata={"verifier_name": "deterministic_code_verifier"},
            )
        return self.verifier.verify(
            VerifierRequest(
                task_id=task.task_id,
                prompt=task.prompt,
                candidate=candidate,
                node_id=node_id,
                parent_idx=parent_idx,
                metadata={"reward_model": task.reward_model, "extra_info": task.extra_info},
            )
        )

    def evaluate(self, task: CodeTask, final_answer: str) -> tuple[float, bool]:
        result = self.verify_candidate(task, final_answer, node_id=0)
        return result.score, result.success

    @staticmethod
    def from_record(record: dict[str, Any]) -> CodeTask:
        raw_prompt = record.get("raw_prompt", record.get("prompt", record.get("uid", "")))
        if isinstance(raw_prompt, list):
            prompt = "\n".join(
                str(item.get("content", "")) if isinstance(item, dict) else str(item) for item in raw_prompt
            )
        else:
            prompt = str(raw_prompt)
        task_id = str(record.get("uid", record.get("index", "code-task")))
        reward_model = record.get("reward_model") or {}
        extra_info = record.get("extra_info") or {}
        return CodeTask(
            task_id=task_id,
            prompt=prompt,
            reward_model=dict(reward_model) if isinstance(reward_model, dict) else {},
            extra_info=dict(extra_info) if isinstance(extra_info, dict) else {},
        )
