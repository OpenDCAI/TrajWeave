from __future__ import annotations

import argparse
from dataclasses import dataclass

from trajweave.backends.local import RuleBasedMathPolicyBackend, TinyTorchPolicyBackend
from trajweave.core.specs import AgentSpec, PolicyGroupSpec, TeamSpec
from trajweave.credit.doctor_mas import DoctorMASCreditAssigner
from trajweave.envs.math import MathTask, SolverVerifierMathEnvironment
from trajweave.orchestration.solver_verifier import SolverVerifierOrchestra
from trajweave.rollout.engine import RolloutEngine, RolloutResult


@dataclass
class SmokeSummary:
    trajectories: int
    samples: int
    success_rate: float
    dataproto_rows: int | None
    dataproto_status: str


def default_team(max_turns: int = 2) -> TeamSpec:
    return TeamSpec(
        name="doctor_mas_solver_verifier_math",
        agents=(
            AgentSpec(name="solver", role="solver", policy_group="shared_qwen_tiny"),
            AgentSpec(name="verifier", role="verifier", policy_group="shared_qwen_tiny"),
        ),
        policy_groups=(PolicyGroupSpec(name="shared_qwen_tiny", backend="local", trainable=True),),
        orchestra="solver_verifier",
        reward="math_exact_match",
        credit="doctor_mas_agent_wise_grpo",
        max_turns=max_turns,
    )


def default_math_tasks() -> list[MathTask]:
    return [
        MathTask(task_id="math_add_1", question="What is 1 + 1?", answer=2),
        MathTask(task_id="math_mul_1", question="What is 2 * 3?", answer=6),
    ]


def build_doctor_mas_math_engine(backend: str = "rule", device: str = "cpu", max_turns: int = 2) -> RolloutEngine:
    if backend == "tiny-torch":
        policy_backend = TinyTorchPolicyBackend(device=device)
    elif backend == "rule":
        policy_backend = RuleBasedMathPolicyBackend()
    else:
        raise ValueError(f"Unknown backend: {backend}")
    return RolloutEngine(
        team=default_team(max_turns=max_turns),
        orchestra=SolverVerifierOrchestra(),
        environment=SolverVerifierMathEnvironment(),
        policy_backend=policy_backend,
        credit_assigner=DoctorMASCreditAssigner(),
    )


def run_smoke(backend: str = "rule", device: str = "cpu", rollouts_per_task: int = 2, max_turns: int = 2) -> tuple[SmokeSummary, RolloutResult]:
    engine = build_doctor_mas_math_engine(backend=backend, device=device, max_turns=max_turns)
    result = engine.run(default_math_tasks(), rollouts_per_task=rollouts_per_task)
    dataproto_rows: int | None = None
    dataproto_status = "skipped"
    try:
        from trajweave.backends.verl import VerlDataProtoAdapter

        dataproto = VerlDataProtoAdapter().build(result.samples)
        dataproto_rows = len(dataproto)
        dataproto_status = "ok"
    except ModuleNotFoundError as exc:
        dataproto_status = f"unavailable: {exc.name}"
    summary = SmokeSummary(
        trajectories=len(result.trajectories),
        samples=len(result.samples),
        success_rate=result.success_rate,
        dataproto_rows=dataproto_rows,
        dataproto_status=dataproto_status,
    )
    return summary, result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the minimal TrajWeave DrMAS solver-verifier math smoke.")
    parser.add_argument("--backend", choices=["rule", "tiny-torch"], default="tiny-torch")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rollouts-per-task", type=int, default=2)
    parser.add_argument("--max-turns", type=int, default=2)
    args = parser.parse_args()
    summary, result = run_smoke(
        backend=args.backend,
        device=args.device,
        rollouts_per_task=args.rollouts_per_task,
        max_turns=args.max_turns,
    )
    print(f"trajectories={summary.trajectories}")
    print(f"samples={summary.samples}")
    print(f"success_rate={summary.success_rate:.3f}")
    print(f"dataproto_status={summary.dataproto_status}")
    if summary.dataproto_rows is not None:
        print(f"dataproto_rows={summary.dataproto_rows}")
    for sample in result.samples[:4]:
        print(
            "sample",
            sample.agent_name,
            f"reward={sample.reward:.3f}",
            f"advantage={sample.advantage:.3f}" if sample.advantage is not None else "advantage=None",
            f"group={sample.metadata.get('advantage_group')}",
        )


if __name__ == "__main__":
    main()
