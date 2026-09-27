from __future__ import annotations

from dataclasses import dataclass

from trajweave.backends.local import RuleBasedMathPolicyBackend, TinyTorchPolicyBackend
from trajweave.core.specs import AgentSpec, PolicyGroupSpec, TeamSpec
from trajweave.credit.gigpo import GiGPOCreditAssigner
from trajweave.envs.math import SolverVerifierMathEnvironment
from trajweave.orchestration.gigpo import GiGPOSolverVerifierOrchestra
from trajweave.recipes.doctor_mas.math_smoke import default_math_tasks
from trajweave.rollout.engine import RolloutEngine, RolloutResult


@dataclass(frozen=True)
class GiGPOSmokeSummary:
    trajectories: int
    samples: int
    success_rate: float


def default_gigpo_team(max_steps: int = 2) -> TeamSpec:
    return TeamSpec(
        name="gigpo_solver_verifier_math",
        agents=(
            AgentSpec(name="solver", role="solver", policy_group="shared", trainable=True),
            AgentSpec(name="verifier", role="verifier", policy_group="shared", trainable=False),
        ),
        policy_groups=(PolicyGroupSpec(name="shared", backend="local", trainable=True),),
        orchestra="gigpo_solver_verifier",
        reward="math_exact_match",
        credit="gigpo_hierarchical_grpo",
        max_turns=max_steps,
        metadata={
            "control": "fixed_protocol",
            "communication_graph": "solver_verifier_loop",
            "training_target": "solver_only",
            "credit_levels": ["episode", "step"],
        },
    )


def run_gigpo_smoke(
    *,
    backend: str = "rule",
    device: str = "cpu",
    rollouts_per_task: int = 2,
    max_steps: int = 2,
    gamma: float = 0.95,
    step_advantage_weight: float = 1.0,
    mode: str = "mean_std_norm",
    enable_similarity: bool = False,
    similarity_threshold: float = 0.95,
) -> tuple[GiGPOSmokeSummary, RolloutResult]:
    if backend == "tiny-torch":
        policy_backend = TinyTorchPolicyBackend(device=device)
    elif backend == "rule":
        policy_backend = RuleBasedMathPolicyBackend()
    else:
        raise ValueError(f"Unsupported GiGPO smoke backend: {backend}")

    engine = RolloutEngine(
        team=default_gigpo_team(max_steps=max_steps),
        orchestra=GiGPOSolverVerifierOrchestra(),
        environment=SolverVerifierMathEnvironment(),
        policy_backend=policy_backend,
        credit_assigner=GiGPOCreditAssigner(
            gamma=gamma,
            step_advantage_weight=step_advantage_weight,
            mode=mode,
            enable_similarity=enable_similarity,
            similarity_threshold=similarity_threshold,
        ),
    )
    result = engine.run(default_math_tasks(), rollouts_per_task=rollouts_per_task)
    return (
        GiGPOSmokeSummary(
            trajectories=len(result.trajectories),
            samples=len(result.samples),
            success_rate=result.success_rate,
        ),
        result,
    )
