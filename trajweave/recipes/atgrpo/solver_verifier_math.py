from __future__ import annotations

from dataclasses import dataclass

from trajweave.backends.local import RuleBasedMathPolicyBackend, TinyTorchPolicyBackend
from trajweave.core.specs import AgentSpec, PolicyGroupSpec, TeamSpec
from trajweave.credit.atgrpo import ATGRPOCreditAssigner
from trajweave.envs.math import SolverVerifierMathEnvironment
from trajweave.orchestration.solver_verifier import SolverVerifierOrchestra
from trajweave.recipes.doctor_mas.math_smoke import default_math_tasks
from trajweave.rollout.engine import RolloutEngine, RolloutResult


@dataclass(frozen=True)
class ATGRPOSmokeSummary:
    trajectories: int
    samples: int
    success_rate: float


def default_atgrpo_team(max_turns: int = 3) -> TeamSpec:
    return TeamSpec(
        name="atgrpo_solver_verifier_math",
        agents=(
            AgentSpec(name="solver", role="solver", policy_group="shared", trainable=True),
            AgentSpec(name="verifier", role="verifier", policy_group="shared", trainable=True),
        ),
        policy_groups=(PolicyGroupSpec(name="shared", backend="local", trainable=True),),
        orchestra="solver_verifier",
        reward="math_exact_match",
        credit="atgrpo_agent_turn_wise_grpo",
        max_turns=max_turns,
        metadata={
            "control": "fixed_protocol",
            "communication_graph": "solver_verifier_loop",
            "training_target": "solver_and_verifier",
            "credit_levels": ["agent_role", "turn"],
        },
    )


def run_atgrpo_smoke(
    *,
    backend: str = "rule",
    device: str = "cpu",
    rollouts_per_task: int = 4,
    max_turns: int = 3,
    normalize_by_std: bool = True,
    mixed_reward_enabled: bool = False,
    alpha: float = 1.0,
    verifier_local_reward: float = 1.0,
) -> tuple[ATGRPOSmokeSummary, RolloutResult]:
    if backend == "tiny-torch":
        policy_backend = TinyTorchPolicyBackend(device=device)
    elif backend == "rule":
        policy_backend = RuleBasedMathPolicyBackend()
    else:
        raise ValueError(f"Unsupported AT-GRPO smoke backend: {backend}")

    engine = RolloutEngine(
        team=default_atgrpo_team(max_turns=max_turns),
        orchestra=SolverVerifierOrchestra(),
        environment=SolverVerifierMathEnvironment(),
        policy_backend=policy_backend,
        credit_assigner=ATGRPOCreditAssigner(
            normalize_by_std=normalize_by_std,
            mixed_reward_enabled=mixed_reward_enabled,
            alpha=alpha,
            verifier_local_reward_scale=verifier_local_reward,
        ),
    )
    result = engine.run(default_math_tasks(), rollouts_per_task=rollouts_per_task)
    return (
        ATGRPOSmokeSummary(
            trajectories=len(result.trajectories),
            samples=len(result.samples),
            success_rate=result.success_rate,
        ),
        result,
    )
