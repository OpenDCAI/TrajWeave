from __future__ import annotations

from trajweave.backends.local import RuleBasedMathPolicyBackend, TinyTorchPolicyBackend
from trajweave.core.specs import AgentSpec, PolicyGroupSpec, TeamSpec
from trajweave.credit.maporl import MAPoRLScoreBonusCreditAssigner
from trajweave.envs.math import SolverVerifierMathEnvironment
from trajweave.orchestration.maporl_debate import MAPoRLDebateOrchestra
from trajweave.recipes.doctor_mas.math_smoke import default_math_tasks
from trajweave.rollout.engine import RolloutEngine, RolloutResult


def default_debate_team(agent_count: int = 2, max_turns: int = 2) -> TeamSpec:
    agents = tuple(
        AgentSpec(name=f"agent_{idx}", role="solver", policy_group="shared", trainable=True)
        for idx in range(agent_count)
    )
    return TeamSpec(
        name="maporl_debate_math",
        agents=agents,
        policy_groups=(PolicyGroupSpec(name="shared", backend="local"),),
        orchestra="maporl_debate",
        reward="exact_match",
        credit="maporl_score_bonus",
        max_turns=max_turns,
        metadata={
            "control": "fixed_protocol",
            "communication_graph": "fully_connected",
            "aggregation": "consensus",
            "training_target": "train_all_shared_policy",
        },
    )


def run_debate_math_smoke(
    *,
    backend: str = "rule",
    device: str = "cpu",
    agent_count: int = 2,
    rollouts_per_task: int = 1,
    max_turns: int = 2,
    consensus_threshold: int | None = None,
    early_stop: bool = True,
    correct_turn_bonus: float = 0.25,
    consensus_bonus: float = 0.25,
    baseline_scope: str = "policy_group",
) -> tuple[object, RolloutResult]:
    if consensus_threshold is None:
        consensus_threshold = max(2, agent_count)
    if backend == "tiny-torch":
        policy_backend = TinyTorchPolicyBackend(device=device)
    elif backend == "rule":
        policy_backend = RuleBasedMathPolicyBackend()
    else:
        raise ValueError(f"Unsupported MAPoRL smoke backend: {backend}")

    engine = RolloutEngine(
        team=default_debate_team(agent_count=agent_count, max_turns=max_turns),
        orchestra=MAPoRLDebateOrchestra(consensus_threshold=consensus_threshold, early_stop=early_stop),
        environment=SolverVerifierMathEnvironment(),
        policy_backend=policy_backend,
        credit_assigner=MAPoRLScoreBonusCreditAssigner(
            correct_turn_bonus=correct_turn_bonus,
            consensus_bonus=consensus_bonus,
            baseline_scope=baseline_scope,
        ),
    )
    result = engine.run(default_math_tasks(), rollouts_per_task=rollouts_per_task)

    class Summary:
        trajectories = len(result.trajectories)
        samples = len(result.samples)
        success_rate = result.success_rate
        dataproto_rows = None
        dataproto_status = "skipped"

    return Summary(), result
