from __future__ import annotations

from trajweave.backends.local import RuleBasedMathPolicyBackend, TinyTorchPolicyBackend
from trajweave.core.specs import AgentSpec, PolicyGroupSpec, TeamSpec
from trajweave.credit.agentflow import FlowGRPOPlannerOnlyCreditAssigner
from trajweave.envs.math import SolverVerifierMathEnvironment
from trajweave.orchestration.agentflow import AgentFlowPlannerToolOrchestra
from trajweave.recipes.doctor_mas.math_smoke import default_math_tasks
from trajweave.rollout.engine import RolloutEngine, RolloutResult


def default_agentflow_team(max_steps: int = 3) -> TeamSpec:
    return TeamSpec(
        name="agentflow_planner_tool",
        agents=(
            AgentSpec(
                name="planner",
                role="planner",
                policy_group="planner",
                trainable=True,
                tools=("base_generator", "python_stub"),
            ),
            AgentSpec(name="executor", role="executor", policy_group="frozen", trainable=False),
            AgentSpec(name="verifier", role="verifier", policy_group="frozen", trainable=False),
        ),
        policy_groups=(
            PolicyGroupSpec(name="planner", backend="local", trainable=True),
            PolicyGroupSpec(name="frozen", backend="local", trainable=False),
        ),
        orchestra="agentflow_planner_tool",
        reward="math_exact_match",
        credit="agentflow_planner_only_grpo",
        max_turns=max_steps,
        metadata={
            "control": "fixed_protocol",
            "communication_graph": "memory_blackboard",
            "aggregation": "verifier_stop_then_final_answer",
            "training_target": "planner_only",
        },
    )


def run_planner_tool_smoke(
    *,
    backend: str = "rule",
    device: str = "cpu",
    rollouts_per_task: int = 1,
    max_steps: int = 3,
) -> tuple[object, RolloutResult]:
    if backend == "tiny-torch":
        policy_backend = TinyTorchPolicyBackend(device=device)
    elif backend == "rule":
        policy_backend = RuleBasedMathPolicyBackend()
    else:
        raise ValueError(f"Unsupported AgentFlow smoke backend: {backend}")

    engine = RolloutEngine(
        team=default_agentflow_team(max_steps=max_steps),
        orchestra=AgentFlowPlannerToolOrchestra(),
        environment=SolverVerifierMathEnvironment(),
        policy_backend=policy_backend,
        credit_assigner=FlowGRPOPlannerOnlyCreditAssigner(),
    )
    result = engine.run(default_math_tasks(), rollouts_per_task=rollouts_per_task)

    class Summary:
        trajectories = len(result.trajectories)
        samples = len(result.samples)
        success_rate = result.success_rate
        dataproto_rows = None
        dataproto_status = "skipped"

    return Summary(), result
