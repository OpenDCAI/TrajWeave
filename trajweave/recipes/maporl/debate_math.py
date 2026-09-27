from __future__ import annotations

from trajweave.backends.local import RuleBasedMathPolicyBackend, TinyTorchPolicyBackend
from trajweave.core.specs import AgentSpec, PolicyGroupSpec, TeamSpec
from trajweave.credit.maporl import MAPoRLPPOScoreRuleCreditAssigner
from trajweave.envs.math import SolverVerifierMathEnvironment
from trajweave.orchestration.maporl_debate import MAPoRLDebateOrchestra
from trajweave.recipes.doctor_mas.math_smoke import default_math_tasks
from trajweave.rollout.engine import RolloutEngine, RolloutResult


def default_debate_team(
    agent_count: int = 2,
    max_turns: int = 2,
    *,
    agent_ids: tuple[str, ...] | None = None,
    model_ids: tuple[str, ...] | None = None,
    policy_separation: bool = True,
    collaboration_separation: bool = True,
    task_training: bool = False,
) -> TeamSpec:
    if agent_ids is None:
        agent_ids = tuple(f"agent_{idx}" for idx in range(agent_count))
    if model_ids is None:
        model_ids = tuple("shared" for _ in agent_ids)
    if len(agent_ids) != len(model_ids):
        raise ValueError("agent_ids and model_ids must have the same length.")
    agents = tuple(
        AgentSpec(name=agent_id, role="solver", policy_group=model_id, trainable=True)
        for agent_id, model_id in zip(agent_ids, model_ids, strict=True)
    )
    policy_groups = tuple(PolicyGroupSpec(name=model_id, backend="local") for model_id in dict.fromkeys(model_ids))
    return TeamSpec(
        name="maporl_debate_math",
        agents=agents,
        policy_groups=policy_groups,
        orchestra="maporl_debate",
        reward="exact_match",
        credit="maporl_ppo_score_rule",
        max_turns=max_turns,
        metadata={
            "control": "fixed_protocol",
            "communication_graph": "fully_connected",
            "aggregation": "consensus",
            "training_target": "train_all_agents",
            "policy_separation": policy_separation,
            "collaboration_separation": collaboration_separation,
            "task_training": task_training,
        },
    )


def run_debate_math_smoke(
    *,
    backend: str = "rule",
    device: str = "cpu",
    agent_count: int = 2,
    rollouts_per_task: int = 1,
    max_turns: int = 2,
    agent_ids: tuple[str, ...] | None = None,
    model_ids: tuple[str, ...] | None = None,
    consensus_threshold: int | None = None,
    early_stop: bool = True,
    reward_feedback: bool = False,
    criteria_for_consensus_percentage: float | None = None,
    criteria_for_consensus_reward_threshold: float = 0.7,
    rule_horizon: str = "discounted_sum",
    rule_agent_share: str = "all",
    rule_discount: float = 0.3,
    alpha: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    policy_separation: bool = True,
    collaboration_separation: bool = True,
    task_training: bool = False,
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
        team=default_debate_team(
            agent_count=agent_count,
            max_turns=max_turns,
            agent_ids=agent_ids,
            model_ids=model_ids,
            policy_separation=policy_separation,
            collaboration_separation=collaboration_separation,
            task_training=task_training,
        ),
        orchestra=MAPoRLDebateOrchestra(
            consensus_threshold=consensus_threshold,
            early_stop=early_stop,
            reward_feedback=reward_feedback,
            criteria_for_consensus_percentage=criteria_for_consensus_percentage,
            criteria_for_consensus_reward_threshold=criteria_for_consensus_reward_threshold,
        ),
        environment=SolverVerifierMathEnvironment(),
        policy_backend=policy_backend,
        credit_assigner=MAPoRLPPOScoreRuleCreditAssigner(
            rule_horizon=rule_horizon,
            rule_agent_share=rule_agent_share,
            rule_discount=rule_discount,
            alpha=alpha,
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
