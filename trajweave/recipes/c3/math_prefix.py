from __future__ import annotations

from dataclasses import dataclass

from trajweave.backends.local import RuleBasedMathPolicyBackend, TinyTorchPolicyBackend
from trajweave.core.specs import AgentSpec, PolicyGroupSpec, TeamSpec
from trajweave.credit.c3 import C3CreditAssigner
from trajweave.envs.math import C3MathEnvironment, C3MathTask
from trajweave.orchestration.c3 import C3PrefixTreeOrchestra
from trajweave.recipes.doctor_mas.math_smoke import default_math_tasks
from trajweave.rollout.engine import RolloutEngine, RolloutResult


@dataclass(frozen=True)
class C3SmokeSummary:
    trajectories: int
    samples: int
    success_rate: float


def default_c3_team(*, model_ids: tuple[str, str] = ("reasoner", "actor")) -> TeamSpec:
    if len(model_ids) != 2 or len(set(model_ids)) != 2 or any(not str(value).strip() for value in model_ids):
        raise ValueError("C3 requires two distinct non-empty model_ids for Reasoner and Actor.")
    return TeamSpec(
        name="c3_reasoner_actor_math",
        agents=(
            AgentSpec(name="reasoner", role="reasoner", policy_group=model_ids[0], trainable=True),
            AgentSpec(name="actor", role="actor", policy_group=model_ids[1], trainable=True),
        ),
        policy_groups=(
            PolicyGroupSpec(name=model_ids[0], backend="local", trainable=True),
            PolicyGroupSpec(name=model_ids[1], backend="local", trainable=True),
        ),
        orchestra="c3_nested_prefix_tree",
        reward="math_exact_match",
        credit="c3_contextual_counterfactual",
        max_turns=2,
        metadata={
            "control": "fixed_context_replay",
            "communication_graph": "reasoner_to_actor",
            "training_target": "reasoner_and_actor",
            "credit_levels": ["role_prefix", "sibling_counterfactual"],
        },
    )


def run_c3_smoke(
    *,
    backend: str = "rule",
    device: str = "cpu",
    fanout: tuple[int, int] = (2, 2),
    variant: str = "reward_only",
    baseline_mode: str = "loo",
    value_assisted_alpha: float = 1.0,
    normalize: bool = True,
) -> tuple[C3SmokeSummary, RolloutResult]:
    if variant != "reward_only":
        raise ValueError("C3 smoke mode supports reward_only; value variants require a configured VERL Q critic.")
    if backend == "tiny-torch":
        policy_backend = TinyTorchPolicyBackend(device=device)
    elif backend == "rule":
        policy_backend = RuleBasedMathPolicyBackend()
    else:
        raise ValueError(f"Unsupported C3 smoke backend: {backend}")
    team = default_c3_team()
    engine = RolloutEngine(
        team=team,
        orchestra=C3PrefixTreeOrchestra(fanout=fanout),
        environment=C3MathEnvironment(),
        policy_backend=policy_backend,
        credit_assigner=C3CreditAssigner(
            variant=variant,
            baseline_mode=baseline_mode,
            value_assisted_alpha=value_assisted_alpha,
            normalize=normalize,
        ),
    )
    tasks = [
        C3MathTask(task_id=task.task_id, question=task.question, answer=str(task.answer))
        for task in default_math_tasks()
    ]
    result = engine.run(tasks, rollouts_per_task=1)
    return (
        C3SmokeSummary(
            trajectories=len(result.trajectories),
            samples=len(result.samples),
            success_rate=result.success_rate,
        ),
        result,
    )
