from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from trajweave.backends.local import RuleBasedMathPolicyBackend, TinyTorchPolicyBackend
from trajweave.core.specs import AgentSpec, PolicyGroupSpec, TeamSpec
from trajweave.credit.marft import MARFTCreditAssigner, MARFTStepRewardFn
from trajweave.envs.math import SolverVerifierMathEnvironment
from trajweave.orchestration.marft import MARFTWorkflowGraph, MARFTWorkflowOrchestra
from trajweave.recipes.doctor_mas.math_smoke import default_math_tasks
from trajweave.rollout.engine import RolloutEngine, RolloutResult

DEFAULT_ROLE_PROMPTS = {
    "planner": (
        "You are a math problem planner. Break the problem into clear numbered steps. "
        "Do not solve it; provide only the plan."
    ),
    "solver": (
        "You are a math solver. Follow the shared plan, solve the problem step by step, "
        "and finish with 'Final answer: <number>'."
    ),
    "reflector": (
        "You are a math solution reviewer. Identify errors or gaps in the shared solution, "
        "or state briefly that it is correct."
    ),
    "verifier": (
        "You are a math verifier. Correct the shared solution if needed and finish with 'Final answer: <number>'."
    ),
}


@dataclass(frozen=True)
class MARFTSmokeSummary:
    trajectories: int
    samples: int
    success_rate: float
    agent_samples: dict[str, int]
    dataproto_rows: int | None
    dataproto_status: str


def default_marft_team(
    *,
    role_names: tuple[str, ...] = ("planner", "solver"),
    model_ids: tuple[str, ...] | None = None,
    role_configs: dict[str, dict[str, Any]] | None = None,
) -> TeamSpec:
    roles = tuple(str(role).strip() for role in role_names)
    if not roles or len(roles) != len(set(roles)):
        raise ValueError("MARFT role_names must contain unique non-empty role names.")
    if model_ids is None:
        models = ("shared",) * len(roles)
    else:
        models = tuple(str(model_id).strip() for model_id in model_ids)
        if len(models) != len(roles) or any(not model_id for model_id in models):
            raise ValueError("MARFT model_ids must align with role_names and be non-empty.")
    configs = role_configs or {role: {"system_prompt": DEFAULT_ROLE_PROMPTS.get(role, role)} for role in roles}
    missing = sorted(set(roles) - set(configs))
    if missing:
        raise ValueError(f"MARFT role_configs is missing roles: {missing}.")
    agents = tuple(
        AgentSpec(
            name=role,
            role=role,
            policy_group=model_id,
            prompt_template=str(configs[role].get("system_prompt", "")),
            generation_config={
                key: configs[role][key]
                for key in ("max_new_tokens", "temperature", "top_p", "lora_name")
                if configs[role].get(key) is not None
            },
        )
        for role, model_id in zip(roles, models, strict=True)
    )
    groups = tuple(PolicyGroupSpec(name=model_id, backend="local") for model_id in dict.fromkeys(models))
    return TeamSpec(
        name="marft_math_workflow",
        agents=agents,
        policy_groups=groups,
        orchestra="marft_static_dag",
        reward="math_exact_match",
        credit="marft_ctde",
        max_turns=len(roles),
        metadata={
            "training_paradigm": "centralized_training_decentralized_execution",
            "shared_policy": len(set(models)) == 1,
        },
    )


def build_marft_engine(
    *,
    backend: str = "rule",
    device: str = "cpu",
    role_names: tuple[str, ...] = ("planner", "solver"),
    model_ids: tuple[str, ...] | None = None,
    role_configs: dict[str, dict[str, Any]] | None = None,
    graph: MARFTWorkflowGraph | None = None,
    credit_strategy: str = "equal",
    credit_discount: float = 1.0,
    return_gamma: float = 1.0,
    step_reward_fn: MARFTStepRewardFn | None = None,
    per_agent_reward_fns: dict[str, MARFTStepRewardFn] | None = None,
) -> RolloutEngine:
    team = default_marft_team(role_names=role_names, model_ids=model_ids, role_configs=role_configs)
    prompts = {agent.name: str(agent.prompt_template or "") for agent in team.agents}
    if graph is None:
        graph = MARFTWorkflowGraph.sequential(role_names)
    if backend == "tiny-torch":
        policy_backend = TinyTorchPolicyBackend(device=device)
    elif backend == "rule":
        policy_backend = RuleBasedMathPolicyBackend()
    else:
        raise ValueError(f"Unsupported MARFT smoke backend: {backend!r}.")
    return RolloutEngine(
        team=team,
        orchestra=MARFTWorkflowOrchestra(graph=graph, role_prompts=prompts),
        environment=SolverVerifierMathEnvironment(),
        policy_backend=policy_backend,
        credit_assigner=MARFTCreditAssigner(
            strategy=credit_strategy,
            discount=credit_discount,
            gamma=return_gamma,
            step_reward_fn=step_reward_fn,
            per_agent_reward_fns=per_agent_reward_fns,
        ),
    )


def run_marft_smoke(
    *,
    backend: str = "rule",
    device: str = "cpu",
    rollouts_per_task: int = 2,
    role_names: tuple[str, ...] = ("planner", "solver"),
    model_ids: tuple[str, ...] | None = None,
    role_configs: dict[str, dict[str, Any]] | None = None,
    graph: MARFTWorkflowGraph | None = None,
    credit_strategy: str = "equal",
    credit_discount: float = 1.0,
    return_gamma: float = 1.0,
    step_reward_fn: MARFTStepRewardFn | None = None,
    per_agent_reward_fns: dict[str, MARFTStepRewardFn] | None = None,
) -> tuple[MARFTSmokeSummary, RolloutResult]:
    engine = build_marft_engine(
        backend=backend,
        device=device,
        role_names=role_names,
        model_ids=model_ids,
        role_configs=role_configs,
        graph=graph,
        credit_strategy=credit_strategy,
        credit_discount=credit_discount,
        return_gamma=return_gamma,
        step_reward_fn=step_reward_fn,
        per_agent_reward_fns=per_agent_reward_fns,
    )
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
    counts = {role: sum(sample.agent_name == role for sample in result.samples) for role in role_names}
    return (
        MARFTSmokeSummary(
            trajectories=len(result.trajectories),
            samples=len(result.samples),
            success_rate=result.success_rate,
            agent_samples=counts,
            dataproto_rows=dataproto_rows,
            dataproto_status=dataproto_status,
        ),
        result,
    )
