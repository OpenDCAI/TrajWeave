from __future__ import annotations

from dataclasses import dataclass, field

from trajweave.backends.policy import PolicyRequest, PolicyResponse, StableByteTokenizer
from trajweave.core.specs import AgentSpec, PolicyGroupSpec, TeamSpec
from trajweave.credit.wideseek_r1 import WideSeekR1CreditAssigner
from trajweave.envs.search import SearchAnswerEnvironment, SearchDocument, SearchTask
from trajweave.orchestration.wideseek_r1 import WideSeekR1Orchestra
from trajweave.rollout.engine import RolloutEngine, RolloutResult


@dataclass
class WideSeekR1SmokeSummary:
    trajectories: int
    samples: int
    success_rate: float
    lead_samples: int
    subagent_samples: int
    active_subagents: int
    dataproto_rows: int | None
    dataproto_status: str


@dataclass
class RuleBasedWideSeekR1PolicyBackend:
    tokenizer: StableByteTokenizer = StableByteTokenizer()
    _attempts: dict[str, int] = field(default_factory=dict)
    _current_attempt: dict[str, int] = field(default_factory=dict)

    def generate(self, request: PolicyRequest) -> PolicyResponse:
        stage = str(request.metadata.get("stage", ""))
        if stage == "wideseek_plan":
            attempt = self._attempts.get(request.task_id, 0)
            self._attempts[request.task_id] = attempt + 1
            self._current_attempt[request.task_id] = attempt
            query = str(request.metadata["search_query"])
            count = int(request.metadata["max_parallel_subagents"])
            text = "\n".join(f"CALL subagent: {query} focus {index + 1}" for index in range(count))
        elif stage == "wideseek_search":
            text = f"CALL search: {request.metadata['search_query']}"
        elif stage == "wideseek_access":
            text = f"CALL access: {request.metadata['search_query']}"
        elif stage == "wideseek_summary":
            text = f"Subagent result: {request.metadata['evidence']}"
        elif stage == "wideseek_final":
            answer = (
                request.metadata["answer"]
                if self._current_attempt.get(request.task_id, 0) % 2 == 0
                else "__wrong__"
            )
            text = f"Final answer: {answer}"
        else:
            raise ValueError(f"Unsupported WideSeek-R1 smoke stage: {stage!r}.")
        token_ids = self.tokenizer.encode(text)
        return PolicyResponse(text=text, token_ids=token_ids, logprobs=[0.0] * len(token_ids))


def default_wideseek_r1_team(
    *,
    max_parallel_subagents: int = 3,
    lead_agent: str = "lead_agent",
    subagent_prefix: str = "subagent_",
    shared_model_id: str = "shared_policy",
) -> TeamSpec:
    if max_parallel_subagents < 1:
        raise ValueError("WideSeek-R1 max_parallel_subagents must be at least 1.")
    agents = [
        AgentSpec(
            name=lead_agent,
            role="lead",
            policy_group=shared_model_id,
            tools=("subagent",),
        )
    ]
    agents.extend(
        AgentSpec(
            name=f"{subagent_prefix}{index}",
            role="subagent",
            policy_group=shared_model_id,
            tools=("search", "access"),
        )
        for index in range(max_parallel_subagents)
    )
    return TeamSpec(
        name="wideseek_r1_broad_search",
        agents=tuple(agents),
        policy_groups=(PolicyGroupSpec(name=shared_model_id, backend="local"),),
        orchestra="wideseek_r1_width_search",
        reward="verifiable_outcome_plus_format_search",
        credit="wideseek_r1_multi_agent_grpo",
        max_turns=2 + max_parallel_subagents * 3,
        metadata={
            "shared_model": True,
            "context_isolation": True,
            "dual_level_reweighting": ("agent", "token"),
        },
    )


def default_wideseek_r1_tasks() -> list[SearchTask]:
    return [
        SearchTask(
            task_id="wideseek_capital_france",
            question="Find the capital of France using parallel research.",
            answer="Paris",
            search_query="France capital",
            documents=(
                SearchDocument(title="France", text="France is a European country whose capital is Paris."),
                SearchDocument(title="Germany", text="Germany's capital is Berlin."),
            ),
        ),
        SearchTask(
            task_id="wideseek_python_creator",
            question="Identify the creator of Python using parallel research.",
            answer="Guido van Rossum",
            search_query="Python creator",
            documents=(
                SearchDocument(title="Python", text="Python was created by Guido van Rossum."),
                SearchDocument(title="Java", text="Java was created by James Gosling."),
            ),
        ),
    ]


def build_wideseek_r1_engine(
    *,
    max_parallel_subagents: int = 3,
    lead_agent: str = "lead_agent",
    subagent_prefix: str = "subagent_",
    shared_model_id: str = "shared_policy",
    format_reward: float = 0.1,
    search_reward: float = 0.05,
    length_limit: int = 3000,
    max_length_limit: int = 5000,
    length_penalty: float = 0.1,
) -> RolloutEngine:
    return RolloutEngine(
        team=default_wideseek_r1_team(
            max_parallel_subagents=max_parallel_subagents,
            lead_agent=lead_agent,
            subagent_prefix=subagent_prefix,
            shared_model_id=shared_model_id,
        ),
        orchestra=WideSeekR1Orchestra(
            lead_agent=lead_agent,
            subagent_prefix=subagent_prefix,
            max_parallel_subagents=max_parallel_subagents,
        ),
        environment=SearchAnswerEnvironment(),
        policy_backend=RuleBasedWideSeekR1PolicyBackend(),
        credit_assigner=WideSeekR1CreditAssigner(
            format_reward=format_reward,
            search_reward=search_reward,
            length_limit=length_limit,
            max_length_limit=max_length_limit,
            length_penalty=length_penalty,
        ),
    )


def run_wideseek_r1_smoke(
    *,
    rollouts_per_task: int = 2,
    max_parallel_subagents: int = 3,
    lead_agent: str = "lead_agent",
    subagent_prefix: str = "subagent_",
    shared_model_id: str = "shared_policy",
    format_reward: float = 0.1,
    search_reward: float = 0.05,
    length_limit: int = 3000,
    max_length_limit: int = 5000,
    length_penalty: float = 0.1,
) -> tuple[WideSeekR1SmokeSummary, RolloutResult]:
    engine = build_wideseek_r1_engine(
        max_parallel_subagents=max_parallel_subagents,
        lead_agent=lead_agent,
        subagent_prefix=subagent_prefix,
        shared_model_id=shared_model_id,
        format_reward=format_reward,
        search_reward=search_reward,
        length_limit=length_limit,
        max_length_limit=max_length_limit,
        length_penalty=length_penalty,
    )
    result = engine.run(default_wideseek_r1_tasks(), rollouts_per_task=rollouts_per_task)
    dataproto_rows: int | None = None
    dataproto_status = "skipped"
    try:
        from trajweave.backends.verl import VerlDataProtoAdapter

        dataproto = VerlDataProtoAdapter().build(result.samples)
        dataproto_rows = len(dataproto)
        dataproto_status = "ok"
    except ModuleNotFoundError as exc:
        dataproto_status = f"unavailable: {exc.name}"
    return (
        WideSeekR1SmokeSummary(
            trajectories=len(result.trajectories),
            samples=len(result.samples),
            success_rate=result.success_rate,
            lead_samples=sum(sample.role == "lead" for sample in result.samples),
            subagent_samples=sum(sample.role == "subagent" for sample in result.samples),
            active_subagents=max_parallel_subagents,
            dataproto_rows=dataproto_rows,
            dataproto_status=dataproto_status,
        ),
        result,
    )
