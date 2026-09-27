from __future__ import annotations

from dataclasses import dataclass, field

from trajweave.backends.policy import PolicyRequest, PolicyResponse, StableByteTokenizer
from trajweave.core.specs import AgentSpec, PolicyGroupSpec, TeamSpec
from trajweave.credit.mrlx import MrlXMGRPOCreditAssigner
from trajweave.envs.search import SearchAnswerEnvironment, SearchDocument, SearchTask
from trajweave.orchestration.mrlx import MrlXResearchOrchestra
from trajweave.rollout.engine import RolloutEngine, RolloutResult


@dataclass
class MrlXSmokeSummary:
    trajectories: int
    samples: int
    success_rate: float
    explorer_samples: int
    adapter_samples: int
    dataproto_rows: int | None
    dataproto_status: str


@dataclass
class RuleBasedMrlXPolicyBackend:
    tokenizer: StableByteTokenizer = StableByteTokenizer()
    _attempts: dict[str, int] = field(default_factory=dict)
    _current_attempt: dict[str, int] = field(default_factory=dict)

    def generate(self, request: PolicyRequest) -> PolicyResponse:
        stage = str(request.metadata.get("stage", ""))
        task_id = request.task_id
        if stage == "mrlx_delegate":
            attempt = self._attempts.get(task_id, 0)
            self._attempts[task_id] = attempt + 1
            self._current_attempt[task_id] = attempt
            text = f"CALL {request.metadata['adapter_agent']}: {request.metadata['search_query']}"
        elif stage == "mrlx_adapter_tool":
            text = f"CALL {request.metadata['tool_name']}: {request.metadata['search_query']}"
        elif stage == "mrlx_adapter_result":
            text = f"Research result: {request.metadata['evidence']}"
        elif stage == "mrlx_final":
            answer = request.metadata["answer"] if self._current_attempt.get(task_id, 0) % 2 == 0 else "__wrong__"
            text = f"Final answer: {answer}"
        else:
            raise ValueError(f"Unsupported MrlX smoke stage: {stage!r}.")
        token_ids = self.tokenizer.encode(text)
        return PolicyResponse(text=text, token_ids=token_ids, logprobs=[0.0] * len(token_ids))


def default_mrlx_team(
    *,
    explorer_agent: str = "main_explorer",
    adapter_agent: str = "sub_adapter",
    explorer_model_id: str = "explorer_policy",
    adapter_model_id: str = "adapter_policy",
    tool_name: str = "search_and_browse",
    research_rounds: int = 1,
) -> TeamSpec:
    return TeamSpec(
        name="mrlx_research_qa",
        agents=(
            AgentSpec(
                name=explorer_agent,
                role="main_explorer",
                policy_group=explorer_model_id,
                tools=(adapter_agent,),
            ),
            AgentSpec(
                name=adapter_agent,
                role="sub_adapter",
                policy_group=adapter_model_id,
                tools=(tool_name,),
            ),
        ),
        policy_groups=(
            PolicyGroupSpec(name=explorer_model_id, backend="local"),
            PolicyGroupSpec(name=adapter_model_id, backend="local"),
        ),
        orchestra="mrlx_async_research",
        reward="mrlx_role_reward",
        credit="mrlx_mgrpo",
        max_turns=research_rounds,
        metadata={
            "explorer_update": "on_policy",
            "adapter_update": "off_policy_one_step_lag",
            "training_target": "train_both_agents",
        },
    )


def default_mrlx_tasks() -> list[SearchTask]:
    return [
        SearchTask(
            task_id="mrlx_capital_france",
            question="What is the capital of France?",
            answer="Paris",
            search_query="capital France",
            documents=(
                SearchDocument(title="France", text="France is a country in Europe. Its capital is Paris."),
                SearchDocument(title="Germany", text="The capital of Germany is Berlin."),
            ),
        ),
        SearchTask(
            task_id="mrlx_python_creator",
            question="Who created the Python programming language?",
            answer="Guido van Rossum",
            search_query="Python programming language creator",
            documents=(
                SearchDocument(title="Python", text="Python was created by Guido van Rossum."),
                SearchDocument(title="Java", text="Java was created by James Gosling."),
            ),
        ),
    ]


def build_mrlx_engine(
    *,
    explorer_agent: str = "main_explorer",
    adapter_agent: str = "sub_adapter",
    explorer_model_id: str = "explorer_policy",
    adapter_model_id: str = "adapter_policy",
    tool_name: str = "search_and_browse",
    research_rounds: int = 1,
    explorer_format_bonus: float = 0.1,
    adapter_format_bonus: float = 0.1,
) -> RolloutEngine:
    team = default_mrlx_team(
        explorer_agent=explorer_agent,
        adapter_agent=adapter_agent,
        explorer_model_id=explorer_model_id,
        adapter_model_id=adapter_model_id,
        tool_name=tool_name,
        research_rounds=research_rounds,
    )
    return RolloutEngine(
        team=team,
        orchestra=MrlXResearchOrchestra(
            explorer_name=explorer_agent,
            adapter_name=adapter_agent,
            tool_name=tool_name,
            research_rounds=research_rounds,
        ),
        environment=SearchAnswerEnvironment(),
        policy_backend=RuleBasedMrlXPolicyBackend(),
        credit_assigner=MrlXMGRPOCreditAssigner(
            explorer_agent=explorer_agent,
            adapter_agent=adapter_agent,
            explorer_format_bonus=explorer_format_bonus,
            adapter_format_bonus=adapter_format_bonus,
        ),
    )


def run_mrlx_smoke(
    *,
    rollouts_per_task: int = 2,
    explorer_agent: str = "main_explorer",
    adapter_agent: str = "sub_adapter",
    explorer_model_id: str = "explorer_policy",
    adapter_model_id: str = "adapter_policy",
    tool_name: str = "search_and_browse",
    research_rounds: int = 1,
    explorer_format_bonus: float = 0.1,
    adapter_format_bonus: float = 0.1,
) -> tuple[MrlXSmokeSummary, RolloutResult]:
    engine = build_mrlx_engine(
        explorer_agent=explorer_agent,
        adapter_agent=adapter_agent,
        explorer_model_id=explorer_model_id,
        adapter_model_id=adapter_model_id,
        tool_name=tool_name,
        research_rounds=research_rounds,
        explorer_format_bonus=explorer_format_bonus,
        adapter_format_bonus=adapter_format_bonus,
    )
    result = engine.run(default_mrlx_tasks(), rollouts_per_task=rollouts_per_task)
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
        MrlXSmokeSummary(
            trajectories=len(result.trajectories),
            samples=len(result.samples),
            success_rate=result.success_rate,
            explorer_samples=sum(sample.agent_name == explorer_agent for sample in result.samples),
            adapter_samples=sum(sample.agent_name == adapter_agent for sample in result.samples),
            dataproto_rows=dataproto_rows,
            dataproto_status=dataproto_status,
        ),
        result,
    )
