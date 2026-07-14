from __future__ import annotations

from dataclasses import dataclass

from trajweave.backends.policy import PolicyRequest, PolicyResponse, StableByteTokenizer
from trajweave.core.specs import AgentSpec, PolicyGroupSpec, TeamSpec
from trajweave.credit.matpo import MATPOParentBroadcastCreditAssigner
from trajweave.envs.search import SearchAnswerEnvironment, SearchDocument, SearchTask
from trajweave.orchestration.matpo import PlannerWorkerOrchestra
from trajweave.rollout.engine import RolloutEngine, RolloutResult


@dataclass
class MATPOSmokeSummary:
    trajectories: int
    samples: int
    success_rate: float
    dataproto_rows: int | None
    dataproto_status: str


@dataclass
class RuleBasedMATPOPolicyBackend:
    tokenizer: StableByteTokenizer = StableByteTokenizer()

    def generate(self, request: PolicyRequest) -> PolicyResponse:
        role = request.agent.role.lower()
        stage = request.metadata.get("stage")
        if role == "planner" and stage == "delegate":
            query = request.metadata.get("search_query") or request.observation
            text = f"CALL search_and_browse: {query}"
        elif role == "worker":
            evidence = str(request.metadata.get("evidence", request.team_context))
            answer = request.metadata.get("answer", "unknown")
            text = f"Evidence summary: {evidence}\nSuggested final answer: {answer}"
        elif role == "planner":
            answer = request.metadata.get("answer", "unknown")
            text = f"Final answer: {answer}"
        else:
            text = "PASS"
        token_ids = self.tokenizer.encode(text)
        return PolicyResponse(text=text, token_ids=token_ids, logprobs=[0.0] * len(token_ids))


def default_team(max_turns: int = 3) -> TeamSpec:
    return TeamSpec(
        name="matpo_planner_worker_browse",
        agents=(
            AgentSpec(name="planner", role="planner", policy_group="shared_qwen_tiny", tools=("search_and_browse",)),
            AgentSpec(name="browsing_agent", role="worker", policy_group="shared_qwen_tiny"),
        ),
        policy_groups=(PolicyGroupSpec(name="shared_qwen_tiny", backend="local", trainable=True),),
        orchestra="planner_worker_agent_tool",
        reward="qa_exact_match",
        credit="matpo_parent_broadcast_grpo",
        max_turns=max_turns,
    )


def default_tasks() -> list[SearchTask]:
    return [
        SearchTask(
            task_id="matpo_capital_france",
            question="What is the capital of France?",
            answer="Paris",
            search_query="capital France",
            documents=(
                SearchDocument(title="France", text="France is a country in Europe. Its capital city is Paris."),
                SearchDocument(title="Germany", text="Germany's capital city is Berlin."),
            ),
        ),
        SearchTask(
            task_id="matpo_python_creator",
            question="Who created the Python programming language?",
            answer="Guido van Rossum",
            search_query="Python programming language creator",
            documents=(
                SearchDocument(title="Python", text="Python was created by Guido van Rossum and first released in 1991."),
                SearchDocument(title="Java", text="Java was originally developed by James Gosling."),
            ),
        ),
    ]


def build_matpo_engine(max_turns: int = 3) -> RolloutEngine:
    return RolloutEngine(
        team=default_team(max_turns=max_turns),
        orchestra=PlannerWorkerOrchestra(),
        environment=SearchAnswerEnvironment(),
        policy_backend=RuleBasedMATPOPolicyBackend(),
        credit_assigner=MATPOParentBroadcastCreditAssigner(),
    )


def run_smoke(rollouts_per_task: int = 2, max_turns: int = 3) -> tuple[MATPOSmokeSummary, RolloutResult]:
    engine = build_matpo_engine(max_turns=max_turns)
    result = engine.run(default_tasks(), rollouts_per_task=rollouts_per_task)
    dataproto_rows: int | None = None
    dataproto_status = "skipped"
    try:
        from trajweave.backends.verl import VerlDataProtoAdapter

        dataproto = VerlDataProtoAdapter().build(result.samples)
        dataproto_rows = len(dataproto)
        dataproto_status = "ok"
    except ModuleNotFoundError as exc:
        dataproto_status = f"unavailable: {exc.name}"
    summary = MATPOSmokeSummary(
        trajectories=len(result.trajectories),
        samples=len(result.samples),
        success_rate=result.success_rate,
        dataproto_rows=dataproto_rows,
        dataproto_status=dataproto_status,
    )
    return summary, result
