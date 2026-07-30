from __future__ import annotations

from dataclasses import dataclass, field

from trajweave.backends.policy import PolicyRequest, PolicyResponse, StableByteTokenizer
from trajweave.core.specs import AgentSpec, PolicyGroupSpec, TeamSpec
from trajweave.credit.matpo import MATPOParentBroadcastCreditAssigner
from trajweave.envs.search import SearchAnswerEnvironment, SearchDocument, SearchTask
from trajweave.orchestration.matpo import PlannerWorkerOrchestra
from trajweave.rollout.engine import RolloutEngine, RolloutResult

_WRONG_ANSWER_SENTINEL = "__wrong__"


@dataclass
class MATPOSmokeSummary:
    trajectories: int
    samples: int
    success_rate: float
    dataproto_rows: int | None
    dataproto_status: str


@dataclass
class RuleBasedMATPOPolicyBackend:
    """Deterministic smoke backend.

    Rollouts alternate between a correct and an incorrect final answer for each
    task (tracked by ``task_id``), so smoke runs exercise both success and failure
    trajectories -- matching what the real HF/VERL emitter path already does via
    ``session_id % 2`` -- instead of always returning the ground truth and
    producing a zero-variance (all-advantage-zero) GRPO batch.
    """

    tokenizer: StableByteTokenizer = StableByteTokenizer()
    _rollout_counts: dict[str, int] = field(default_factory=dict)
    _current_attempt: dict[str, int] = field(default_factory=dict)

    def generate(self, request: PolicyRequest) -> PolicyResponse:
        role = request.agent.role.lower()
        stage = request.metadata.get("stage")
        task_id = request.task_id
        if role == "planner" and stage == "delegate":
            attempt = self._rollout_counts.get(task_id, 0)
            self._rollout_counts[task_id] = attempt + 1
            self._current_attempt[task_id] = attempt
            query = request.metadata.get("search_query") or request.observation
            worker_agent = request.metadata.get("worker_agent", "browsing_agent")
            text = f"CALL {worker_agent}: {query}"
        elif role == "worker" and stage == "worker_call":
            query = request.metadata.get("search_query") or request.observation
            tool_name = request.metadata.get("tool_name", "search_and_browse")
            text = f"CALL {tool_name}: {query}"
        elif role == "worker" and stage == "worker_summary":
            evidence = str(request.metadata.get("evidence", request.team_context))
            answer = self._answer_for(task_id, request.metadata.get("answer", "unknown"))
            text = f"Evidence summary: {evidence}\nSuggested final answer: {answer}"
        elif role == "planner":
            answer = self._answer_for(task_id, request.metadata.get("answer", "unknown"))
            text = f"Final answer: {answer}"
        else:
            text = "PASS"
        token_ids = self.tokenizer.encode(text)
        return PolicyResponse(text=text, token_ids=token_ids, logprobs=[0.0] * len(token_ids))

    def _answer_for(self, task_id: str, ground_truth: str) -> str:
        attempt = self._current_attempt.get(task_id, 0)
        return ground_truth if attempt % 2 == 0 else _WRONG_ANSWER_SENTINEL


def default_team(
    max_turns: int = 3,
    *,
    planner_agent: str = "planner",
    worker_agent: str = "browsing_agent",
    tool_name: str = "search_and_browse",
) -> TeamSpec:
    return TeamSpec(
        name="matpo_planner_worker_browse",
        agents=(
            AgentSpec(name=planner_agent, role="planner", policy_group="shared_qwen_tiny", tools=(worker_agent,)),
            AgentSpec(name=worker_agent, role="worker", policy_group="shared_qwen_tiny", tools=(tool_name,)),
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
                SearchDocument(
                    title="Python", text="Python was created by Guido van Rossum and first released in 1991."
                ),
                SearchDocument(title="Java", text="Java was originally developed by James Gosling."),
            ),
        ),
    ]


def build_matpo_engine(
    max_turns: int = 3,
    *,
    planner_agent: str = "planner",
    worker_agent: str = "browsing_agent",
    tool_name: str = "search_and_browse",
    accuracy_reward_weight: float = 0.9,
    tool_format_reward_weight: float = 0.1,
) -> RolloutEngine:
    return RolloutEngine(
        team=default_team(
            max_turns=max_turns,
            planner_agent=planner_agent,
            worker_agent=worker_agent,
            tool_name=tool_name,
        ),
        orchestra=PlannerWorkerOrchestra(
            planner_name=planner_agent,
            worker_name=worker_agent,
            tool_name=tool_name,
        ),
        environment=SearchAnswerEnvironment(),
        policy_backend=RuleBasedMATPOPolicyBackend(),
        credit_assigner=MATPOParentBroadcastCreditAssigner(
            accuracy_reward_weight=accuracy_reward_weight,
            tool_format_reward_weight=tool_format_reward_weight,
        ),
    )


def run_smoke(
    rollouts_per_task: int = 2,
    max_turns: int = 3,
    *,
    planner_agent: str = "planner",
    worker_agent: str = "browsing_agent",
    tool_name: str = "search_and_browse",
    accuracy_reward_weight: float = 0.9,
    tool_format_reward_weight: float = 0.1,
) -> tuple[MATPOSmokeSummary, RolloutResult]:
    engine = build_matpo_engine(
        max_turns=max_turns,
        planner_agent=planner_agent,
        worker_agent=worker_agent,
        tool_name=tool_name,
        accuracy_reward_weight=accuracy_reward_weight,
        tool_format_reward_weight=tool_format_reward_weight,
    )
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
