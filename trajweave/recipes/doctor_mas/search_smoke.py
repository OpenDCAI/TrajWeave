from __future__ import annotations

import argparse
from dataclasses import dataclass

from trajweave.backends.search import RuleBasedSearchPolicyBackend
from trajweave.core.specs import AgentSpec, PolicyGroupSpec, TeamSpec
from trajweave.credit.doctor_mas import DoctorMASCreditAssigner
from trajweave.envs.search import SearchAnswerEnvironment, SearchDocument, SearchTask
from trajweave.orchestration.search_answer import SearchAnswerOrchestra
from trajweave.rollout.engine import RolloutEngine, RolloutResult


@dataclass
class SearchSmokeSummary:
    trajectories: int
    samples: int
    success_rate: float
    dataproto_rows: int | None
    dataproto_status: str


def default_search_team(max_turns: int = 2) -> TeamSpec:
    return TeamSpec(
        name="doctor_mas_search_answer",
        agents=(
            AgentSpec(name="verifier", role="verifier", policy_group="shared_search_llm"),
            AgentSpec(name="searcher", role="searcher", policy_group="shared_search_llm"),
            AgentSpec(name="answer", role="answer", policy_group="shared_search_llm"),
        ),
        policy_groups=(PolicyGroupSpec(name="shared_search_llm", backend="local", trainable=True),),
        orchestra="search_answer",
        reward="search_exact_match",
        credit="doctor_mas_agent_wise_grpo",
        max_turns=max_turns,
    )


def default_search_tasks() -> list[SearchTask]:
    return [
        SearchTask(
            task_id="search_capital_france",
            question="Which city is the capital of France?",
            answer="Paris",
            search_query="capital France",
            documents=(
                SearchDocument(title="France", text="France is a country in Europe. Its capital city is Paris."),
                SearchDocument(title="Germany", text="Germany's capital city is Berlin."),
            ),
        ),
        SearchTask(
            task_id="search_python_creator",
            question="Who created the Python programming language?",
            answer="Guido van Rossum",
            search_query="Python programming language creator",
            documents=(
                SearchDocument(
                    title="Python",
                    text="Python was created by Guido van Rossum and first released in 1991.",
                ),
                SearchDocument(title="Java", text="Java was originally developed by James Gosling."),
            ),
        ),
    ]


def build_doctor_mas_search_engine(max_turns: int = 2) -> RolloutEngine:
    return RolloutEngine(
        team=default_search_team(max_turns=max_turns),
        orchestra=SearchAnswerOrchestra(),
        environment=SearchAnswerEnvironment(),
        policy_backend=RuleBasedSearchPolicyBackend(),
        credit_assigner=DoctorMASCreditAssigner(),
    )


def run_search_smoke(rollouts_per_task: int = 2, max_turns: int = 2) -> tuple[SearchSmokeSummary, RolloutResult]:
    engine = build_doctor_mas_search_engine(max_turns=max_turns)
    result = engine.run(default_search_tasks(), rollouts_per_task=rollouts_per_task)
    dataproto_rows: int | None = None
    dataproto_status = "skipped"
    try:
        from trajweave.backends.verl import VerlDataProtoAdapter

        dataproto = VerlDataProtoAdapter().build(result.samples)
        dataproto_rows = len(dataproto)
        dataproto_status = "ok"
    except ModuleNotFoundError as exc:
        dataproto_status = f"unavailable: {exc.name}"
    summary = SearchSmokeSummary(
        trajectories=len(result.trajectories),
        samples=len(result.samples),
        success_rate=result.success_rate,
        dataproto_rows=dataproto_rows,
        dataproto_status=dataproto_status,
    )
    return summary, result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the TrajWeave DrMAS search-answer smoke.")
    parser.add_argument("--rollouts-per-task", type=int, default=2)
    parser.add_argument("--max-turns", type=int, default=2)
    args = parser.parse_args()
    summary, result = run_search_smoke(rollouts_per_task=args.rollouts_per_task, max_turns=args.max_turns)
    print(f"trajectories={summary.trajectories}")
    print(f"samples={summary.samples}")
    print(f"success_rate={summary.success_rate:.3f}")
    print(f"dataproto_status={summary.dataproto_status}")
    if summary.dataproto_rows is not None:
        print(f"dataproto_rows={summary.dataproto_rows}")
    for sample in result.samples[:6]:
        print(
            "sample",
            sample.agent_name,
            f"reward={sample.reward:.3f}",
            f"advantage={sample.advantage:.3f}" if sample.advantage is not None else "advantage=None",
            f"group={sample.metadata.get('advantage_group')}",
        )


if __name__ == "__main__":
    main()
