import pytest

from trajweave.backends.policy import PolicyResponse
from trajweave.envs.search import SearchAnswerEnvironment, SearchDocument, SearchTask
from trajweave.orchestration.search_answer import SearchAnswerOrchestra
from trajweave.recipes.doctor_mas.search_smoke import default_search_team, run_search_smoke


def test_doctor_mas_search_recipe_collects_verifier_searcher_answer_turns():
    summary, result = run_search_smoke(rollouts_per_task=1, max_turns=2)

    assert summary.trajectories == 2
    assert summary.success_rate == 1.0
    assert {sample.agent_name for sample in result.samples} == {"verifier", "searcher", "answer"}
    assert all(sample.metadata["credit"] == "doctor_mas_agent_wise_grpo" for sample in result.samples)
    assert any("tool_result" in turn.metadata for trajectory in result.trajectories for turn in trajectory.turns)


def test_search_tool_returns_only_retrieved_evidence():
    task = SearchTask(
        task_id="search_no_hint",
        question="Which city is the capital?",
        answer="secret-answer",
        documents=(SearchDocument(title="Reference", text="The retrieved document has no answer."),),
    )

    result = SearchAnswerEnvironment().search(task, "capital")

    assert "Evidence: Reference" in result
    assert "answer hint" not in result.lower()
    assert task.answer not in result


@pytest.mark.parametrize(
    "final_answer",
    [
        "Final answer: not Paris",
        "Paris is not the answer.",
        "The answer is Berlin, not Paris.",
    ],
)
def test_search_evaluation_rejects_negated_answer_mentions(final_answer):
    task = SearchTask(
        task_id="search_negation",
        question="Which city is the capital of France?",
        answer="Paris",
        documents=(SearchDocument(title="France", text="Its capital is Paris."),),
    )

    reward, success = SearchAnswerEnvironment().evaluate(task, final_answer)

    assert reward == 0.0
    assert success is False


def test_search_evaluation_accepts_an_exact_final_answer():
    task = SearchTask(
        task_id="search_exact",
        question="Which city is the capital of France?",
        answer="Paris",
        documents=(SearchDocument(title="France", text="Its capital is Paris."),),
    )

    assert SearchAnswerEnvironment().evaluate(task, "Reasoning first.\nFinal answer: Paris.") == (1.0, True)


class _NotApprovedSearchBackend:
    def generate(self, request):
        responses = {
            "verifier": "NOT APPROVED: more evidence is required.",
            "searcher": "SEARCH: capital France",
            "answer": "Final answer: Paris",
        }
        text = responses[request.agent.role]
        return PolicyResponse(text=text, token_ids=[1], logprobs=[0.0])


def test_search_orchestra_does_not_treat_not_approved_as_approval():
    task = SearchTask(
        task_id="search_not_approved",
        question="Which city is the capital of France?",
        answer="Paris",
        documents=(SearchDocument(title="France", text="Its capital is Paris."),),
    )

    trajectory = SearchAnswerOrchestra().run(
        episode_id="episode",
        rollout_group="group",
        task=task,
        team=default_search_team(max_turns=1),
        observation=task.question,
        policy_backend=_NotApprovedSearchBackend(),
        environment=SearchAnswerEnvironment(),
    )

    assert [turn.agent_name for turn in trajectory.turns] == ["verifier", "searcher", "answer"]
    assert trajectory.turns[0].metadata["approved"] is False
    assert "answer hint" not in trajectory.turns[1].metadata["tool_result"].lower()
