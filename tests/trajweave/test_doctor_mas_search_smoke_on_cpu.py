from trajweave.recipes.doctor_mas.search_smoke import run_search_smoke


def test_doctor_mas_search_recipe_collects_verifier_searcher_answer_turns():
    summary, result = run_search_smoke(rollouts_per_task=1, max_turns=2)

    assert summary.trajectories == 2
    assert summary.success_rate == 1.0
    assert {sample.agent_name for sample in result.samples} == {"verifier", "searcher", "answer"}
    assert all(sample.metadata["credit"] == "doctor_mas_agent_wise_grpo" for sample in result.samples)
    assert any("tool_result" in turn.metadata for trajectory in result.trajectories for turn in trajectory.turns)
