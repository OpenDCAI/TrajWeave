import pytest

from trajweave.backends.policy import PolicyRequest, PolicyResponse, StableByteTokenizer
from trajweave.envs.search import SearchAnswerEnvironment, SearchDocument, SearchTask
from trajweave.orchestration.matpo import PlannerWorkerOrchestra
from trajweave.recipes.matpo import run_smoke
from trajweave.recipes.matpo.config import build_matpo_launch_overrides
from trajweave.recipes.matpo.smoke import default_team
from trajweave.recipes.registry import resolve_recipe
from trajweave.runner import run_from_config


def test_matpo_recipe_registry_alias():
    recipe = resolve_recipe("matpo_browse")
    assert recipe.name == "matpo.browse_qa.parent_broadcast"
    assert recipe.runtime_recipe == "matpo_browse"


def test_matpo_smoke_builds_parent_child_samples():
    summary, result = run_smoke(rollouts_per_task=2, max_turns=3)

    assert summary.trajectories == 4
    assert summary.samples == 16
    assert summary.success_rate == 0.5
    assert all(sample.metadata["credit"] == "matpo_parent_broadcast_grpo" for sample in result.samples)

    child_samples = [sample for sample in result.samples if sample.metadata.get("is_from_subagent_tool")]
    main_samples = [sample for sample in result.samples if not sample.metadata.get("is_from_subagent_tool")]
    assert child_samples
    assert main_samples
    parent_ids = {sample.metadata["reqs_id"] for sample in main_samples}
    assert all(sample.metadata["parent_reqs_id"] in parent_ids for sample in child_samples)
    assert all(sample.metadata.get("parent_advantage_broadcast") for sample in child_samples)
    assert any(sample.advantage not in (None, 0.0) for sample in main_samples)
    assert any(sample.advantage not in (None, 0.0) for sample in child_samples)
    for trajectory in result.trajectories:
        trajectory_samples = [sample for sample in result.samples if sample.episode_id == trajectory.episode_id]
        assert len({sample.advantage for sample in trajectory_samples}) == 1
        assert len({sample.reward for sample in trajectory_samples}) == 1
        expected_reward = 1.0 if trajectory.success else 0.1
        assert trajectory.metadata["matpo_combined_reward"] == pytest.approx(expected_reward)
        assert trajectory_samples[0].reward == pytest.approx(expected_reward)


def test_matpo_smoke_honors_custom_planner_worker_tool_names():
    summary, result = run_smoke(
        rollouts_per_task=1,
        max_turns=3,
        planner_agent="lead",
        worker_agent="researcher",
        tool_name="web_search",
    )

    assert summary.trajectories == 2
    assert {sample.agent_name for sample in result.samples} == {"lead", "researcher"}
    assert all(sample.metadata["tool_name"] == "web_search" for sample in result.samples)


def test_matpo_verl_config_wires_explicit_reward_weights_and_replaces_conflicts():
    result = run_from_config(
        {
            "recipe": "matpo.browse_qa.parent_broadcast",
            "mode": "verl_plan",
            "prepare": {"tiny_verl_assets": {"enabled": False}},
            "matpo": {
                "agent_loop_backend": "hf_local_tq",
                "max_turns": 2,
                "planner_agent": "lead",
                "worker_agent": "researcher",
                "tool_name": "web_search",
            },
            "verl": {
                "enabled": True,
                "execute": False,
                "overrides": [
                    "trainer.use_v1=true",
                    "+agent.orchestra.matpo.accuracy_reward_weight=0.2",
                    "+agent.orchestra.matpo.tool_format_reward_weight=0.8",
                ],
            },
        },
        config_path="configs/matpo/browse_verl_tiny.yaml",
    )

    command = result["verl_launch"]["command"]
    assert '+agent.orchestra.matpo.planner_agent="lead"' in command
    assert '+agent.orchestra.matpo.worker_agent="researcher"' in command
    assert '+agent.orchestra.matpo.tool_name="web_search"' in command
    assert sum("agent.orchestra.matpo.accuracy_reward_weight=" in item for item in command) == 1
    assert sum("agent.orchestra.matpo.tool_format_reward_weight=" in item for item in command) == 1
    assert "+agent.orchestra.matpo.accuracy_reward_weight=0.9" in command
    assert "+agent.orchestra.matpo.tool_format_reward_weight=0.1" in command
    assert "tool_format_reward_scale" not in command
    assert "+agent.orchestra.matpo.max_turns=2" in command


def test_matpo_config_rejects_removed_reward_scale_and_invalid_weights():
    with pytest.raises(ValueError, match="tool_format_reward_scale"):
        build_matpo_launch_overrides(
            {"matpo": {"tool_format_reward_scale": 1.0}},
            config_path="configs/matpo/browse_verl_tiny.yaml",
        )
    with pytest.raises(ValueError, match="must sum to 1.0"):
        build_matpo_launch_overrides(
            {"matpo": {"accuracy_reward_weight": 0.8, "tool_format_reward_weight": 0.1}},
            config_path="configs/matpo/browse_verl_tiny.yaml",
        )
    with pytest.raises(ValueError, match="tool_name must be one of"):
        build_matpo_launch_overrides(
            {"matpo": {"tool_name": "shell"}},
            config_path="configs/matpo/browse_verl_tiny.yaml",
        )


def _scripted_policy_backend(
    planner_delegation: str,
    worker_call: str,
    worker_summary: str,
    planner_final: str,
):
    tokenizer = StableByteTokenizer()
    responses = {
        "delegate": planner_delegation,
        "worker_call": worker_call,
        "worker_summary": worker_summary,
        "final": planner_final,
    }

    class _ScriptedBackend:
        def generate(self, request: PolicyRequest) -> PolicyResponse:
            text = responses[request.metadata["stage"]]
            token_ids = tokenizer.encode(text)
            return PolicyResponse(text=text, token_ids=token_ids, logprobs=[0.0] * len(token_ids))

    return _ScriptedBackend()


def _germany_capital_task() -> SearchTask:
    return SearchTask(
        task_id="matpo_capital_task",
        question="What is the capital of France?",
        answer="Berlin",
        search_query="capital France",
        documents=(
            SearchDocument(title="France", text="France is a country in Europe. Its capital city is Paris."),
            SearchDocument(title="Germany", text="Germany's capital city is Berlin."),
        ),
    )


def test_planner_worker_orchestra_runs_worker_call_offline_tool_and_summary():
    task = _germany_capital_task()
    backend = _scripted_policy_backend(
        planner_delegation="CALL browsing_agent: capital of Germany",
        worker_call="CALL search_and_browse: capital of Germany",
        worker_summary="Evidence summary: Germany capital is Berlin.",
        planner_final="Final answer: Berlin",
    )
    trajectory = PlannerWorkerOrchestra().run(
        episode_id="ep-1",
        rollout_group=task.task_id,
        task=task,
        team=default_team(max_turns=3),
        observation=task.question,
        policy_backend=backend,
        environment=SearchAnswerEnvironment(),
    )

    delegate_turn, worker_call_turn, worker_summary_turn, final_turn = trajectory.turns
    assert [turn.metadata["matpo_turn_role"] for turn in trajectory.turns] == [
        "delegate",
        "worker_call",
        "worker_summary",
        "final",
    ]
    assert delegate_turn.metadata["sub_goal"] == "capital of Germany"
    assert worker_call_turn.observation == "capital of Germany"
    assert worker_call_turn.metadata["tool_request"] == "capital of Germany"
    assert "Berlin" in worker_summary_turn.prompt
    assert worker_summary_turn.metadata["tool_observation"].startswith("Evidence: Germany")
    assert "Planner: CALL browsing_agent" in final_turn.prompt
    assert "Worker tool call: CALL search_and_browse" in final_turn.prompt
    assert "Offline tool observation: Evidence: Germany" in final_turn.prompt
    assert "Worker summary: Evidence summary" in final_turn.prompt


def test_planner_worker_orchestra_does_not_invoke_worker_without_valid_planner_action():
    task = _germany_capital_task()
    backend = _scripted_policy_backend(
        planner_delegation="I will look into this.",
        worker_call="CALL search_and_browse: capital France",
        worker_summary="Evidence summary: France capital is Paris.",
        planner_final="Final answer: Paris",
    )
    trajectory = PlannerWorkerOrchestra().run(
        episode_id="ep-2",
        rollout_group=task.task_id,
        task=task,
        team=default_team(max_turns=3),
        observation=task.question,
        policy_backend=backend,
        environment=SearchAnswerEnvironment(),
    )

    assert len(trajectory.turns) == 1
    final_turn = trajectory.turns[0]
    assert final_turn.metadata["matpo_tool_call_count"] == 0
    assert final_turn.metadata["matpo_tool_format_valid"] is False
    assert final_turn.metadata["matpo_turn_role"] == "invalid_planner_call"


def test_planner_worker_orchestra_rejects_same_line_duplicate_planner_call():
    task = _germany_capital_task()
    trajectory = PlannerWorkerOrchestra().run(
        episode_id="ep-duplicate-planner",
        rollout_group=task.task_id,
        task=task,
        team=default_team(max_turns=3),
        observation=task.question,
        policy_backend=_scripted_policy_backend(
            planner_delegation="CALL browsing_agent: Germany CALL browsing_agent: France",
            worker_call="CALL search_and_browse: capital France",
            worker_summary="Evidence summary: France capital is Paris.",
            planner_final="Final answer: Paris",
        ),
        environment=SearchAnswerEnvironment(),
    )

    assert len(trajectory.turns) == 1
    assert trajectory.turns[0].metadata["matpo_turn_role"] == "invalid_planner_call"
    assert trajectory.turns[0].metadata["matpo_tool_call_count"] == 2


def test_planner_worker_orchestra_rejects_same_line_duplicate_worker_call_without_executing_tool():
    task = _germany_capital_task()
    executed: list[str] = []

    class _RecordingEnvironment(SearchAnswerEnvironment):
        def execute_tool(self, tool_name, task, request):
            executed.append(request)
            return super().execute_tool(tool_name, task, request)

    trajectory = PlannerWorkerOrchestra().run(
        episode_id="ep-duplicate-worker",
        rollout_group=task.task_id,
        task=task,
        team=default_team(max_turns=3),
        observation=task.question,
        policy_backend=_scripted_policy_backend(
            planner_delegation="CALL browsing_agent: capital of Germany",
            worker_call="CALL search_and_browse: Germany CALL search_and_browse: France",
            worker_summary="Evidence summary: Germany capital is Berlin.",
            planner_final="Final answer: Berlin",
        ),
        environment=_RecordingEnvironment(),
    )

    assert executed == []
    assert [turn.metadata["matpo_turn_role"] for turn in trajectory.turns] == [
        "delegate",
        "invalid_worker_call",
        "final",
    ]
    assert trajectory.turns[1].metadata["matpo_tool_call_count"] == 2


def test_planner_worker_orchestra_invalid_worker_call_executes_no_tool():
    task = _germany_capital_task()
    tokenizer = StableByteTokenizer()
    executed: list[str] = []

    class _RecordingEnvironment(SearchAnswerEnvironment):
        def execute_tool(self, tool_name, task, request):
            executed.append(request)
            return super().execute_tool(tool_name, task, request)

    class _InvalidWorkerBackend:
        def generate(self, request: PolicyRequest) -> PolicyResponse:
            if request.metadata["stage"] == "delegate":
                text = "CALL browsing_agent: capital of Germany"
            elif request.metadata["stage"] == "worker_call":
                text = "search for capital of Germany"
            else:
                text = "Final answer: Berlin"
            token_ids = tokenizer.encode(text)
            return PolicyResponse(text=text, token_ids=token_ids, logprobs=[0.0] * len(token_ids))

    trajectory = PlannerWorkerOrchestra().run(
        episode_id="ep-invalid-worker",
        rollout_group=task.task_id,
        task=task,
        team=default_team(max_turns=3),
        observation=task.question,
        policy_backend=_InvalidWorkerBackend(),
        environment=_RecordingEnvironment(),
    )

    assert executed == []
    assert [turn.metadata["matpo_turn_role"] for turn in trajectory.turns] == [
        "delegate",
        "invalid_worker_call",
        "final",
    ]
    assert "Worker tool call: search for capital" in trajectory.turns[-1].prompt
    assert "Offline tool observation:" not in trajectory.turns[-1].prompt


def test_planner_worker_orchestra_supports_multi_round_delegation_with_complete_history():
    task = _germany_capital_task()
    tokenizer = StableByteTokenizer()
    planner_contexts: list[str] = []
    planner_calls = 0

    class _TwoRoundBackend:
        def generate(self, request: PolicyRequest) -> PolicyResponse:
            nonlocal planner_calls
            stage = request.metadata["stage"]
            if stage in {"delegate", "final"}:
                planner_calls += 1
                planner_contexts.append(request.team_context)
                if planner_calls == 1:
                    text = "CALL browsing_agent: capital of Germany"
                elif planner_calls == 2:
                    text = "CALL browsing_agent: capital of Germany Berlin"
                else:
                    text = "Final answer: Berlin"
            elif stage == "worker_call":
                text = f"CALL search_and_browse: {request.metadata['search_query']}"
            else:
                text = "Evidence summary: Germany capital is Berlin"
            token_ids = tokenizer.encode(text)
            return PolicyResponse(text=text, token_ids=token_ids, logprobs=[0.0] * len(token_ids))

    trajectory = PlannerWorkerOrchestra().run(
        episode_id="ep-multi",
        rollout_group=task.task_id,
        task=task,
        team=default_team(max_turns=5),
        observation=task.question,
        policy_backend=_TwoRoundBackend(),
        environment=SearchAnswerEnvironment(),
    )

    assert [turn.metadata["matpo_turn_role"] for turn in trajectory.turns] == [
        "delegate",
        "worker_call",
        "worker_summary",
        "delegate",
        "worker_call",
        "worker_summary",
        "final",
    ]
    reqs_ids = [turn.metadata["reqs_id"] for turn in trajectory.turns]
    assert len(reqs_ids) == len(set(reqs_ids))
    assert trajectory.turns[-1].done is True
    assert trajectory.final_answer == "Final answer: Berlin"
    assert "capital of Germany" in planner_contexts[1]
    assert "Offline tool observation: Evidence:" in planner_contexts[1]
    assert planner_contexts[2].count("Worker summary:") == 2


def test_planner_worker_orchestra_forces_final_answer_when_max_turns_exhausted():
    task = _germany_capital_task()
    tokenizer = StableByteTokenizer()

    class _NeverConvergeBackend:
        def generate(self, request: PolicyRequest) -> PolicyResponse:
            if request.metadata.get("forced"):
                text = "Final answer: Berlin"
            elif request.metadata["stage"] == "worker_call":
                text = "CALL search_and_browse: capital of Germany"
            elif request.metadata["stage"] == "worker_summary":
                text = "Evidence summary: Germany capital is Berlin"
            else:
                text = "CALL browsing_agent: capital of Germany"
            token_ids = tokenizer.encode(text)
            return PolicyResponse(text=text, token_ids=token_ids, logprobs=[0.0] * len(token_ids))

    turn_counts = {}
    for max_turns in (1, 3, 8):
        trajectory = PlannerWorkerOrchestra().run(
            episode_id=f"ep-nc-{max_turns}",
            rollout_group=task.task_id,
            task=task,
            team=default_team(max_turns=max_turns),
            observation=task.question,
            policy_backend=_NeverConvergeBackend(),
            environment=SearchAnswerEnvironment(),
        )
        turn_counts[max_turns] = len(trajectory.turns)
        reqs_ids = [turn.metadata["reqs_id"] for turn in trajectory.turns]
        assert len(reqs_ids) == len(set(reqs_ids))
        assert trajectory.turns[-1].done is True
        assert trajectory.final_answer == "Final answer: Berlin"

    assert turn_counts[1] == 4
    assert turn_counts[3] == 10
    assert turn_counts[8] == 25
    assert turn_counts[1] < turn_counts[3] < turn_counts[8]
