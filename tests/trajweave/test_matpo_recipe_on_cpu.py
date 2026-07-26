from trajweave.backends.policy import PolicyRequest, PolicyResponse, StableByteTokenizer
from trajweave.envs.search import SearchAnswerEnvironment, SearchDocument, SearchTask
from trajweave.orchestration.matpo import PlannerWorkerOrchestra
from trajweave.recipes.matpo import run_smoke
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
    assert summary.samples == 12
    # The smoke backend alternates correct/incorrect answers per task so the batch
    # has real reward variance, exercising both success and failure rollouts instead
    # of an all-success, zero-advantage batch.
    assert summary.success_rate == 0.5
    assert all(sample.metadata["credit"] == "matpo_parent_broadcast_grpo" for sample in result.samples)

    child_samples = [sample for sample in result.samples if sample.metadata.get("is_from_subagent_tool")]
    main_samples = [sample for sample in result.samples if not sample.metadata.get("is_from_subagent_tool")]
    assert child_samples
    assert main_samples
    parent_ids = {sample.metadata["reqs_id"] for sample in main_samples}
    assert all(sample.metadata["parent_reqs_id"] in parent_ids for sample in child_samples)
    assert all(sample.metadata.get("parent_advantage_broadcast") for sample in child_samples)
    # With reward variance present, GRPO advantages must not collapse to all-zero.
    assert any(sample.advantage not in (None, 0.0) for sample in main_samples)
    assert any(sample.advantage not in (None, 0.0) for sample in child_samples)


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


def test_matpo_verl_config_wires_planner_worker_tool_overrides():
    result = run_from_config(
        {
            "recipe": "matpo.browse_qa.parent_broadcast",
            "mode": "verl_train",
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
                "overrides": ["trainer.use_v1=true"],
            },
        },
        config_path="configs/matpo/browse_verl_tiny.yaml",
    )

    command = result["verl_launch"]["command"]
    assert '+agent.orchestra.matpo.planner_agent="lead"' in command
    assert '+agent.orchestra.matpo.worker_agent="researcher"' in command
    assert '+agent.orchestra.matpo.tool_name="web_search"' in command
    assert "+agent.orchestra.matpo.max_turns=2" in command


def _scripted_policy_backend(planner_delegation: str, worker_text: str, planner_final: str):
    tokenizer = StableByteTokenizer()
    responses = {"delegate": planner_delegation, "worker": worker_text, "final": planner_final}

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
        # search_query intentionally points at the wrong (France) topic so we can prove
        # the worker follows the planner's delegation instead of this preset value.
        search_query="capital France",
        documents=(
            SearchDocument(title="France", text="France is a country in Europe. Its capital city is Paris."),
            SearchDocument(title="Germany", text="Germany's capital city is Berlin."),
        ),
    )


def test_planner_worker_orchestra_worker_follows_parsed_planner_delegation():
    task = _germany_capital_task()
    team = default_team(max_turns=3)
    orchestra = PlannerWorkerOrchestra()
    backend = _scripted_policy_backend(
        planner_delegation="CALL search_and_browse: capital of Germany",
        worker_text="Evidence summary: Germany capital is Berlin.\nSuggested final answer: Berlin",
        planner_final="Final answer: Berlin",
    )

    trajectory = orchestra.run(
        episode_id="ep-1",
        rollout_group=task.task_id,
        task=task,
        team=team,
        observation=task.question,
        policy_backend=backend,
        environment=SearchAnswerEnvironment(),
    )

    delegate_turn, worker_turn, final_turn = trajectory.turns
    assert delegate_turn.metadata["sub_goal"] == "capital of Germany"
    assert delegate_turn.metadata["matpo_tool_call_count"] == 1
    assert delegate_turn.metadata["matpo_tool_format_valid"] is True
    # The worker's observation/prompt must reflect the planner's parsed delegation,
    # not the task's preset (and here deliberately misleading) search_query.
    assert worker_turn.observation == "capital of Germany"
    assert "Germany" in worker_turn.prompt
    assert "Berlin" in worker_turn.prompt


def test_planner_worker_orchestra_falls_back_to_search_query_on_malformed_delegation():
    task = _germany_capital_task()
    team = default_team(max_turns=3)
    orchestra = PlannerWorkerOrchestra()
    backend = _scripted_policy_backend(
        planner_delegation="I will look into this.",
        worker_text="Evidence summary: France capital is Paris.\nSuggested final answer: Paris",
        planner_final="Final answer: Paris",
    )

    trajectory = orchestra.run(
        episode_id="ep-2",
        rollout_group=task.task_id,
        task=task,
        team=team,
        observation=task.question,
        policy_backend=backend,
        environment=SearchAnswerEnvironment(),
    )

    delegate_turn, worker_turn, _final_turn = trajectory.turns
    assert delegate_turn.metadata["matpo_tool_call_count"] == 0
    assert delegate_turn.metadata["matpo_tool_format_valid"] is False
    assert worker_turn.observation == task.search_query


def test_planner_worker_orchestra_supports_multi_round_delegation():
    # Round 0's evidence is insufficient, so the planner delegates a second, more
    # specific subtask instead of answering immediately. Only after round 1's
    # worker turn does the planner converge with "Final answer: ...".
    task = _germany_capital_task()
    team = default_team(max_turns=5)
    orchestra = PlannerWorkerOrchestra()
    tokenizer = StableByteTokenizer()
    calls: list[str] = []

    class _TwoRoundBackend:
        def generate(self, request: PolicyRequest) -> PolicyResponse:
            stage = request.metadata["stage"]
            calls.append(stage)
            if stage == "delegate":
                text = "CALL search_and_browse: capital of Germany"
            elif stage == "worker":
                text = "Evidence summary: inconclusive, need more detail"
            elif stage == "final" and calls.count("final") == 1:
                text = "CALL search_and_browse: capital of Germany Berlin"
            else:
                text = "Final answer: Berlin"
            token_ids = tokenizer.encode(text)
            return PolicyResponse(text=text, token_ids=token_ids, logprobs=[0.0] * len(token_ids))

    trajectory = orchestra.run(
        episode_id="ep-multi",
        rollout_group=task.task_id,
        task=task,
        team=team,
        observation=task.question,
        policy_backend=_TwoRoundBackend(),
        environment=SearchAnswerEnvironment(),
    )

    assert len(trajectory.turns) == 5
    roles = [turn.metadata.get("matpo_turn_role") for turn in trajectory.turns]
    assert roles == ["delegate", "worker", "delegate", "worker", "final"]
    reqs_ids = [turn.metadata["reqs_id"] for turn in trajectory.turns]
    assert len(reqs_ids) == len(set(reqs_ids)), f"reqs_id must be unique per turn, got {reqs_ids}"
    assert trajectory.turns[-1].done is True
    assert trajectory.final_answer == "Final answer: Berlin"


def test_planner_worker_orchestra_forces_final_answer_when_max_turns_exhausted():
    # A planner that never stops delegating must still be cut off at team.max_turns,
    # with one extra forced final turn appended so the trajectory always converges.
    task = _germany_capital_task()
    tokenizer = StableByteTokenizer()

    class _NeverConvergeBackend:
        def generate(self, request: PolicyRequest) -> PolicyResponse:
            if request.metadata.get("forced"):
                text = "Final answer: Berlin"
            else:
                text = "CALL search_and_browse: capital of Germany"
            token_ids = tokenizer.encode(text)
            return PolicyResponse(text=text, token_ids=token_ids, logprobs=[0.0] * len(token_ids))

    turn_counts = {}
    for max_turns in (1, 3, 8):
        team = default_team(max_turns=max_turns)
        trajectory = PlannerWorkerOrchestra().run(
            episode_id=f"ep-nc-{max_turns}",
            rollout_group=task.task_id,
            task=task,
            team=team,
            observation=task.question,
            policy_backend=_NeverConvergeBackend(),
            environment=SearchAnswerEnvironment(),
        )
        turn_counts[max_turns] = len(trajectory.turns)
        reqs_ids = [turn.metadata["reqs_id"] for turn in trajectory.turns]
        assert len(reqs_ids) == len(set(reqs_ids))
        assert trajectory.turns[-1].done is True
        assert trajectory.final_answer == "Final answer: Berlin"

    # max_turns must actually change the amount of rollout work performed: each
    # additional round adds one delegate + one worker turn, plus one forced final turn.
    assert turn_counts[1] == 3
    assert turn_counts[3] == 7
    assert turn_counts[8] == 17
    assert turn_counts[1] < turn_counts[3] < turn_counts[8]
