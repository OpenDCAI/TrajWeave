from __future__ import annotations

import json
from pathlib import Path

import pytest

from trajweave.backends.verl.trainers.comlrl_iterative import (
    IterativeConfig,
    MADPOIterWorkflow,
    PromptBatch,
    RewardScores,
    UnsafeResumeError,
)
from trajweave.credit.comlrl.iterative import compare_policy_candidates_by_index, select_policy_comparisons
from trajweave.orchestration.comlrl.comparator import FrozenPolicySnapshot


class FakePolicy:
    def __init__(self, label: str, events: list[tuple]) -> None:
        self.label = label
        self.events = events

    def generate(self, prompt, *, agent_index, num_candidates, context=None):
        self.events.append(("generate", self.label, agent_index, num_candidates))
        return [f"{self.label}:{agent_index}:{index}" for index in range(num_candidates)]


class FakeLifecycle:
    def __init__(self, root: Path, events: list[tuple]) -> None:
        self.root = root
        self.events = events
        self.frozen: list[FrozenPolicySnapshot] = []

    def freeze_current(self, *, policy, iteration):
        path = self.root / "comparator" / f"iteration_{iteration:04d}"
        path.mkdir(parents=True, exist_ok=True)
        snapshot = FrozenPolicySnapshot(
            [FakePolicy(f"copy-{iteration}-{index}", self.events) for index in range(len(policy))],
            f"copy-{iteration}",
            str(path),
            {"api_key": "must-not-persist"},
        )
        self.frozen.append(snapshot)
        self.events.append(("freeze", iteration))
        return snapshot

    def save(self, *, policy, path, iteration, purpose):
        path.mkdir(parents=True, exist_ok=True)
        (path / "policy.txt").write_text(f"{purpose}:{iteration}")
        self.events.append(("save", purpose, iteration))
        return {"purpose": purpose, "iteration": iteration}

    def load(self, *, path, snapshot_id):
        self.events.append(("load", path.name, snapshot_id))
        return FrozenPolicySnapshot(
            [FakePolicy(f"loaded-{path.name}-{index}", self.events) for index in range(2)],
            snapshot_id,
            str(path),
        )

    def release(self, snapshot):
        self.events.append(("release", snapshot.snapshot_id))


class IndexedScorer:
    def __init__(self, events, *, ties=False, fail=False):
        self.events = events
        self.calls = 0
        self.ties = ties
        self.fail = fail

    def __call__(self, *, prompts, candidates_by_agent, context, iteration):
        source = "current" if self.calls % 2 == 0 else "comparator"
        self.calls += 1
        self.events.append(("score", iteration, source))
        if self.fail:
            raise RuntimeError("score failed")
        count = len(candidates_by_agent[0])
        if self.ties:
            processed = tuple(1.0 for _ in range(count))
        elif source == "current":
            processed = (3.0, 1.0, 2.0)[:count]
        else:
            processed = (0.0, 4.0, 1.0)[:count]
        raw = tuple(value + (10.0 if source == "current" else 20.0) for value in processed)
        return RewardScores(raw, processed)


def _workflow(tmp_path, events, *, config, runner, scorer=None, prompt_source=None):
    policies = [FakePolicy("live-0", events), FakePolicy("live-1", events)]
    lifecycle = FakeLifecycle(tmp_path, events)
    workflow = MADPOIterWorkflow(
        config=config,
        artifact_dir=tmp_path / "artifacts",
        current_policy=policies,
        prompt_source=prompt_source or (lambda *, iteration: [PromptBatch((f"p{iteration}-a", f"p{iteration}-b"))]),
        task_reward_scorer=scorer or IndexedScorer(events),
        policy_lifecycle=lifecycle,
        madpo_runner=runner,
    )
    return workflow, lifecycle


def test_indexed_comparison_uses_min_length_drops_ties_sorts_by_comparator_reward_and_rejects_nonfinite():
    comparisons = compare_policy_candidates_by_index([5.0, 1.0, 2.0, 99.0], [0.0, 4.0, 2.0])
    assert [(item.candidate_index, item.winner_source) for item in comparisons] == [(0, "current"), (1, "comparator")]
    selected = select_policy_comparisons(comparisons, mode="comparator_reward", limit=1)
    assert [item.candidate_index for item in selected] == [1]
    with pytest.raises(FloatingPointError, match="finite"):
        compare_policy_candidates_by_index([float("nan")], [0.0])


def test_config_defaults_and_v141_validation():
    config = IterativeConfig()
    assert config.num_iterations == 6
    assert config.preference_num_candidates == 20
    assert config.pair_selection == "comparator_reward"
    assert config.preference_replay_mode == "current"
    with pytest.raises(ValueError, match="num_iterations"):
        IterativeConfig(num_iterations=0)
    with pytest.raises(ValueError, match="preference_replay_k"):
        IterativeConfig(preference_replay_mode="nearest_k")
    with pytest.raises(ValueError, match="comparator_api_url"):
        IterativeConfig(comparator_policy="api")


def test_no_pair_skips_training_but_writes_zero_based_replay_snapshot_and_state(tmp_path):
    events = []
    runner_calls = []
    workflow, _ = _workflow(
        tmp_path,
        events,
        config=IterativeConfig(num_iterations=1, preference_num_candidates=2),
        runner=lambda **kwargs: runner_calls.append(kwargs),
        scorer=IndexedScorer(events, ties=True),
    )

    [state] = workflow.run()

    assert not runner_calls
    assert state.iteration == 0 and state.completed
    assert "training_skipped" in state.completed_stages
    assert (tmp_path / "artifacts/replay/iteration_0000.json").is_file()
    assert (tmp_path / "artifacts/policy_snapshots/iteration_0000/policy.txt").is_file()
    manifest = json.loads((tmp_path / "artifacts/iteration_state/manifest.json").read_text())
    assert manifest["iteration_cursor"] == 1
    assert manifest["completed_iterations"] == [0]


def test_new_workflow_rejects_resume_after_madpo_runner_side_effect(tmp_path):
    events = []
    prompt_calls = []

    def prompts(*, iteration):
        prompt_calls.append(iteration)
        return [PromptBatch((f"p{iteration}-a", f"p{iteration}-b"))]

    def interrupting_runner(*, iteration, **_kwargs):
        events.append(("train", iteration))
        if iteration == 1:
            raise RuntimeError("interrupt")

    config = IterativeConfig(num_iterations=2, preference_num_candidates=2)
    first, _ = _workflow(tmp_path, events, config=config, runner=interrupting_runner, prompt_source=prompts)
    with pytest.raises(RuntimeError, match="interrupt"):
        first.run()
    assert prompt_calls == [0, 1]

    resumed_iterations = []
    second, _ = _workflow(
        tmp_path,
        events,
        config=config,
        runner=lambda *, iteration, **_kwargs: resumed_iterations.append(iteration),
        prompt_source=prompts,
    )
    with pytest.raises(UnsafeResumeError, match="madpo_training"):
        second.run()

    assert resumed_iterations == []
    assert prompt_calls == [0, 1]
    assert events.count(("train", 0)) == 1
    state = json.loads((tmp_path / "artifacts/iteration_state/iteration_0001.json").read_text())
    assert "madpo_training" in state["completed_stages"]


def test_comparator_context_releases_current_copy_on_exception_and_redacts_secret_metadata(tmp_path):
    events = []
    workflow, lifecycle = _workflow(
        tmp_path,
        events,
        config=IterativeConfig(
            num_iterations=1,
            preference_num_candidates=2,
            comparator_policy="current_copy",
        ),
        runner=lambda **_kwargs: None,
        scorer=IndexedScorer(events, fail=True),
    )

    with pytest.raises(RuntimeError, match="score failed"):
        workflow.run()

    assert lifecycle.frozen
    assert ("release", "copy-0") in events
    state_text = (tmp_path / "artifacts/iteration_state/iteration_0000.json").read_text()
    assert "must-not-persist" not in state_text
