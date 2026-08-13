from __future__ import annotations

import json

import pytest

from trajweave.backends.verl.trainers.comlrl_iterative import (
    IterativeConfig,
    MADPOIterWorkflow,
    PromptBatch,
    RewardScores,
)
from trajweave.orchestration.comlrl.comparator import FrozenPolicySnapshot


class Policy:
    def __init__(self, name, events):
        self.name = name
        self.events = events

    def generate(self, prompt, *, agent_index, num_candidates, context=None):
        self.events.append(("generate", self.name, agent_index, num_candidates))
        return [f"{self.name}/agent-{agent_index}/candidate-{index}" for index in range(num_candidates)]


class Lifecycle:
    def __init__(self, tmp_path, events):
        self.tmp_path = tmp_path
        self.events = events

    def freeze_current(self, *, policy, iteration):
        path = self.tmp_path / "copies" / f"iteration_{iteration:04d}"
        path.mkdir(parents=True, exist_ok=True)
        self.events.append(("prepare-copy", iteration))
        return FrozenPolicySnapshot(
            [Policy(f"copy-{iteration}-{index}", self.events) for index in range(len(policy))],
            f"copy-{iteration}",
            str(path),
            {"authorization": "secret"},
        )

    def save(self, *, policy, path, iteration, purpose):
        path.mkdir(parents=True, exist_ok=True)
        (path / "saved").write_text(f"{purpose}:{iteration}")
        self.events.append(("snapshot", iteration))
        return {"iteration": iteration}

    def load(self, *, path, snapshot_id):
        self.events.append(("history", path.name))
        return FrozenPolicySnapshot(
            [Policy(f"history-{path.name}-{index}", self.events) for index in range(2)],
            snapshot_id,
            str(path),
        )

    def release(self, snapshot):
        self.events.append(("cleanup", snapshot.snapshot_id))


class Scorer:
    def __init__(self, events):
        self.events = events
        self.counts = {}

    def __call__(self, *, prompts, candidates_by_agent, context, iteration):
        call = self.counts.get(iteration, 0)
        self.counts[iteration] = call + 1
        source = "current" if call == 0 else "comparator"
        self.events.append(("reward", iteration, source))
        processed = ((3.0, 1.0, 2.0) if source == "current" else (0.0, 4.0, 1.0))[: len(candidates_by_agent[0])]
        offset = 100.0 if source == "current" else 200.0
        return RewardScores(tuple(offset + value for value in processed), tuple(processed))


def build(tmp_path, config, events, runner):
    current = [Policy("live-0", events), Policy("live-1", events)]
    return MADPOIterWorkflow(
        config=config,
        artifact_dir=tmp_path / "run",
        current_policy=current,
        prompt_source=lambda *, iteration: [PromptBatch((f"prompt-{iteration}-0", f"prompt-{iteration}-1"))],
        task_reward_scorer=Scorer(events),
        policy_lifecycle=Lifecycle(tmp_path, events),
        madpo_runner=runner,
    )


def test_two_iteration_current_copy_order_same_index_raw_processed_replay_and_artifacts(tmp_path):
    events = []
    runner_batches = []

    def runner(*, iteration, preference_records, num_train_epochs):
        events.append(("madpo", iteration))
        runner_batches.append((iteration, preference_records, num_train_epochs))

    workflow = build(
        tmp_path,
        IterativeConfig(
            num_iterations=2,
            preference_num_candidates=3,
            comparator_num_candidates=2,
            comparator_policy="current_copy",
            preference_pairs_per_sample=2,
        ),
        events,
        runner,
    )
    states = workflow.run()

    assert [item[0] for item in runner_batches] == [0, 1]
    assert all(len(item[1]) == 2 for item in runner_batches)
    assert all(record.pair_id.find("candidate-000002") < 0 for _, records, _ in runner_batches for record in records)
    assert [event[0] for event in events].count("prepare-copy") == 2
    for iteration in range(2):
        prepare = events.index(("prepare-copy", iteration))
        current_generate = events.index(("generate", "live-0", 0, 3), prepare)
        comparator_generate = events.index(("generate", f"copy-{iteration}-0", 0, 2), current_generate)
        task_reward = events.index(("reward", iteration, "current"), comparator_generate)
        comparator_reward = events.index(("reward", iteration, "comparator"), task_reward)
        train = events.index(("madpo", iteration), comparator_reward)
        snapshot = events.index(("snapshot", iteration), train)
        assert prepare < current_generate < comparator_generate < task_reward < comparator_reward < train < snapshot

    replay = json.loads((tmp_path / "run/replay/iteration_0000.json").read_text())
    assert replay["iteration"] == 0
    assert replay["records"][0]["raw_rewards"]["policy"] >= 100.0
    assert replay["records"][0]["raw_rewards"]["policy"] != replay["records"][0]["processed_rewards"]["chosen"]
    assert "secret" not in json.dumps(replay)
    assert [state.iteration for state in states] == [0, 1]
    assert all(
        state.completed and state.policy_snapshot.endswith(f"iteration_{state.iteration:04d}") for state in states
    )


def test_nearest_k_replay_selects_history_and_uses_seeded_requested_sample_size(tmp_path):
    events = []
    batches = []
    workflow = build(
        tmp_path,
        IterativeConfig(
            num_iterations=2,
            preference_num_candidates=2,
            preference_replay_mode="nearest_k",
            preference_replay_k=2,
            preference_replay_sample_size=4,
        ),
        events,
        lambda *, iteration, preference_records, **_kwargs: batches.append(
            (iteration, tuple(record.iteration for record in preference_records))
        ),
    )

    workflow.run()

    assert batches[0] == (0, (0, 0, 0, 0))
    assert batches[1][0] == 1
    assert set(batches[1][1]) == {0, 1}
    assert len(batches[1][1]) == 4


@pytest.mark.parametrize("policy", ["current", "current_copy", "history"])
def test_current_current_copy_and_snapshot_store_history_resolver(policy, tmp_path):
    events = []
    kwargs = {"comparator_policy": policy}
    if policy == "history":
        kwargs["comparator_history_k"] = 1
    workflow = build(
        tmp_path,
        IterativeConfig(num_iterations=2, preference_num_candidates=2, **kwargs),
        events,
        lambda **_kwargs: None,
    )

    workflow.run()

    if policy == "current":
        assert not any(event[0] in {"prepare-copy", "history"} for event in events)
    elif policy == "current_copy":
        assert [event for event in events if event[0] == "prepare-copy"] == [
            ("prepare-copy", 0),
            ("prepare-copy", 1),
        ]
    else:
        assert [event for event in events if event[0] == "history"] == [
            ("history", "initial"),
            ("history", "iteration_0000"),
        ]
