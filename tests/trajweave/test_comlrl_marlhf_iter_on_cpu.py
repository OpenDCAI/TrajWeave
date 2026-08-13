from __future__ import annotations

from dataclasses import dataclass

import pytest

from trajweave.backends.verl.trainers.comlrl_iterative import (
    IterativeConfig,
    MARLHFIterWorkflow,
    PromptBatch,
    RewardScores,
    UnsafeResumeError,
)
from trajweave.orchestration.comlrl.comparator import FrozenPolicySnapshot


class Policy:
    def __init__(self, name, events):
        self.name = name
        self.events = events

    def generate(self, prompt, *, agent_index, num_candidates, context=None):
        self.events.append(("generate", self.name, agent_index, num_candidates))
        return [f"{self.name}:{agent_index}:{index}" for index in range(num_candidates)]


class Lifecycle:
    def __init__(self, events):
        self.events = events

    def freeze_current(self, *, policy, iteration):
        raise AssertionError("not used")

    def save(self, *, policy, path, iteration, purpose):
        path.mkdir(parents=True, exist_ok=True)
        (path / "policy").write_text(f"{purpose}:{iteration}")
        self.events.append(("snapshot", iteration))
        return {"iteration": iteration}

    def load(self, *, path, snapshot_id):
        return FrozenPolicySnapshot(
            [Policy(f"history-{snapshot_id}-0", self.events), Policy(f"history-{snapshot_id}-1", self.events)],
            snapshot_id,
            str(path),
        )


class TaskScorer:
    def __init__(self, events):
        self.events = events
        self.calls = {}

    def __call__(self, *, prompts, candidates_by_agent, context, iteration):
        call = self.calls.get(iteration, 0)
        self.calls[iteration] = call + 1
        source = "current" if call % 2 == 0 else "comparator"
        self.events.append(("task-score", iteration, source))
        processed = ((3.0, 1.0) if source == "current" else (0.0, 4.0))[: len(candidates_by_agent[0])]
        return RewardScores(tuple(value + 50.0 for value in processed), tuple(processed))


@dataclass
class FrozenScorer:
    version: str
    events: list[tuple]
    is_frozen: bool = True

    def __post_init__(self):
        self.calls = {}

    def score(self, *, prompts, candidates_by_agent, context, iteration):
        call = self.calls.get(iteration, 0)
        self.calls[iteration] = call + 1
        source = "current" if call % 2 == 0 else "comparator"
        self.events.append(("rm-score", self.version, iteration, source))
        values = ((2.0, 0.0) if source == "current" else (0.0, 2.0))[: len(candidates_by_agent[0])]
        return RewardScores(tuple(values), tuple(values))


class RewardModelWorker:
    def __init__(self, iteration, events):
        self.iteration = iteration
        self.events = events
        self.optimizer = object()
        self.scorer = None

    def train(self, *, preference_records, iteration, num_train_epochs):
        assert iteration == self.iteration
        self.events.append(("rm-train", iteration, len(preference_records), num_train_epochs, id(self.optimizer)))

    def freeze(self):
        self.events.append(("rm-freeze", self.iteration))
        self.optimizer = None
        self.scorer = FrozenScorer(f"iteration_{self.iteration:04d}", self.events)
        return self.scorer

    def save_checkpoint(self, path):
        path.write_text(f"rm:{self.iteration}")
        self.events.append(("rm-save", self.iteration))

    def close(self):
        self.events.append(("rm-close", self.iteration))


class RewardModelFactory:
    def __init__(self, events):
        self.events = events
        self.instances = []
        self.loaded = []

    def create(self, *, iteration):
        worker = RewardModelWorker(iteration, self.events)
        self.instances.append(worker)
        self.events.append(("rm-create", iteration))
        return worker

    def load_frozen(self, *, checkpoint, version):
        assert checkpoint.is_file()
        self.loaded.append(version)
        self.events.append(("rm-load", version))
        return FrozenScorer(version, self.events)


def build(tmp_path, events, *, config, factory, runner, active=None, prompt_calls=None):
    calls = prompt_calls if prompt_calls is not None else []

    def prompt_source(*, iteration):
        calls.append(iteration)
        return [PromptBatch((f"prompt-{iteration}-0", f"prompt-{iteration}-1"))]

    workflow = MARLHFIterWorkflow(
        config=config,
        artifact_dir=tmp_path / "run",
        current_policy=[Policy("live-0", events), Policy("live-1", events)],
        prompt_source=prompt_source,
        task_reward_scorer=TaskScorer(events),
        policy_lifecycle=Lifecycle(events),
        reward_model_factory=factory,
        rl_runner=runner,
        active_reward_model_setter=active,
    )
    return workflow, calls


def test_two_iterations_bootstrap_previous_rm_rebuild_freeze_order_and_state(tmp_path):
    events = []
    active = []
    factory = RewardModelFactory(events)
    rl_calls = []

    def runner(*, iteration, dispatch, reward_scorer, preference_records, **_kwargs):
        events.append(("rl", iteration, dispatch.algorithm, reward_scorer.version))
        rl_calls.append((iteration, dispatch, reward_scorer, preference_records))

    workflow, prompt_calls = build(
        tmp_path,
        events,
        config=IterativeConfig(
            num_iterations=2,
            preference_num_candidates=2,
            preference_scoring_reward="reward_model",
            reward_num_train_epochs=3,
            rl_algorithm="mareinforce",
        ),
        factory=factory,
        runner=runner,
        active=lambda scorer: active.append(None if scorer is None else scorer.version),
    )
    states = workflow.run()

    assert prompt_calls == [0, 1]
    assert [event for event in events if event[0] == "task-score"] == [
        ("task-score", 0, "current"),
        ("task-score", 0, "comparator"),
    ]
    assert [event for event in events if event[0] == "rm-score"] == [
        ("rm-score", "iteration_0000", 1, "current"),
        ("rm-score", "iteration_0000", 1, "comparator"),
    ]
    assert active == [None, "iteration_0000", None, "iteration_0001"]
    assert len(factory.instances) == 2
    assert factory.instances[0] is not factory.instances[1]
    optimizer_ids = [event[-1] for event in events if event[0] == "rm-train"]
    assert len(set(optimizer_ids)) == 2
    assert all(worker.optimizer is None for worker in factory.instances)
    assert [call[1].algorithm for call in rl_calls] == ["mareinforce", "mareinforce"]
    for iteration in range(2):
        create = events.index(("rm-create", iteration))
        train = next(index for index, event in enumerate(events) if event[:2] == ("rm-train", iteration))
        freeze = events.index(("rm-freeze", iteration))
        save = events.index(("rm-save", iteration))
        rl = next(index for index, event in enumerate(events) if event[:2] == ("rl", iteration))
        snapshot = events.index(("snapshot", iteration), rl)
        assert create < train < freeze < save < rl < snapshot
    assert [state.active_reward_model_version for state in states] == ["iteration_0000", "iteration_0001"]
    assert all(state.reward_model_checkpoint.endswith(f"iteration_{state.iteration:04d}.pt") for state in states)
    assert all(state.completed for state in states)


def test_task_preference_scoring_never_uses_previous_rm(tmp_path):
    events = []
    factory = RewardModelFactory(events)
    workflow, _ = build(
        tmp_path,
        events,
        config=IterativeConfig(num_iterations=2, preference_num_candidates=2, preference_scoring_reward="task"),
        factory=factory,
        runner=lambda **_kwargs: None,
    )

    workflow.run()

    assert len([event for event in events if event[0] == "task-score"]) == 4
    assert not [event for event in events if event[0] == "rm-score"]


@pytest.mark.parametrize(
    ("algorithm", "trainer", "advantage", "topology"),
    [
        ("magrpo", "trajweave_multi_actor_sync", "mean", None),
        ("mareinforce", "trajweave_multi_actor_sync", "raw", None),
        ("marloo", "trajweave_multi_actor_sync", "rloo", None),
        ("maremax", "trajweave_multi_actor_sync", "max", None),
        ("iac", "trajweave_multi_actor_critic_sync", None, "independent"),
        ("maac", "trajweave_multi_actor_critic_sync", None, "centralized"),
    ],
)
def test_six_way_rl_dispatch_is_passed_to_injected_runner(algorithm, trainer, advantage, topology, tmp_path):
    events = []
    factory = RewardModelFactory(events)
    dispatches = []
    workflow, _ = build(
        tmp_path,
        events,
        config=IterativeConfig(num_iterations=1, preference_num_candidates=2, rl_algorithm=algorithm),
        factory=factory,
        runner=lambda *, dispatch, **_kwargs: dispatches.append(dispatch),
    )

    workflow.run()

    [dispatch] = dispatches
    assert (dispatch.algorithm, dispatch.trainer, dispatch.advantage_mode, dispatch.critic_topology) == (
        algorithm,
        trainer,
        advantage,
        topology,
    )


def test_resume_rejects_interrupted_online_rl_with_possible_side_effects(tmp_path):
    events = []
    factory = RewardModelFactory(events)
    prompt_calls = []

    def interrupt(*, iteration, **_kwargs):
        events.append(("rl-attempt", iteration))
        if iteration == 1:
            raise RuntimeError("interrupted")

    config = IterativeConfig(
        num_iterations=2,
        preference_num_candidates=2,
        preference_scoring_reward="reward_model",
    )
    first, _ = build(
        tmp_path,
        events,
        config=config,
        factory=factory,
        runner=interrupt,
        prompt_calls=prompt_calls,
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        first.run()
    assert len(factory.instances) == 2
    assert prompt_calls == [0, 1]

    resumed = []
    second, _ = build(
        tmp_path,
        events,
        config=config,
        factory=factory,
        runner=lambda *, iteration, **_kwargs: resumed.append(iteration),
        prompt_calls=prompt_calls,
    )
    with pytest.raises(UnsafeResumeError, match="may already have produced side effects"):
        second.run()

    assert not resumed
    assert prompt_calls == [0, 1]
    assert len(factory.instances) == 2
