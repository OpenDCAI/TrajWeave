from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from trajweave.backends.verl.routing import safe_actor_role_key
from trajweave.backends.verl.trainers import comlrl_iterative_runtime
from trajweave.backends.verl.trainers.comlrl_iterative import (
    CoMLRLIterativeConfig,
    CoMLRLIterativeController,
    IterativeWorkflowError,
    UnsafeResumeError,
    records_to_joint_preference_pairs,
)
from trajweave.backends.verl.trainers.comlrl_iterative_runtime import (
    apply_iterative_resume_model_paths,
    iterative_settings,
)
from trajweave.orchestration.comlrl.comparator import ComparatorResult
from trajweave.storage.preference_replay import (
    PolicySnapshotStore,
    PreferenceReplayRecord,
    PreferenceReplayStore,
)

AGENTS = ("alice", "bob")


class StubComparator:
    def __init__(self, iteration: int) -> None:
        self.iteration = iteration

    def generate(self, prompts, *, num_candidates, iteration=None, context=None):
        return ComparatorResult(
            prompts=tuple(prompts),
            candidates_by_agent=tuple(tuple("candidate" for _ in range(num_candidates)) for _ in prompts),
            policy="current",
            generation_mode="decentralized",
            iteration=iteration,
            provenance={},
        )


class TrackingReplayStore:
    def __init__(self, root: Path, events: list[tuple[Any, ...]], *, fail_select_once: bool = False) -> None:
        self.store = PreferenceReplayStore(root)
        self.events = events
        self.fail_select_once = fail_select_once

    @property
    def available_iterations(self):
        return self.store.available_iterations

    def write_iteration(self, iteration, records):
        self.events.append(("replay_write", iteration, tuple(record.pair_id for record in records)))
        return self.store.write_iteration(iteration, records)

    def load_iteration(self, iteration):
        self.events.append(("replay_load", iteration))
        return self.store.load_iteration(iteration)

    def select(self, mode, **kwargs):
        self.events.append(("replay_select", mode, dict(kwargs)))
        if self.fail_select_once:
            self.fail_select_once = False
            raise RuntimeError("simulated stop after replay shard commit")
        return self.store.select(mode, **kwargs)


class TrackingSnapshotStore:
    def __init__(self, root: Path, events: list[tuple[Any, ...]]) -> None:
        self.store = PolicySnapshotStore(root)
        self.events = events

    @property
    def available_iterations(self):
        return self.store.available_iterations

    def commit_iteration(self, iteration, writer, metadata=None):
        self.events.append(("snapshot_commit", iteration, f"iteration_{iteration:04d}"))
        return self.store.commit_iteration(iteration, writer, metadata)


def replay_record(iteration: int, index: int = 0, *, agents: int = 2, tied: bool = False):
    chosen_reward = 1.0 if tied else 2.0
    return PreferenceReplayRecord(
        pair_id=f"pair-{iteration}-{index}",
        iteration=iteration,
        prompts=tuple(f"prompt-{iteration}-{index}-{agent}" for agent in range(agents)),
        chosen=tuple(f"chosen-{iteration}-{index}-{agent}" for agent in range(agents)),
        rejected=tuple(f"rejected-{iteration}-{index}-{agent}" for agent in range(agents)),
        processed_chosen_reward=chosen_reward,
        processed_rejected_reward=1.0,
        raw_policy_reward=2.0,
        raw_comparator_reward=1.0,
        raw_candidate_rewards=(2.0, 1.0),
        candidate_mean=1.5,
        policy_provenance={"policy": "current"},
        comparator_provenance={"policy": "history"},
    )


def stores(tmp_path, events):
    return (
        TrackingReplayStore(tmp_path / "replay", events),
        TrackingSnapshotStore(tmp_path / "snapshots", events),
    )


def save_marker(iteration, path):
    (path / "policy.txt").write_text(str(iteration))


def test_madpo_iter_two_rounds_have_exact_order_and_iteration_numbering(tmp_path):
    events: list[tuple[Any, ...]] = []
    replay_store, snapshot_store = stores(tmp_path, events)

    def comparator_factory(iteration):
        events.append(("comparator", iteration))
        return StubComparator(iteration)

    def collect(iteration, comparator, candidates, pairs_per_sample, pair_selection):
        events.append(("collect", iteration, comparator.iteration, candidates, pairs_per_sample, pair_selection))
        return [replay_record(iteration)]

    def train(iteration, pairs):
        pair = pairs[0]
        events.append(
            (
                "madpo",
                iteration,
                pair.preference_pair_id,
                tuple(pair.prompts_by_agent),
                pair.metadata["replay_iteration"],
            )
        )

    def save(iteration, target_path):
        events.append(("save", iteration, f"iteration_{iteration:04d}"))
        (target_path / "policy.txt").write_text(str(iteration))

    controller = CoMLRLIterativeController(
        CoMLRLIterativeConfig(num_iterations=2),
        agent_ids=AGENTS,
        workflow_dir=tmp_path / "workflow",
        replay_store=replay_store,
        snapshot_store=snapshot_store,
        comparator_factory=comparator_factory,
        collect_preferences=collect,
        train_madpo=train,
        save_policy_snapshot=save,
    )
    controller.run()

    select_0 = {
        "current_iteration": 0,
        "sample_size": None,
        "nearest_k": None,
        "replay_lambda": None,
        "seed": None,
        "replacement": True,
    }
    select_1 = {**select_0, "current_iteration": 1}
    assert events == [
        ("comparator", 0),
        ("collect", 0, 0, 20, 4, "comparator_reward"),
        ("replay_write", 0, ("pair-0-0",)),
        ("replay_select", "current", select_0),
        ("madpo", 0, "pair-0-0", AGENTS, 0),
        ("snapshot_commit", 0, "iteration_0000"),
        ("save", 0, "iteration_0000"),
        ("comparator", 1),
        ("collect", 1, 1, 20, 4, "comparator_reward"),
        ("replay_write", 1, ("pair-1-0",)),
        ("replay_select", "current", select_1),
        ("madpo", 1, "pair-1-0", AGENTS, 1),
        ("snapshot_commit", 1, "iteration_0001"),
        ("save", 1, "iteration_0001"),
    ]
    assert replay_store.available_iterations == (0, 1)
    assert snapshot_store.available_iterations == (0, 1)
    manifest = json.loads((tmp_path / "workflow" / "workflow_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["next_iteration"] == 2
    assert "absent from upstream CoMLRL v1.4.1" in manifest["resume_extension"]
    assert not list((tmp_path / "workflow").glob("*.tmp"))


def test_marlhf_iter_runs_reward_model_before_online_rl_and_passes_artifact(tmp_path):
    events: list[tuple[Any, ...]] = []
    replay_store, snapshot_store = stores(tmp_path, events)
    artifacts = [object(), object()]

    def reward_model(iteration, pairs):
        events.append(("rm", iteration, pairs[0].preference_pair_id))
        return artifacts[iteration]

    def online(iteration, pairs, artifact):
        events.append(("online", iteration, pairs[0].preference_pair_id, artifact))

    controller = CoMLRLIterativeController(
        CoMLRLIterativeConfig(algorithm="MARLHFIter", num_iterations=2),
        agent_ids=AGENTS,
        workflow_dir=tmp_path / "workflow",
        replay_store=replay_store,
        snapshot_store=snapshot_store,
        comparator_factory=StubComparator,
        collect_preferences=lambda iteration, *_args: [replay_record(iteration)],
        train_reward_model=reward_model,
        train_online_rl=online,
        save_policy_snapshot=lambda iteration, path: (
            events.append(("save", iteration)),
            (path / "policy.txt").write_text(str(iteration)),
        )[1],
    )
    controller.run()

    training_events = [event for event in events if event[0] in {"rm", "online"}]
    assert training_events == [
        ("rm", 0, "pair-0-0"),
        ("online", 0, "pair-0-0", artifacts[0]),
        ("rm", 1, "pair-1-0"),
        ("online", 1, "pair-1-0", artifacts[1]),
    ]


@pytest.mark.parametrize(
    ("mode", "config_kwargs", "expected"),
    [
        ("current", {}, {"nearest_k": None, "replay_lambda": None, "sample_size": None}),
        ("nearest_k", {"nearest_k": 3, "replay_sample_size": 2}, {"nearest_k": 3, "sample_size": 2}),
        ("all_history", {"replay_sample_size": 2}, {"nearest_k": None, "sample_size": 2}),
        (
            "lambda_decay",
            {"replay_lambda": 0.6, "replay_sample_size": 2},
            {"replay_lambda": 0.6, "sample_size": 2},
        ),
    ],
)
def test_all_replay_modes_pass_selection_parameters(tmp_path, mode, config_kwargs, expected):
    events: list[tuple[Any, ...]] = []
    replay_store, snapshot_store = stores(tmp_path, events)
    trained: list[tuple[str, ...]] = []
    controller = CoMLRLIterativeController(
        CoMLRLIterativeConfig(num_iterations=1, replay_mode=mode, **config_kwargs),
        agent_ids=AGENTS,
        workflow_dir=tmp_path / "workflow",
        replay_store=replay_store,
        snapshot_store=snapshot_store,
        comparator_factory=StubComparator,
        collect_preferences=lambda iteration, *_args: [
            replay_record(iteration, 0),
            replay_record(iteration, 1),
        ],
        train_madpo=lambda _iteration, pairs: trained.append(tuple(pair.preference_pair_id for pair in pairs)),
        save_policy_snapshot=save_marker,
    )
    controller.run()

    selection = next(event for event in events if event[0] == "replay_select")
    assert selection[1] == mode
    for key, value in expected.items():
        assert selection[2][key] == value
    assert trained and len(trained[0]) == 2


def test_resume_from_completed_iteration_boundary_starts_at_next_iteration(tmp_path):
    collected: list[int] = []

    def stop_before_second_collection(iteration):
        if iteration == 1:
            raise RuntimeError("stop at completed iteration boundary")
        return StubComparator(iteration)

    common = {
        "config": CoMLRLIterativeConfig(num_iterations=2),
        "agent_ids": AGENTS,
        "workflow_dir": tmp_path / "workflow",
        "collect_preferences": lambda iteration, *_args: collected.append(iteration) or [replay_record(iteration)],
        "train_madpo": lambda *_args: None,
        "save_policy_snapshot": save_marker,
    }
    first = CoMLRLIterativeController(
        **common,
        comparator_factory=stop_before_second_collection,
    )
    with pytest.raises(RuntimeError, match="completed iteration boundary"):
        first.run()
    boundary = json.loads(first.manifest_path.read_text())
    assert boundary["status"] == "ready"
    assert boundary["next_iteration"] == 1

    resumed = CoMLRLIterativeController(
        **common,
        comparator_factory=StubComparator,
    )
    resumed.run("resume")
    assert collected == [0, 1]
    assert resumed.replay_store.available_iterations == (0, 1)
    assert resumed.snapshot_store.available_iterations == (0, 1)


def test_resume_from_replay_ready_does_not_collect_twice(tmp_path):
    events: list[tuple[Any, ...]] = []
    replay_store = TrackingReplayStore(tmp_path / "replay", events, fail_select_once=True)
    snapshot_store = TrackingSnapshotStore(tmp_path / "snapshots", events)
    collect_calls = 0

    def collect(iteration, *_args):
        nonlocal collect_calls
        collect_calls += 1
        return [replay_record(iteration)]

    kwargs = {
        "config": CoMLRLIterativeConfig(num_iterations=1),
        "agent_ids": AGENTS,
        "workflow_dir": tmp_path / "workflow",
        "comparator_factory": StubComparator,
        "collect_preferences": collect,
        "train_madpo": lambda *_args: events.append(("train",)),
        "save_policy_snapshot": save_marker,
    }
    first = CoMLRLIterativeController(
        **kwargs,
        replay_store=replay_store,
        snapshot_store=snapshot_store,
    )
    with pytest.raises(RuntimeError, match="simulated stop"):
        first.run()
    assert collect_calls == 1
    assert json.loads(first.manifest_path.read_text())["status"] == "replay_ready"

    resumed = CoMLRLIterativeController(
        **kwargs,
        replay_store=TrackingReplayStore(tmp_path / "replay", events),
        snapshot_store=TrackingSnapshotStore(tmp_path / "snapshots", events),
    )
    resumed.run("resume")
    assert collect_calls == 1
    assert ("replay_load", 0) in events
    assert ("train",) in events


@pytest.mark.parametrize("interruption", ["training", "snapshot"])
def test_resume_rejects_training_and_snapshot_interruptions(tmp_path, interruption):
    def fail_train(*_args):
        raise RuntimeError("training stopped")

    def fail_snapshot(*_args):
        raise RuntimeError("snapshot stopped")

    controller = CoMLRLIterativeController(
        CoMLRLIterativeConfig(num_iterations=1),
        agent_ids=AGENTS,
        workflow_dir=tmp_path / "workflow",
        comparator_factory=StubComparator,
        collect_preferences=lambda iteration, *_args: [replay_record(iteration)],
        train_madpo=fail_train if interruption == "training" else lambda *_args: None,
        save_policy_snapshot=fail_snapshot if interruption == "snapshot" else lambda *_args: None,
    )
    with pytest.raises(RuntimeError, match="stopped"):
        controller.run()
    assert json.loads(controller.manifest_path.read_text())["status"] == interruption

    resumed = CoMLRLIterativeController(
        CoMLRLIterativeConfig(num_iterations=1),
        agent_ids=AGENTS,
        workflow_dir=tmp_path / "workflow",
        comparator_factory=StubComparator,
        collect_preferences=lambda *_args: pytest.fail("unsafe resume must not collect"),
        train_madpo=lambda *_args: pytest.fail("unsafe resume must not train"),
        save_policy_snapshot=lambda *_args: pytest.fail("unsafe resume must not save"),
    )
    with pytest.raises(UnsafeResumeError, match="optimizer/RNG state is not resumable"):
        resumed.run("resume")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"algorithm": "MADPO"},
        {"num_iterations": 0},
        {"num_target_candidates": 0},
        {"max_turns": 2},
        {"joint_mode": "align"},
        {"pair_selection": "all", "pairs_per_sample": 4},
        {"replay_mode": "nearest_k"},
        {"replay_mode": "lambda_decay", "replay_lambda": 1.1},
    ],
)
def test_invalid_configuration_fails_fast(kwargs):
    with pytest.raises((TypeError, ValueError)):
        CoMLRLIterativeConfig(**kwargs)


@pytest.mark.parametrize(
    ("records", "agent_ids", "message"),
    [
        ([replay_record(0, agents=2)], ("alice",), "wrong agent count"),
        ([replay_record(0, tied=True)], AGENTS, "tied"),
        ([object()], AGENTS, "PreferenceReplayRecord"),
    ],
)
def test_empty_tied_malformed_and_agent_mismatched_collection_fail(tmp_path, records, agent_ids, message):
    controller = CoMLRLIterativeController(
        CoMLRLIterativeConfig(num_iterations=1),
        agent_ids=agent_ids,
        workflow_dir=tmp_path / "workflow",
        comparator_factory=StubComparator,
        collect_preferences=lambda *_args: records,
        train_madpo=lambda *_args: pytest.fail("invalid pairs must not train"),
        save_policy_snapshot=save_marker,
    )
    with pytest.raises((TypeError, ValueError), match=message):
        controller.run()


def test_empty_collection_skips_training_but_commits_policy_boundary(tmp_path):
    trained = []
    controller = CoMLRLIterativeController(
        CoMLRLIterativeConfig(num_iterations=1),
        agent_ids=AGENTS,
        workflow_dir=tmp_path / "workflow",
        comparator_factory=StubComparator,
        collect_preferences=lambda *_args: [],
        train_madpo=lambda *_args: trained.append(True),
        save_policy_snapshot=lambda _iteration, path: (path / "policy.txt").write_text("saved"),
    )

    controller.run()

    assert not trained
    assert controller.replay_store.load_iteration(0) == ()
    assert (tmp_path / "workflow/policy_snapshots/iteration_0000/policy.txt").is_file()


def test_replay_replacement_gets_unique_training_pair_ids():
    source = replay_record(1)
    pairs = records_to_joint_preference_pairs([source, source], agent_ids=AGENTS)

    assert [pair.preference_pair_id for pair in pairs] == ["pair-1-0", "pair-1-0:draw:1"]
    assert [pair.metadata["source_preference_pair_id"] for pair in pairs] == ["pair-1-0", "pair-1-0"]
    assert [pair.metadata["replay_occurrence"] for pair in pairs] == [0, 1]


def test_strict_conversion_uses_agent_id_order_and_rejects_future_records():
    pair = records_to_joint_preference_pairs([replay_record(2)], agent_ids=("second", "first"))[0]
    assert list(pair.prompts_by_agent) == ["second", "first"]
    assert pair.prompts_by_agent["second"].endswith("-0")
    assert pair.prompts_by_agent["first"].endswith("-1")
    with pytest.raises(ValueError, match="future shard"):
        records_to_joint_preference_pairs([replay_record(2)], agent_ids=AGENTS, expected_max_iteration=1)


def test_resume_rejects_identity_drift_and_fresh_rejects_existing_state(tmp_path):
    controller = CoMLRLIterativeController(
        CoMLRLIterativeConfig(num_iterations=1),
        agent_ids=AGENTS,
        workflow_dir=tmp_path / "workflow",
        comparator_factory=StubComparator,
        collect_preferences=lambda iteration, *_args: [replay_record(iteration)],
        train_madpo=lambda *_args: None,
        save_policy_snapshot=save_marker,
    )
    controller.run()

    with pytest.raises(FileExistsError, match="fresh"):
        controller.run("fresh")

    drifted = CoMLRLIterativeController(
        CoMLRLIterativeConfig(num_iterations=2),
        agent_ids=AGENTS,
        workflow_dir=tmp_path / "workflow",
        comparator_factory=StubComparator,
        collect_preferences=lambda *_args: [],
        train_madpo=lambda *_args: None,
        save_policy_snapshot=save_marker,
    )
    with pytest.raises(IterativeWorkflowError, match="configuration drift"):
        drifted.run("resume")


def test_iterative_runtime_settings_merge_launch_wrapper_and_resume_actor_paths(tmp_path):
    snapshot_store = PolicySnapshotStore(tmp_path / "state/policy_snapshots")

    def write_snapshot(snapshot):
        for group_id in ("actor-a", "actor-b"):
            model_path = snapshot / safe_actor_role_key(group_id) / "huggingface"
            model_path.mkdir(parents=True)
            (model_path / "config.json").write_text("{}")

    snapshot_store.commit_iteration(1, write_snapshot)
    config = OmegaConf.create(
        {
            "trajweave": {
                "comlrl": {
                    "iterative": {
                        "enabled": True,
                        "algorithm": "madpo_iter",
                        "config": {
                            "num_iterations": 3,
                            "state_dir": str(tmp_path / "state"),
                            "run_mode": "resume",
                            "replay": {"mode": "current"},
                        },
                    }
                }
            },
            "agent": {
                "model_ids": ["actor-a", "actor-b"],
                "worker_groups": {
                    "actor-a": {"model_path": "/old/a"},
                    "actor-b": {"model_path": "/old/b"},
                },
            },
            "trainer": {"default_local_dir": str(tmp_path), "resume_mode": "auto"},
        }
    )

    settings = iterative_settings(config)
    apply_iterative_resume_model_paths(config)

    assert settings["num_iterations"] == 3
    assert settings["algorithm"] == "madpo_iter"
    assert config.agent.worker_groups["actor-a"].model_path.endswith(
        "policy_snapshots/iteration_0001/trajweave_actor_actor_a/huggingface"
    )
    assert config.agent.worker_groups["actor-b"].model_path.endswith(
        "policy_snapshots/iteration_0001/trajweave_actor_actor_b/huggingface"
    )
    assert config.trainer.resume_mode == "disable"


def test_runtime_replay_materialization_pads_variable_lengths_before_tq_put(monkeypatch):
    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 9

        @staticmethod
        def encode(text, add_special_tokens=False):
            values = [ord(char) % 17 + 1 for char in text]
            return ([9] if add_special_tokens else []) + values

    trainer = SimpleNamespace(
        tokenizer=Tokenizer(),
        global_steps=3,
        config=SimpleNamespace(
            actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(prompt_length=6, response_length=4))
        ),
    )
    record = replay_record(1)
    record = PreferenceReplayRecord(
        iteration=record.iteration,
        prompts=("short", "a much longer prompt"),
        chosen=("winner", "ok"),
        rejected=("loser", "different"),
        processed_chosen_reward=record.processed_chosen_reward,
        processed_rejected_reward=record.processed_rejected_reward,
        raw_policy_reward=record.raw_policy_reward,
        raw_comparator_reward=record.raw_comparator_reward,
        raw_candidate_rewards=record.raw_candidate_rewards,
        candidate_mean=record.candidate_mean,
        policy_provenance=record.policy_provenance,
        comparator_provenance=record.comparator_provenance,
    )
    pairs = records_to_joint_preference_pairs([record], agent_ids=AGENTS)
    captured = {}

    def capture_put(*, keys, partition_id, fields, tags):
        captured.update(keys=keys, partition_id=partition_id, fields=fields, tags=tags)

    monkeypatch.setattr(comlrl_iterative_runtime.tq, "kv_batch_put", capture_put)
    batch = comlrl_iterative_runtime._materialize_pairs_to_tq(
        trainer,
        pairs,
        agent_ids=AGENTS,
        model_ids=("actor-a", "actor-b"),
        iteration=0,
        epoch=0,
    )

    assert len(batch.keys) == 4
    assert captured["fields"]["prompts"].shape == torch.Size([4, 6])
    assert captured["fields"]["responses"].shape == torch.Size([4, 4])
    assert captured["fields"]["input_ids"].shape == torch.Size([4, 10])
    assert captured["fields"]["response_mask"].shape == torch.Size([4, 4])
    assert all(tag["seq_len"] == 10 and tag["global_steps"] == 3 for tag in captured["tags"])


def test_madpo_replay_training_shuffles_and_updates_once_per_pair_batch(monkeypatch):
    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 9

        @staticmethod
        def encode(text, add_special_tokens=False):
            values = [ord(char) % 17 + 1 for char in text]
            return ([9] if add_special_tokens else []) + values

    class Trainer:
        def __init__(self):
            self.tokenizer = Tokenizer()
            self.global_steps = 0
            self.config = SimpleNamespace(
                actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(prompt_length=6, response_length=4))
            )
            self.update_row_counts = []
            self.step_begin_calls = 0
            self.step_end_calls = 0

        def _balance_batch(self, batch, metrics):
            metrics["row_count"] = len(batch.keys)
            return batch

        @staticmethod
        def _compute_old_log_prob(batch, _metrics):
            return batch

        @staticmethod
        def _compute_advantage(batch, _metrics):
            return batch

        def _update_actor(self, batch, _metrics):
            self.update_row_counts.append(len(batch.keys))

        def on_step_begin(self):
            self.step_begin_calls += 1

        def on_step_end(self):
            self.step_end_calls += 1

    settings = {"train_batch_size": 2, "preference_train_batch_size": 1}
    config = SimpleNamespace(data=SimpleNamespace(train_batch_size=5))
    batch_size = comlrl_iterative_runtime._iterative_preference_train_batch_size(settings, config)
    pairs = records_to_joint_preference_pairs(
        [replay_record(0, index=index) for index in range(5)],
        agent_ids=AGENTS,
    )
    shuffle_sizes = []
    put_steps = []
    clear_sizes = []

    def reverse_shuffle(values):
        shuffle_sizes.append(len(values))
        values.reverse()

    monkeypatch.setattr(comlrl_iterative_runtime.random, "shuffle", reverse_shuffle)
    monkeypatch.setattr(
        comlrl_iterative_runtime.tq,
        "kv_batch_put",
        lambda **kwargs: put_steps.append({tag["global_steps"] for tag in kwargs["tags"]}),
    )
    monkeypatch.setattr(
        comlrl_iterative_runtime.tq,
        "kv_clear",
        lambda *, keys, partition_id: clear_sizes.append((len(keys), partition_id)),
    )
    trainer = Trainer()

    metrics = comlrl_iterative_runtime._train_madpo_pair_batches(
        trainer,
        pairs,
        agent_ids=AGENTS,
        model_ids=("actor-a", "actor-b"),
        iteration=0,
        num_train_epochs=1,
        batch_size=batch_size,
    )

    assert batch_size == 2
    assert shuffle_sizes == [5]
    assert [metric["row_count"] for metric in metrics] == [8, 8, 4]
    assert trainer.update_row_counts == [8, 8, 4]
    assert trainer.global_steps == 3
    assert trainer.step_begin_calls == trainer.step_end_calls == 3
    assert put_steps == [{1}, {2}, {3}]
    assert clear_sizes == [(8, "train"), (8, "train"), (4, "train")]


def test_runtime_replay_materialization_allows_agent_local_ties_and_truncation_collisions(monkeypatch):
    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 9

        @staticmethod
        def encode(text, add_special_tokens=False):
            del add_special_tokens
            return [1, *[ord(char) for char in text]]

    trainer = SimpleNamespace(
        tokenizer=Tokenizer(),
        global_steps=0,
        config=SimpleNamespace(
            actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(prompt_length=4, response_length=1))
        ),
    )
    record = replay_record(1)
    record = PreferenceReplayRecord(
        iteration=record.iteration,
        prompts=record.prompts,
        chosen=(record.rejected[0], record.chosen[1]),
        rejected=record.rejected,
        processed_chosen_reward=record.processed_chosen_reward,
        processed_rejected_reward=record.processed_rejected_reward,
        raw_policy_reward=record.raw_policy_reward,
        raw_comparator_reward=record.raw_comparator_reward,
        raw_candidate_rewards=record.raw_candidate_rewards,
        candidate_mean=record.candidate_mean,
        policy_provenance=record.policy_provenance,
        comparator_provenance=record.comparator_provenance,
    )
    pairs = records_to_joint_preference_pairs([record], agent_ids=AGENTS)
    captured = {}
    monkeypatch.setattr(
        comlrl_iterative_runtime.tq,
        "kv_batch_put",
        lambda **kwargs: captured.update(kwargs),
    )

    batch = comlrl_iterative_runtime._materialize_pairs_to_tq(
        trainer,
        pairs,
        agent_ids=AGENTS,
        model_ids=("actor-a", "actor-b"),
        iteration=0,
        epoch=0,
    )

    assert len(batch.keys) == 4
    assert torch.equal(captured["fields"]["responses"], torch.ones(4, 1, dtype=torch.long))
