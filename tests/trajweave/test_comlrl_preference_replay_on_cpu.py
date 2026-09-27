import json
import os
import shutil
from collections import Counter
from dataclasses import replace

import pytest

from trajweave.storage.preference_replay import (
    DuplicateIterationError,
    PolicySnapshotStore,
    PreferenceReplayRecord,
    PreferenceReplaySchemaError,
    PreferenceReplayStore,
    canonical_replay_mode,
)


def record(iteration: int, index: int = 0, *, provenance=None) -> PreferenceReplayRecord:
    return PreferenceReplayRecord(
        iteration=iteration,
        prompts=(f"prompt-{iteration}-{index}-a", f"prompt-{iteration}-{index}-b"),
        chosen=(f"chosen-{iteration}-{index}-a", f"chosen-{iteration}-{index}-b"),
        rejected=(f"rejected-{iteration}-{index}-a", f"rejected-{iteration}-{index}-b"),
        processed_chosen_reward=1.0 + index,
        processed_rejected_reward=0.25,
        raw_policy_reward=3.0 + index,
        raw_comparator_reward=-2.0 - index,
        raw_candidate_rewards=(3.0 + index, -2.0 - index),
        candidate_mean=0.5,
        policy_provenance=provenance or {"policy": "current", "iteration": iteration},
        comparator_provenance={"policy": "history", "snapshot": f"iteration_{iteration:04d}"},
    )


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [("latest", "current"), ("recent_k", "nearest_k"), ("all_history", "all_history"), ("td-lambda", "lambda_decay")],
)
def test_replay_mode_aliases_match_upstream(alias, canonical):
    assert canonical_replay_mode(alias) == canonical


def test_record_json_round_trip_keeps_processed_and_raw_rewards_separate():
    value = record(2, 4)
    payload = value.to_dict()
    restored = PreferenceReplayRecord.from_dict(payload)

    assert restored == value
    assert value.pair_id == record(2, 4).pair_id
    assert payload["processed_rewards"] == {"chosen": 5.0, "rejected": 0.25}
    assert payload["raw_rewards"] == {"policy": 7.0, "comparator": -6.0, "candidates": [7.0, -6.0]}
    assert "raw_reward" not in payload


def test_replay_write_is_atomic_redacted_and_manifested(tmp_path, monkeypatch):
    store = PreferenceReplayStore(tmp_path)
    calls = []
    real_replace = os.replace

    def tracked_replace(source, target):
        calls.append((source, target))
        real_replace(source, target)

    monkeypatch.setattr(os, "replace", tracked_replace)
    secret_record = record(0, provenance={"api_key": "do-not-persist", "policy": "api"})
    shard = store.write_iteration(0, [secret_record])

    assert shard.path == tmp_path / "iteration_0000.json"
    assert calls and all(str(source).endswith(".tmp") for source, _ in calls)
    assert not list(tmp_path.glob("*.tmp"))
    persisted = shard.path.read_text()
    assert "do-not-persist" not in persisted
    assert "[REDACTED]" in persisted
    assert store.manifest["available_iterations"] == [0]
    assert json.loads((tmp_path / "manifest.json").read_text())["shards"][0]["num_records"] == 1


def test_current_nearest_all_history_and_seeded_selection(tmp_path):
    store = PreferenceReplayStore(tmp_path)
    for iteration in range(3):
        store.write_iteration(iteration, [record(iteration, index) for index in range(3)])

    assert {item.iteration for item in store.select("current")} == {2}
    nearest = store.select("nearest_k", nearest_k=2, sample_size=80, seed=9)
    assert {item.iteration for item in nearest} == {1, 2}
    first = store.select("all_history", sample_size=40, seed=123)
    second = store.select("all_history", sample_size=40, seed=123)
    assert [item.pair_id for item in first] == [item.pair_id for item in second]
    assert {item.iteration for item in first} == {0, 1, 2}


def test_shard_uniform_sampling_is_not_record_uniform(tmp_path):
    store = PreferenceReplayStore(tmp_path)
    store.write_iteration(0, [record(0, 0)])
    store.write_iteration(1, [record(1, index) for index in range(9)])

    sampled = store.select("all_history", sample_size=2000, seed=7)
    counts = Counter(item.iteration for item in sampled)

    assert 850 < counts[0] < 1150
    assert 850 < counts[1] < 1150


def test_lambda_zero_uses_only_latest_and_lambda_one_is_uniform(tmp_path):
    store = PreferenceReplayStore(tmp_path)
    for iteration in range(3):
        store.write_iteration(iteration, [record(iteration)])

    latest = store.select("lambda_decay", replay_lambda=0.0, sample_size=100, seed=1)
    assert {item.iteration for item in latest} == {2}

    uniform = store.select("lambda_decay", replay_lambda=1.0, sample_size=3000, seed=2)
    counts = Counter(item.iteration for item in uniform)
    assert all(850 < counts[iteration] < 1150 for iteration in range(3))


def test_replacement_cycles_records_and_no_replacement_validates_capacity(tmp_path):
    store = PreferenceReplayStore(tmp_path)
    values = [record(0, 0), record(0, 1)]
    store.write_iteration(0, values)

    sampled = store.select("current", sample_size=5, seed=4, replacement=True)
    counts = Counter(item.pair_id for item in sampled)
    assert sorted(counts.values()) == [2, 3]

    without_replacement = store.select("current", sample_size=2, seed=4, replacement=False)
    assert {item.pair_id for item in without_replacement} == {item.pair_id for item in values}
    with pytest.raises(ValueError, match="exceeds available"):
        store.select("current", sample_size=3, replacement=False)


def test_restart_scans_existing_shards_and_duplicate_iteration_is_explicit(tmp_path):
    first = PreferenceReplayStore(tmp_path)
    first.write_iteration(0, [record(0)])
    first.write_iteration(2, [record(2)])

    restarted = PreferenceReplayStore(tmp_path)
    assert restarted.available_iterations == (0, 2)
    assert restarted.load_iteration(2)[0].iteration == 2
    with pytest.raises(DuplicateIterationError, match="already exists"):
        restarted.write_iteration(2, [record(2, 1)])


def test_corrupt_schema_and_duplicate_files_fail_during_scan(tmp_path):
    corrupt = tmp_path / "corrupt"
    corrupt.mkdir()
    (corrupt / "iteration_0000.json").write_text('{"schema_version": 99, "iteration": 0, "records": []}')
    with pytest.raises(PreferenceReplaySchemaError, match="schema_version"):
        PreferenceReplayStore(corrupt)

    duplicate = tmp_path / "duplicate"
    store = PreferenceReplayStore(duplicate)
    store.write_iteration(1, [record(1)])
    shutil.copy2(duplicate / "iteration_0001.json", duplicate / "iteration_1.json")
    with pytest.raises(DuplicateIterationError, match="duplicate preference replay files"):
        PreferenceReplayStore(duplicate)


def test_record_schema_damage_and_parameter_validation_are_clear(tmp_path):
    store = PreferenceReplayStore(tmp_path)
    store.write_iteration(0, [record(0)])
    payload = json.loads((tmp_path / "iteration_0000.json").read_text())
    payload["records"][0]["raw_rewards"] = "ambiguous"
    (tmp_path / "iteration_0000.json").write_text(json.dumps(payload))
    with pytest.raises(PreferenceReplaySchemaError, match="raw_rewards"):
        PreferenceReplayStore(tmp_path)

    clean = PreferenceReplayStore(tmp_path / "clean")
    clean.write_iteration(0, [record(0)])
    with pytest.raises(ValueError, match="nearest_k must be"):
        clean.select("nearest_k", nearest_k=0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        clean.select("lambda_decay", replay_lambda=1.1)
    with pytest.raises(ValueError, match="only valid"):
        clean.select("all_history", replay_lambda=0.5)


def test_snapshot_store_paths_manifest_and_history_resolution(tmp_path):
    store = PolicySnapshotStore(tmp_path)
    initial = store.register_initial({"source": "base", "api_key": "snapshot-secret"})
    iteration_zero = store.register_iteration(0, {"step": 10})
    iteration_three = store.register_iteration(3, {"step": 40})

    assert initial == tmp_path / "initial"
    assert iteration_zero == tmp_path / "iteration_0000"
    assert iteration_three == tmp_path / "iteration_0003"
    assert store.resolve_history(0, history_k=1) == initial
    assert store.resolve_history(1, history_k=1) == iteration_zero
    assert store.resolve_history(3, history_k=1) == iteration_zero
    assert store.resolve_history(4, history_k=1) == iteration_three
    assert "snapshot-secret" not in (tmp_path / "manifest.json").read_text()

    restarted = PolicySnapshotStore(tmp_path)
    assert restarted.available_iterations == (0, 3)
    assert restarted.metadata_for(3) == {"step": 40}


def test_snapshot_store_atomic_commit_only_manifests_complete_directories(tmp_path):
    store = PolicySnapshotStore(tmp_path)

    initial = store.commit_initial(
        lambda temporary: ((temporary / "model.bin").write_bytes(b"initial"), {"source": "base"})[1]
    )
    iteration = store.commit_iteration(
        0,
        lambda temporary: ((temporary / "model.bin").write_bytes(b"iteration"), {"step": 1})[1],
    )

    assert initial == tmp_path / "initial"
    assert iteration == tmp_path / "iteration_0000"
    assert store.metadata_for(None) == {"source": "base"}
    assert store.metadata_for(0) == {"step": 1}
    assert store.resolve_history(1) == iteration
    assert not list(tmp_path.glob(".*.tmp"))


def test_snapshot_store_failed_or_orphaned_writes_are_never_history_visible(tmp_path):
    store = PolicySnapshotStore(tmp_path)
    store.commit_initial(lambda temporary: ((temporary / "model.bin").write_bytes(b"initial"), None)[1])

    def fail_after_write(temporary):
        (temporary / "partial.bin").write_bytes(b"partial")
        raise RuntimeError("snapshot interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        store.commit_iteration(0, fail_after_write)
    assert store.available_iterations == ()
    assert not (tmp_path / "iteration_0000").exists()
    assert PolicySnapshotStore(tmp_path).resolve_history(1) == tmp_path / "initial"

    orphan = tmp_path / "iteration_0000"
    orphan.mkdir()
    (orphan / "partial.bin").write_bytes(b"partial")
    with pytest.raises(PreferenceReplaySchemaError, match="uncommitted"):
        PolicySnapshotStore(tmp_path)


def test_snapshot_store_rejects_empty_atomic_commit(tmp_path):
    store = PolicySnapshotStore(tmp_path)
    with pytest.raises(RuntimeError, match="empty directory"):
        store.commit_initial(lambda _temporary: None)
    assert not store.initial_path.exists()


def test_snapshot_store_only_manages_paths_not_model_files(tmp_path):
    store = PolicySnapshotStore(tmp_path)
    path = store.create_initial()
    assert path.is_dir()
    assert list(path.iterdir()) == []
    iteration_path = store.create_iteration(5)
    assert iteration_path.is_dir()
    assert list(iteration_path.iterdir()) == []

    with pytest.raises(ValueError, match="history_k"):
        store.resolve_history(5, history_k=0)


def test_custom_pair_id_survives_round_trip():
    value = replace(record(0), pair_id="external-stable-id")
    assert PreferenceReplayRecord.from_dict(value.to_dict()).pair_id == "external-stable-id"
