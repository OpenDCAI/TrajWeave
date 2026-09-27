from __future__ import annotations

import json
from pathlib import Path

from trajweave.pipeline.launch import _register_verl_training_artifacts
from trajweave.runtime import ExperimentTracker
from trajweave.storage import RunStore, RunStoreConfig


def _tracker(tmp_path: Path) -> ExperimentTracker:
    store = RunStore(
        RunStoreConfig(root_dir=tmp_path, name="artifact-test"),
        recipe="test_recipe",
        canonical_recipe="test_recipe",
        mode="verl_train",
        config_path=None,
        raw_config={},
    )
    store.initialize()
    return ExperimentTracker(store, logging_config={"console": False})


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_verl_checkpoint_directories_and_manifests_are_registered(tmp_path):
    tracker = _tracker(tmp_path)
    checkpoint_root = tracker.run_dir / "checkpoints"
    step = checkpoint_root / "global_step_3"
    step.mkdir(parents=True)
    (step / "multi_actor_weight_sync.json").write_text("{}", encoding="utf-8")
    (checkpoint_root / "latest_checkpointed_iteration.txt").write_text("3", encoding="utf-8")

    _register_verl_training_artifacts(
        ("trainer.default_local_dir=${oc.env:TRAJWEAVE_RUN_DIR}/checkpoints",),
        tracker=tracker,
        launch_cwd=None,
        executed=True,
    )

    rows = _rows(tracker.run_dir / "artifacts" / "artifact_index.jsonl")
    names = {row["name"] for row in rows}
    assert "checkpoint/global_step_3" in names
    assert "checkpoint/global_step_3/multi_actor_weight_sync.json" in names
    assert "checkpoint/latest_checkpointed_iteration.txt" in names
    assert all(row["exists"] for row in rows)


def test_verl_checkpoint_registration_skips_dry_run_and_missing_paths(tmp_path):
    tracker = _tracker(tmp_path)
    overrides = (f"trainer.default_local_dir={tmp_path / 'missing'}",)

    _register_verl_training_artifacts(overrides, tracker=tracker, launch_cwd=None, executed=False)
    _register_verl_training_artifacts(overrides, tracker=tracker, launch_cwd=None, executed=True)

    index = tracker.run_dir / "artifacts" / "artifact_index.jsonl"
    assert not index.exists() or not index.read_text(encoding="utf-8").strip()
