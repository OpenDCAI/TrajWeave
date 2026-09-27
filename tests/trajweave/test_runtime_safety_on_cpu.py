import json
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from trajweave.backends.verl.launcher import VerlTrainerLaunchConfig, VerlTrainerLauncher
from trajweave.pipeline.launch import _validate_training_progress, _with_runtime_env, _with_runtime_overrides
from trajweave.runner import _enforce_runtime_identity
from trajweave.runtime import StructuredLogger
from trajweave.storage import ArtifactStore, RunStore, RunStoreConfig


def make_run_store(
    root_dir: Path,
    *,
    run_id: str = "run-test",
    resume: bool = False,
    raw_config: dict | None = None,
) -> RunStore:
    return RunStore(
        RunStoreConfig(root_dir=str(root_dir), run_id=run_id, resume=resume),
        recipe="doctor_mas_math",
        canonical_recipe="doctor_mas_math",
        mode="smoke",
        config_path=None,
        raw_config=raw_config or {},
    )


@pytest.mark.parametrize(
    "run_id",
    ["", ".", "..", "../outside", "nested/../outside", "/tmp/absolute", r"C:\absolute"],
)
def test_run_id_rejects_root_absolute_and_parent_paths(tmp_path: Path, run_id: str):
    with pytest.raises((TypeError, ValueError), match="run.id"):
        make_run_store(tmp_path / "runs", run_id=run_id)


def test_run_id_rejects_symlink_escape(tmp_path: Path):
    root_dir = tmp_path / "runs"
    outside = tmp_path / "outside"
    root_dir.mkdir()
    outside.mkdir()
    (root_dir / "escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="root_dir"):
        make_run_store(root_dir, run_id="escape/run")


def test_nested_run_id_stays_under_root_dir(tmp_path: Path):
    root_dir = tmp_path / "runs"
    store = make_run_store(root_dir, run_id="group/run")

    store.initialize()

    assert store.run_dir == (root_dir / "group" / "run").resolve()
    assert store.run_dir.is_relative_to(root_dir.resolve())


def test_run_store_cannot_release_a_lock_it_does_not_own(tmp_path: Path):
    root_dir = tmp_path / "runs"
    owner = make_run_store(root_dir)
    contender = make_run_store(root_dir)
    owner.initialize()

    with pytest.raises(RuntimeError, match="already active"):
        contender.initialize()
    contender.release_lock()

    assert owner._lock_path.is_file()
    owner.release_lock()
    assert not owner._lock_path.exists()


def test_run_resume_is_explicitly_rejected_without_backend_recovery(tmp_path: Path):
    store = make_run_store(tmp_path / "runs", resume=True)

    with pytest.raises(ValueError, match="run.resume=true is not supported"):
        store.initialize()

    assert not store.run_dir.exists()


def test_runtime_identity_overrides_conflicting_child_values(tmp_path: Path):
    tracker = SimpleNamespace(run_id="parent-run", run_dir=(tmp_path / "parent-run").resolve())
    overrides = (
        "+trajweave.run_id=child-run",
        "++trajweave.run_dir=/tmp/child-run",
        "trainer.use_v1=true",
    )

    effective = _with_runtime_overrides(overrides, tracker=tracker)
    env = _with_runtime_env(
        {"TRAJWEAVE_RUN_ID": "child-run", "TRAJWEAVE_RUN_DIR": "/tmp/child-run"},
        tracker=tracker,
    )

    assert '+trajweave.run_id="parent-run"' in effective
    assert f'+trajweave.run_dir="{tracker.run_dir}"' in effective
    assert all("child-run" not in item for item in effective)
    assert env["TRAJWEAVE_RUN_ID"] == "parent-run"
    assert env["TRAJWEAVE_RUN_DIR"] == str(tracker.run_dir)


def test_training_progress_validation_rejects_partial_success(tmp_path: Path):
    stdout_path = tmp_path / "stdout.log"
    stdout_path.write_text("step:1 - actor/loss:0.1 - training/global_step:1\n", encoding="utf-8")
    result = {"status": "ok", "returncode": 0, "stdout_path": str(stdout_path)}

    _validate_training_progress(result, ("trainer.total_training_steps=2",))

    assert result["status"] == "failed"
    assert result["observed_training_steps"] == 1
    assert result["expected_training_steps"] == 2
    assert "1/2" in result["validation_error"]


def test_training_progress_validation_rejects_non_finite_metrics(tmp_path: Path):
    stdout_path = tmp_path / "stdout.log"
    stdout_path.write_text(
        "step:1 - training/global_step:1 - actor/loss:nan - actor/grad_norm:inf\n",
        encoding="utf-8",
    )
    result = {"status": "ok", "returncode": 0, "stdout_path": str(stdout_path)}

    _validate_training_progress(result, ("trainer.total_training_steps=1",))

    assert result["status"] == "failed"
    assert "actor/grad_norm" in result["validation_error"]
    assert "actor/loss" in result["validation_error"]


def test_training_progress_validation_requires_a_readable_log(tmp_path: Path):
    result = {"status": "ok", "returncode": 0, "stdout_path": str(tmp_path / "missing.log")}

    _validate_training_progress(result, ("trainer.total_training_steps=1",))

    assert result["status"] == "failed"
    assert "readable stdout" in result["validation_error"]


def test_non_finite_values_are_persisted_as_standard_json_strings(tmp_path: Path):
    from trajweave.storage.jsonl import JsonlWriter

    path = tmp_path / "metrics.jsonl"
    JsonlWriter(path).write({"nan": float("nan"), "positive_inf": float("inf")})

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "nan": "nan",
        "positive_inf": "inf",
    }


def test_recipe_output_cannot_claim_another_run_identity(tmp_path: Path):
    store = make_run_store(tmp_path / "runs")

    assert _enforce_runtime_identity({}, run_store=store) == {
        "run_id": store.run_id,
        "run_dir": str(store.run_dir),
    }
    with pytest.raises(RuntimeError, match="does not match parent run_id"):
        _enforce_runtime_identity({"run_id": "child-run"}, run_store=store)
    with pytest.raises(RuntimeError, match="does not match parent run_dir"):
        _enforce_runtime_identity({"run_dir": str(tmp_path / "child-run")}, run_store=store)


def test_config_and_structured_logs_redact_common_secrets(tmp_path: Path):
    secret_values = ["config-secret", "url-secret", "log-secret", "bearer-secret", "payload-secret"]
    store = make_run_store(
        tmp_path / "runs",
        raw_config={
            "backend": {"client_secret_key": secret_values[0], "max_new_tokens": 8},
            "endpoint": f"https://user:{secret_values[1]}@example.test/v1",
        },
    )
    store.initialize()
    logger = StructuredLogger(store.run_id, store.run_dir, console=False)

    logger.info(
        "secret_event",
        f"password={secret_values[2]} Authorization: Bearer {secret_values[3]}",
        {"access_token": secret_values[4], "tokenizer_path": "safe-tokenizer"},
    )

    config_text = (store.run_dir / "config.yaml").read_text(encoding="utf-8")
    event_text = (store.run_dir / "logs" / "events.jsonl").read_text(encoding="utf-8")
    console_text = (store.run_dir / "logs" / "console.log").read_text(encoding="utf-8")
    persisted = config_text + event_text + console_text
    assert all(secret not in persisted for secret in secret_values)
    assert "[REDACTED]" in persisted
    assert "safe-tokenizer" in event_text
    assert stat.S_IMODE((store.run_dir / "config.yaml").stat().st_mode) == 0o600
    assert stat.S_IMODE((store.run_dir / "logs" / "events.jsonl").stat().st_mode) == 0o600
    assert stat.S_IMODE((store.run_dir / "logs" / "console.log").stat().st_mode) == 0o600


def test_artifact_index_skips_missing_checkpoint_and_asset_paths(tmp_path: Path):
    store = ArtifactStore(tmp_path / "run")

    store.register(name="checkpoint.pt", path=tmp_path / "missing.pt", kind="checkpoint")
    store.register(name="task_family", path="math", kind="prepared_asset")
    assert not store.index_path.exists()

    artifact = tmp_path / "real.pt"
    artifact.write_text("checkpoint", encoding="utf-8")
    store.register(name="checkpoint.pt", path=artifact, kind="checkpoint")
    rows = [json.loads(line) for line in store.index_path.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {
            "name": "checkpoint.pt",
            "kind": "checkpoint",
            "path": str(artifact.resolve()),
            "exists": True,
            "metadata": {},
        }
    ]


def test_verl_launcher_streams_redacted_output_and_tightens_permissions(tmp_path: Path):
    module_path = tmp_path / "fake_verl.py"
    module_path.write_text(
        "import sys\n"
        "print('step:1 - actor/pg_loss:0.25')\n"
        "print('api_key=stdout-secret')\n"
        "print('x' * 10000)\n"
        "print('password=stderr-secret', file=sys.stderr)\n",
        encoding="utf-8",
    )
    stdout_path = tmp_path / "logs" / "stdout.log"
    stderr_path = tmp_path / "logs" / "stderr.log"
    launcher = VerlTrainerLauncher(
        VerlTrainerLaunchConfig(
            python=sys.executable,
            module="fake_verl",
            cwd=str(tmp_path),
            execute=True,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
        )
    )
    command_path = launcher.write_command_file(tmp_path / "run.sh")

    result = launcher.run()

    stdout_text = stdout_path.read_text(encoding="utf-8")
    stderr_text = stderr_path.read_text(encoding="utf-8")
    assert result["status"] == "ok"
    assert result["returncode"] == 0
    assert len(result["stdout"]) <= 4000
    assert "stdout-secret" not in stdout_text
    assert "stderr-secret" not in stderr_text
    assert "[REDACTED]" in stdout_text
    assert "[REDACTED]" in stderr_text
    assert stat.S_IMODE(command_path.stat().st_mode) == 0o700
    assert stat.S_IMODE(stdout_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(stderr_path.stat().st_mode) == 0o600
