from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from trajweave.backends.verl.launcher import VerlTrainerLaunchConfig, VerlTrainerLauncher
from trajweave.metrics import parse_verl_console_metrics
from trajweave.runtime import ExperimentTracker


def maybe_run_verl_launch(
    config: dict[str, Any],
    output: dict[str, Any],
    *,
    overrides: tuple[str, ...] | None = None,
    default_enabled: bool = False,
    default_module: str = "verl.trainer.main_ppo",
    tracker: ExperimentTracker | None = None,
) -> None:
    verl_cfg = config.get("verl", {})
    if not verl_cfg.get("enabled", default_enabled):
        if default_enabled:
            output["verl_launch"] = {"status": "disabled"}
        return

    launch_overrides = (
        overrides if overrides is not None else tuple(str(item) for item in verl_cfg.get("overrides", []))
    )
    if tracker is not None:
        launch_overrides = _with_runtime_overrides(
            launch_overrides,
            tracker=tracker,
            capture_native_rollouts=bool(verl_cfg.get("capture_native_rollouts", False)),
        )
    stdout_path = verl_cfg.get("stdout_path")
    stderr_path = verl_cfg.get("stderr_path")
    if tracker is not None and bool(verl_cfg.get("execute", False)):
        stdout_path = str(tracker.run_dir / "logs" / "verl_stdout.log")
        stderr_path = str(tracker.run_dir / "logs" / "verl_stderr.log")
    launch_config = VerlTrainerLaunchConfig(
        python=str(verl_cfg.get("python", VerlTrainerLaunchConfig.python)),
        module=str(verl_cfg.get("module", default_module)),
        overrides=launch_overrides,
        env=_with_runtime_env(dict(verl_cfg.get("env", {})), tracker=tracker),
        cwd=verl_cfg.get("cwd"),
        execute=bool(verl_cfg.get("execute", False)),
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    launcher = VerlTrainerLauncher(launch_config)
    command_file = verl_cfg.get("command_file")
    if tracker is not None:
        command_file = str(tracker.artifacts.path("run_verl_ppo.sh"))
    if command_file:
        command_path = str(launcher.write_command_file(command_file))
        output["verl_command_file"] = command_path
        if tracker is not None:
            tracker.artifacts.copy_file(name="run_verl_ppo.sh", source=command_path, kind="command")
    launch_result = launcher.run()
    _validate_training_progress(launch_result, launch_overrides)
    output["verl_launch"] = launch_result
    if tracker is not None:
        tracker.log_verl_result(output["verl_launch"])


def _with_runtime_env(env: dict[str, str], *, tracker: ExperimentTracker | None) -> dict[str, str]:
    if tracker is None:
        return env
    return {**env, "TRAJWEAVE_RUN_ID": tracker.run_id, "TRAJWEAVE_RUN_DIR": str(tracker.run_dir)}


def _with_runtime_overrides(
    overrides: tuple[str, ...],
    *,
    tracker: ExperimentTracker,
    capture_native_rollouts: bool = False,
) -> tuple[str, ...]:
    forced_values = {
        "trajweave.run_id": tracker.run_id,
        "trajweave.run_dir": str(tracker.run_dir),
        "trainer.default_local_dir": str(tracker.run_dir / "checkpoints"),
    }
    if capture_native_rollouts:
        forced_values.update(
            {
                "trainer.rollout_data_dir": str(tracker.run_dir / "trajectories" / "verl_rollouts"),
                "trainer.validation_data_dir": str(tracker.run_dir / "trajectories" / "verl_validation"),
            }
        )
    filtered = tuple(item for item in overrides if _override_key(item) not in forced_values)
    existing_keys = {_override_key(item) for item in filtered}
    additions = []
    for key, value in forced_values.items():
        prefix = "+" if key.startswith("trajweave.") else ""
        additions.append(f"{prefix}{key}={json.dumps(value, ensure_ascii=False)}")
    if "trajweave.capture_online_turns" not in existing_keys:
        additions.append("+trajweave.capture_online_turns=true")
    return (*filtered, *additions)


def _override_key(override: str) -> str | None:
    if "=" not in override:
        return None
    return override.split("=", 1)[0].lstrip("+")


def _validate_training_progress(result: dict[str, Any], overrides: tuple[str, ...]) -> None:
    """拒绝“进程返回 0，但实际训练步数不足”的假成功。"""

    if result.get("status") != "ok":
        return
    expected = _integer_override(overrides, "trainer.total_training_steps")
    if expected is None or expected <= 0:
        return
    result["expected_training_steps"] = expected
    stdout_path = result.get("stdout_path")
    if not stdout_path or not Path(str(stdout_path)).is_file():
        result["status"] = "failed"
        result["validation_error"] = "VERL completed without a readable stdout log for training validation."
        return
    stdout = Path(str(stdout_path)).read_text(encoding="utf-8", errors="replace")
    events = parse_verl_console_metrics(stdout, run_id="progress-validation")
    observed_steps = [event.step for event in events if event.step is not None]
    observed = max(observed_steps, default=0)
    result["observed_training_steps"] = observed
    if observed < expected:
        result["status"] = "failed"
        result["validation_error"] = (
            f"VERL exited successfully but completed only {observed}/{expected} requested training steps. "
            "Check trainer.total_epochs and dataloader length."
        )
        return
    non_finite = sorted(
        {event.name for event in events if isinstance(event.value, float) and not math.isfinite(event.value)}
    )
    if non_finite:
        result["status"] = "failed"
        result["validation_error"] = f"VERL emitted non-finite training metrics: {non_finite}."


def _integer_override(overrides: tuple[str, ...], key: str) -> int | None:
    value: str | None = None
    for override in overrides:
        if _override_key(override) == key:
            value = override.split("=", 1)[1]
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None
