from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from trajweave.backends.verl.launcher import VerlTrainerLaunchConfig, VerlTrainerLauncher
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

    launch_overrides = overrides if overrides is not None else tuple(str(item) for item in verl_cfg.get("overrides", []))
    if tracker is not None:
        launch_overrides = _with_runtime_overrides(launch_overrides, tracker=tracker)
    stdout_path = verl_cfg.get("stdout_path")
    stderr_path = verl_cfg.get("stderr_path")
    if tracker is not None and bool(verl_cfg.get("execute", False)):
        stdout_path = stdout_path or str(tracker.run_dir / "logs" / "verl_stdout.log")
        stderr_path = stderr_path or str(tracker.run_dir / "logs" / "verl_stderr.log")
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
    if tracker is not None and not command_file:
        command_file = str(tracker.artifacts.path("run_verl_ppo.sh"))
    if command_file:
        command_path = str(launcher.write_command_file(command_file))
        output["verl_command_file"] = command_path
        if tracker is not None:
            tracker.artifacts.copy_file(name="run_verl_ppo.sh", source=command_path, kind="command")
    output["verl_launch"] = launcher.run()
    if tracker is not None:
        tracker.log_verl_result(output["verl_launch"])
        _register_verl_training_artifacts(
            launch_overrides,
            tracker=tracker,
            launch_cwd=launch_config.cwd,
            executed=launch_config.execute,
        )


def _with_runtime_env(env: dict[str, str], *, tracker: ExperimentTracker | None) -> dict[str, str]:
    if tracker is None:
        return env
    return {**env, "TRAJWEAVE_RUN_ID": tracker.run_id, "TRAJWEAVE_RUN_DIR": str(tracker.run_dir)}


def _with_runtime_overrides(overrides: tuple[str, ...], *, tracker: ExperimentTracker) -> tuple[str, ...]:
    existing_keys = {item.split("=", 1)[0].lstrip("+") for item in overrides if "=" in item}
    additions = []
    if "trajweave.run_id" not in existing_keys:
        additions.append(f"+trajweave.run_id={tracker.run_id}")
    if "trajweave.run_dir" not in existing_keys:
        additions.append(f"+trajweave.run_dir={tracker.run_dir}")
    if "trajweave.capture_online_turns" not in existing_keys:
        additions.append("+trajweave.capture_online_turns=true")
    return (*overrides, *additions)


def _register_verl_training_artifacts(
    overrides: tuple[str, ...],
    *,
    tracker: ExperimentTracker,
    launch_cwd: str | None,
    executed: bool,
) -> None:
    """Index VERL checkpoints in-place after a real launch.

    Checkpoint shards can be large, so the artifact store records their
    directories and manifests without copying them into the run directory.
    """
    if not executed:
        return
    configured = _override_value(overrides, "trainer.default_local_dir")
    if not configured:
        return
    expanded = configured.replace("${oc.env:TRAJWEAVE_RUN_DIR}", str(tracker.run_dir))
    expanded = os.path.expandvars(expanded)
    checkpoint_root = Path(expanded)
    if not checkpoint_root.is_absolute():
        checkpoint_root = Path(launch_cwd or Path.cwd()) / checkpoint_root
    checkpoint_root = checkpoint_root.resolve()
    if not checkpoint_root.exists():
        return
    for checkpoint_dir in sorted(path for path in checkpoint_root.glob("global_step_*") if path.is_dir()):
        tracker.log_artifact(
            name=f"checkpoint/{checkpoint_dir.name}",
            path=checkpoint_dir,
            kind="checkpoint",
            metadata={"root": str(checkpoint_root)},
        )
        manifest = checkpoint_dir / "multi_actor_weight_sync.json"
        if manifest.is_file():
            tracker.log_artifact(
                name=f"checkpoint/{checkpoint_dir.name}/multi_actor_weight_sync.json",
                path=manifest,
                kind="checkpoint_manifest",
            )
    latest = checkpoint_root / "latest_checkpointed_iteration.txt"
    if latest.is_file():
        tracker.log_artifact(
            name="checkpoint/latest_checkpointed_iteration.txt",
            path=latest,
            kind="checkpoint_pointer",
        )


def _override_value(overrides: tuple[str, ...], key: str) -> str | None:
    value = None
    for item in overrides:
        if "=" not in item:
            continue
        item_key, item_value = item.split("=", 1)
        if item_key.lstrip("+") == key:
            value = item_value
    return value
