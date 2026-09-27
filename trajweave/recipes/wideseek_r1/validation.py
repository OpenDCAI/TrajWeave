from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from trajweave.metrics import parse_verl_console_metrics


def validate_wideseek_training_result(output: dict[str, Any], *, run_dir: Path, mode: str) -> None:
    """Reject successful processes that did not perform a real WideSeek update."""

    if mode != "verl_train":
        return
    launch = output.get("verl_launch")
    if not isinstance(launch, dict) or launch.get("status") != "ok":
        return
    stdout_path = launch.get("stdout_path")
    if not stdout_path or not Path(str(stdout_path)).is_file():
        _fail_training_validation(launch, "WideSeek-R1 training produced no readable VERL stdout log.")
        return

    stdout = Path(str(stdout_path)).read_text(encoding="utf-8", errors="replace")
    events = parse_verl_console_metrics(stdout, run_id="wideseek-training-validation")
    by_name: dict[str, list[float]] = {}
    for event in events:
        if isinstance(event.value, (int, float)) and math.isfinite(float(event.value)):
            by_name.setdefault(event.name, []).append(float(event.value))

    errors: list[str] = []
    if not by_name.get("trajweave/wideseek_r1/loss_scale_mean"):
        errors.append("the WideSeek-R1 dual-level reweighting hook emitted no metrics")
    if max(by_name.get("trajweave/wideseek_r1/active_agent_instances", [0.0])) < 2.0:
        errors.append("fewer than one lead plus one subagent reached the Actor batch")
    advantages = by_name.get("critic/advantages/min", []) + by_name.get("critic/advantages/max", [])
    if not any(abs(value) > 1e-12 for value in advantages):
        errors.append("all trajectory advantages were zero")
    if not any(value > 1e-12 for value in by_name.get("actor/grad_norm", [])):
        errors.append("the Actor gradient norm stayed zero")

    expected_steps = int(launch.get("expected_training_steps") or 0)
    if expected_steps >= 2:
        policy_versions = _online_policy_versions(run_dir)
        if len(policy_versions) < 2:
            errors.append("online trajectories do not show two successive rollout policy versions")
    if errors:
        _fail_training_validation(launch, "WideSeek-R1 integration validation failed: " + "; ".join(errors) + ".")


def _online_policy_versions(run_dir: Path) -> set[int]:
    versions: set[int] = set()
    for path in (Path(run_dir) / "trajectories" / "online_turns").glob("*.jsonl"):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                    value = (row.get("metadata") or {}).get("policy_version")
                    if value is not None:
                        versions.add(int(value))
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
    return versions


def _fail_training_validation(launch: dict[str, Any], message: str) -> None:
    launch["status"] = "failed"
    launch["validation_error"] = message
