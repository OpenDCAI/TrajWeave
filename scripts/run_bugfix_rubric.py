#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trajweave.storage.run_store import git_worktree_fingerprint  # noqa: E402


@dataclass(frozen=True)
class RubricSpec:
    rubric_id: str
    name: str
    selectors: tuple[str, ...]


@dataclass
class RubricResult:
    rubric_id: str
    name: str
    status: str
    duration_seconds: float
    evidence: dict[str, Any]


RUBRICS = (
    RubricSpec(
        "R01",
        "配置入口完整性",
        ("tests/trajweave/test_bugfix_gate_on_cpu.py::test_required_recipe_and_mode_never_fall_back",),
    ),
    RubricSpec(
        "R02",
        "Recipe 与 mode 严格匹配",
        ("tests/trajweave/test_bugfix_gate_on_cpu.py::test_every_recipe_rejects_unknown_mode",),
    ),
    RubricSpec(
        "R03",
        "训练状态真实性",
        (
            "tests/trajweave/test_bugfix_gate_on_cpu.py::test_verl_train_never_accepts_missing_disabled_or_dry_run_launch",
            "tests/trajweave/test_bugfix_gate_on_cpu.py::test_verl_train_requires_real_execution",
        ),
    ),
    RubricSpec(
        "R04",
        "Plan 与 Train 隔离",
        ("tests/trajweave/test_bugfix_gate_on_cpu.py::test_verl_plan_generates_command_and_finalizes_as_planned",),
    ),
    RubricSpec(
        "R05",
        "训练步数与数值有效性",
        (
            "tests/trajweave/test_bugfix_gate_on_cpu.py::test_training_progress_requires_positive_explicit_step_count",
            "tests/trajweave/test_runtime_safety_on_cpu.py::test_training_progress_validation_rejects_partial_success",
            "tests/trajweave/test_runtime_safety_on_cpu.py::test_training_progress_validation_rejects_non_finite_metrics",
        ),
    ),
    RubricSpec(
        "R06",
        "失败 Trajectory 隔离",
        ("tests/trainer/ppo/v2/test_replay_buffer_on_cpu.py::test_sample_rejects_failure_trajectories",),
    ),
    RubricSpec(
        "R07",
        "GRPO Group 完整性与超时",
        (
            "tests/trainer/ppo/v2/test_replay_buffer_on_cpu.py::test_sync_grpo_step_returns_complete_groups",
            "tests/trainer/ppo/v2/test_replay_buffer_on_cpu.py::test_sample_times_out_when_rollout_never_finishes",
        ),
    ),
    RubricSpec(
        "R08",
        "Worker Group 更新完整性",
        (
            "tests/trajweave/test_bugfix_gate_on_cpu.py::test_missing_trainable_worker_group_aborts_before_any_actor_update",
        ),
    ),
    RubricSpec(
        "R09",
        "Rollout 非阻塞性",
        (
            "tests/trajweave/test_bugfix_gate_on_cpu.py::test_agent_loop_background_dispatch_returns_before_rollout_finishes",
        ),
    ),
    RubricSpec(
        "R10",
        "禁止静默 Fallback",
        (
            "tests/trajweave/test_bugfix_gate_on_cpu.py::test_chat_template_failure_is_strict_unless_fallback_is_explicit",
            "tests/trajweave/test_bugfix_gate_on_cpu.py::test_tiny_torch_does_not_silently_fall_back_when_torch_import_fails",
            "tests/trajweave/test_bugfix_gate_on_cpu.py::test_tiny_torch_does_not_silently_move_cuda_requests_to_cpu",
            "tests/trajweave/test_verl_extension_hooks_on_cpu.py",
        ),
    ),
    RubricSpec(
        "R11",
        "Reward 输入与仓库配置完整性",
        (
            "tests/trajweave/test_bugfix_gate_on_cpu.py::test_ground_truth_is_required_for_training_emitters",
            "tests/trajweave/test_bugfix_gate_on_cpu.py::test_all_checked_in_configs_follow_strict_run_contract",
        ),
    ),
    RubricSpec(
        "R12",
        "全量相关回归",
        ("tests/trajweave", "tests/trainer/ppo/v2/test_replay_buffer_on_cpu.py"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="逐项执行 TrajWeave bugfix Rubric。")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "rubrics" / "bugfix-rubric.json",
        help="JSON 报告路径。",
    )
    parser.add_argument(
        "--e2e-run-dir",
        action="append",
        type=Path,
        default=[],
        help="需要审计的真实 VERL 训练 run 目录；至少提供一个，否则整体不通过。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    evidence_dir = args.output.parent / f"{args.output.stem}-evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    results = [_run_pytest_rubric(spec, evidence_dir) for spec in RUBRICS]
    results.append(_audit_e2e_runs(args.e2e_run_dir))
    source_fingerprint = git_worktree_fingerprint(REPO_ROOT)
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_output("rev-parse", "HEAD"),
        "worktree_diff_sha256": source_fingerprint,
        "status": "PASS" if all(item.status == "PASS" for item in results) else "FAIL",
        "rubrics": [asdict(item) for item in results],
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for item in results:
        print(f"{item.rubric_id} {item.status:<4} {item.name}")
    print(f"OVERALL {report['status']} -> {args.output}")
    return 0 if report["status"] == "PASS" else 1


def _run_pytest_rubric(spec: RubricSpec, evidence_dir: Path) -> RubricResult:
    junit_path = evidence_dir / f"{spec.rubric_id.lower()}.xml"
    command = [sys.executable, "-m", "pytest", "-q", *spec.selectors, f"--junitxml={junit_path}"]
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    duration = time.monotonic() - started
    counts = _junit_counts(junit_path)
    passed = (
        completed.returncode == 0
        and counts["tests"] > 0
        and counts["failures"] == 0
        and counts["errors"] == 0
        and counts["skipped"] == 0
    )
    return RubricResult(
        rubric_id=spec.rubric_id,
        name=spec.name,
        status="PASS" if passed else "FAIL",
        duration_seconds=round(duration, 3),
        evidence={
            "command": command,
            "returncode": completed.returncode,
            "junit_xml": str(junit_path),
            "counts": counts,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        },
    )


def _junit_counts(path: Path) -> dict[str, int]:
    counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    if not path.is_file():
        return counts
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    for suite in suites:
        for key in counts:
            counts[key] += int(suite.attrib.get(key, 0))
    return counts


def _audit_e2e_runs(run_dirs: list[Path]) -> RubricResult:
    started = time.monotonic()
    evidence: dict[str, Any] = {"runs": []}
    failures: list[str] = []
    multi_actor_runs = 0
    source_fingerprint = git_worktree_fingerprint(REPO_ROOT)
    if source_fingerprint is None:
        failures.append("Could not compute the current worktree source fingerprint.")
    if not run_dirs:
        failures.append("No --e2e-run-dir was provided.")
    for raw_path in run_dirs:
        run_dir = raw_path.expanduser().resolve()
        run_failures, run_evidence = _audit_one_run(run_dir, source_fingerprint=source_fingerprint)
        if run_evidence.get("native_multi_actor_training") is True:
            multi_actor_runs += 1
        evidence["runs"].append(
            {
                "run_dir": str(run_dir),
                "facts": run_evidence,
                "failures": run_failures,
            }
        )
        failures.extend(f"{run_dir}: {message}" for message in run_failures)
    if multi_actor_runs == 0:
        failures.append("No native multi-actor VERL training run was provided.")
    evidence["current_source_fingerprint"] = source_fingerprint
    evidence["native_multi_actor_runs"] = multi_actor_runs
    evidence["failures"] = failures
    return RubricResult(
        rubric_id="R13",
        name="真实训练产物审计",
        status="PASS" if not failures else "FAIL",
        duration_seconds=round(time.monotonic() - started, 3),
        evidence=evidence,
    )


def _audit_one_run(run_dir: Path, *, source_fingerprint: str | None = None) -> tuple[list[str], dict[str, Any]]:
    failures: list[str] = []
    evidence: dict[str, Any] = {}
    required_files = (
        "manifest.json",
        "status.json",
        "summary.json",
        "logs/verl_stdout.log",
        "metrics/metrics.jsonl",
        "artifacts/artifact_index.jsonl",
    )
    for relative in required_files:
        path = run_dir / relative
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f"missing or empty {relative}")

    manifest = _load_json(run_dir / "manifest.json", failures)
    manifest_fingerprint = manifest.get("worktree_diff_sha256")
    evidence["manifest_source_fingerprint"] = manifest_fingerprint
    evidence["source_fingerprint_matches"] = (
        source_fingerprint is not None and manifest_fingerprint == source_fingerprint
    )
    if not manifest_fingerprint:
        failures.append("manifest has no worktree_diff_sha256")
    elif source_fingerprint is not None and manifest_fingerprint != source_fingerprint:
        failures.append("training manifest source fingerprint does not match the current worktree")

    summary = _load_json(run_dir / "summary.json", failures)
    status_path = run_dir / "status.json"
    if status_path.is_file():
        status = _load_json(status_path, failures)
        evidence["status"] = status.get("status")
        if status.get("status") != "completed":
            failures.append(f"status is {status.get('status')!r}, expected 'completed'")

    launch = summary.get("verl_launch", {})
    if launch.get("status") != "ok" or launch.get("returncode") != 0:
        failures.append(
            "VERL launch did not finish cleanly: "
            f"status={launch.get('status')!r}, returncode={launch.get('returncode')!r}"
        )
    evidence["launch_status"] = launch.get("status")
    evidence["launch_returncode"] = launch.get("returncode")

    metrics_path = run_dir / "metrics" / "metrics.jsonl"
    metric_rows: list[dict[str, Any]] = []
    if metrics_path.is_file():
        metric_rows = _load_jsonl(metrics_path, failures)
        if not metric_rows:
            failures.append("metrics JSONL has no rows")
        elif _contains_non_finite(metric_rows):
            failures.append("metrics contain NaN or Inf")

    latest_metrics = summary.get("metric_summary", {}).get("latest", {})
    expected_steps = _expected_training_steps(launch.get("command", []))
    observed_steps = latest_metrics.get("training/global_step")
    evidence["expected_training_steps"] = expected_steps
    evidence["observed_training_steps"] = observed_steps
    if expected_steps is None:
        failures.append("launch command has no explicit positive trainer.total_training_steps")
    elif observed_steps != expected_steps:
        failures.append(f"training progress mismatch: expected {expected_steps}, observed {observed_steps!r}")

    trajectory_files = list((run_dir / "trajectories").glob("*.jsonl")) + list(
        (run_dir / "trajectories" / "online_turns").glob("*.jsonl")
    )
    if not any(path.stat().st_size > 0 for path in trajectory_files):
        failures.append("no non-empty trajectory JSONL artifact")
    trajectory_rows: list[dict[str, Any]] = []
    for path in trajectory_files:
        trajectory_rows.extend(_load_jsonl(path, failures))
    evidence["trajectory_rows"] = len(trajectory_rows)

    checkpoint_root = run_dir / "checkpoints"
    if not checkpoint_root.is_dir() or not any(checkpoint_root.glob("global_step_*")):
        failures.append("no global_step checkpoint directory")

    multi_actor = _multi_actor_metadata(summary)
    evidence["native_multi_actor_training"] = multi_actor is not None
    if multi_actor is not None:
        _audit_multi_actor_training(
            run_dir=run_dir,
            metadata=multi_actor,
            metric_rows=metric_rows,
            latest_metrics=latest_metrics,
            expected_steps=expected_steps,
            trajectory_rows=trajectory_rows,
            failures=failures,
            evidence=evidence,
        )

    artifact_path = run_dir / "artifacts" / "artifact_index.jsonl"
    if artifact_path.is_file():
        artifacts = _load_jsonl(artifact_path, failures)
        kinds = {row.get("kind") for row in artifacts if row.get("exists") is True}
        missing_kinds = {"log", "trajectory", "checkpoint"} - kinds
        if missing_kinds:
            failures.append(f"artifact index misses kinds: {sorted(missing_kinds)}")
        evidence["artifact_kinds"] = sorted(str(kind) for kind in kinds)

    error_markers = ("Traceback (most recent call last)", "RayTaskError(", "RuntimeError:")
    for path in (run_dir / "logs").glob("*.log"):
        content = path.read_text(encoding="utf-8", errors="replace")
        found = [marker for marker in error_markers if marker in content]
        if found:
            failures.append(f"{path.relative_to(run_dir)} contains fatal markers: {found}")
    return failures, evidence


def _load_json(path: Path, failures: list[str]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"cannot parse {path}: {exc}")
        return {}
    if not isinstance(value, dict):
        failures.append(f"{path} does not contain a JSON object")
        return {}
    return value


def _load_jsonl(path: Path, failures: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        failures.append(f"cannot read {path}: {exc}")
        return rows
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            failures.append(f"cannot parse {path}:{line_number}: {exc}")
            continue
        if not isinstance(value, dict):
            failures.append(f"{path}:{line_number} is not a JSON object")
            continue
        rows.append(value)
    return rows


def _expected_training_steps(command: Any) -> int | None:
    if not isinstance(command, list):
        return None
    prefix = "trainer.total_training_steps="
    for argument in command:
        if isinstance(argument, str) and argument.startswith(prefix):
            try:
                value = int(argument.removeprefix(prefix))
            except ValueError:
                return None
            return value if value > 0 else None
    return None


def _multi_actor_metadata(summary: dict[str, Any]) -> dict[str, Any] | None:
    for value in summary.values():
        if isinstance(value, dict) and value.get("native_multi_actor_training") is True:
            return value
    return None


def _metric_values(metric_rows: list[dict[str, Any]], suffix: str) -> list[Any]:
    return [row.get("value") for row in metric_rows if str(row.get("name", "")).endswith(suffix)]


def _metric_steps(metric_rows: list[dict[str, Any]], suffix: str) -> list[Any]:
    return [row.get("step") for row in metric_rows if str(row.get("name", "")).endswith(suffix)]


def _latest_metric(latest_metrics: dict[str, Any], suffix: str) -> Any:
    matches = [value for name, value in latest_metrics.items() if name.endswith(suffix)]
    return matches[0] if len(matches) == 1 else None


def _audit_multi_actor_training(
    *,
    run_dir: Path,
    metadata: dict[str, Any],
    metric_rows: list[dict[str, Any]],
    latest_metrics: dict[str, Any],
    expected_steps: int | None,
    trajectory_rows: list[dict[str, Any]],
    failures: list[str],
    evidence: dict[str, Any],
) -> None:
    groups = metadata.get("trainable_worker_groups", [])
    if not isinstance(groups, list) or len(groups) < 2 or not all(isinstance(item, str) for item in groups):
        failures.append(f"invalid multi-actor trainable groups: {groups!r}")
        return

    evidence["trainable_worker_groups"] = groups
    evidence["worker_group_metrics"] = {}
    expected_step_set = set(range(1, expected_steps + 1)) if expected_steps is not None else set()
    for group in groups:
        sample_values = _metric_values(metric_rows, f"/actor_groups/{group}/samples")
        update_values = _metric_values(metric_rows, f"/actor_groups/{group}/updated")
        sample_steps = _metric_steps(metric_rows, f"/actor_groups/{group}/samples")
        update_steps = _metric_steps(metric_rows, f"/actor_groups/{group}/updated")
        evidence["worker_group_metrics"][group] = {
            "sample_values": sample_values,
            "sample_steps": sample_steps,
            "update_values": update_values,
            "update_steps": update_steps,
        }
        if expected_steps is not None and len(sample_values) != expected_steps:
            failures.append(
                f"worker group {group!r} sample metric count is {len(sample_values)}, expected {expected_steps}"
            )
        if not sample_values or any(not isinstance(value, int | float) or value <= 0 for value in sample_values):
            failures.append(f"worker group {group!r} has missing or non-positive samples: {sample_values}")
        if expected_steps is not None and set(sample_steps) != expected_step_set:
            failures.append(
                f"worker group {group!r} sample steps are {sample_steps}, expected {sorted(expected_step_set)}"
            )
        if expected_steps is not None and len(update_values) != expected_steps:
            failures.append(
                f"worker group {group!r} update metric count is {len(update_values)}, expected {expected_steps}"
            )
        if not update_values or any(value != 1 for value in update_values):
            failures.append(f"worker group {group!r} was not updated on every step: {update_values}")
        if expected_steps is not None and set(update_steps) != expected_step_set:
            failures.append(
                f"worker group {group!r} update steps are {update_steps}, expected {sorted(expected_step_set)}"
            )

    missing_values = _metric_values(metric_rows, "/actor_groups/missing_trainable")
    if not missing_values or any(value != 0 for value in missing_values):
        failures.append(f"missing trainable worker groups were reported: {missing_values}")
    if _latest_metric(latest_metrics, "/actor_groups/updated") != len(groups):
        failures.append("latest actor group update count does not equal trainable group count")
    if _latest_metric(latest_metrics, "/actor_groups/total") != len(groups):
        failures.append("latest actor group total does not equal trainable group count")

    if expected_steps is not None:
        actor_root = run_dir / "checkpoints" / f"global_step_{expected_steps}" / "actors"
        for group in groups:
            group_dir = actor_root / group
            if not list(group_dir.glob("model_*.pt")):
                failures.append(f"worker group {group!r} has no model checkpoint")
            if not list(group_dir.glob("optim_*.pt")):
                failures.append(f"worker group {group!r} has no optimizer checkpoint")

    required_fields = {"recipe", "uid", "turn_id", "agent_id", "worker_group", "reward_score"}
    malformed = [index for index, row in enumerate(trajectory_rows) if not required_fields <= row.keys()]
    if malformed:
        failures.append(f"trajectory rows miss required fields at indexes: {malformed[:10]}")
    trajectory_groups = {row.get("worker_group") for row in trajectory_rows}
    evidence["trajectory_worker_groups"] = sorted(str(group) for group in trajectory_groups)
    missing_trajectory_groups = set(groups) - trajectory_groups
    if missing_trajectory_groups:
        failures.append(f"trajectory misses worker groups: {sorted(missing_trajectory_groups)}")
    rewards = [row.get("reward_score") for row in trajectory_rows]
    if not rewards or any(not isinstance(value, int | float) or not math.isfinite(value) for value in rewards):
        failures.append("trajectory rewards are missing or non-finite")


def _contains_non_finite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, str):
        return value.strip().lower() in {"nan", "inf", "+inf", "-inf", "infinity", "-infinity"}
    if isinstance(value, dict):
        return any(_contains_non_finite(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_non_finite(item) for item in value)
    return False


def _git_output(*args: str) -> str | None:
    completed = subprocess.run(["git", *args], cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    return completed.stdout.strip() or None


if __name__ == "__main__":
    raise SystemExit(main())
