from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from trajweave.core.trajectory import MultiAgentTrajectory, TrainingSample
from trajweave.metrics import JsonlMetricSink, MetricAggregator, MetricEvent, parse_verl_console_metrics
from trajweave.rollout.engine import RolloutResult
from trajweave.runtime.logging import StructuredLogger
from trajweave.storage import ArtifactStore, RunStore, TrajectoryStore


class ExperimentTracker:
    def __init__(self, run_store: RunStore, *, logging_config: dict[str, Any] | None = None) -> None:
        logging_config = logging_config or {}
        self.run_store = run_store
        self.run_id = run_store.run_id
        self.run_dir = run_store.run_dir
        self.logger = StructuredLogger(
            run_store.run_id,
            run_store.run_dir,
            level=str(logging_config.get("level", "INFO")),
            console=bool(logging_config.get("console", True)),
        )
        self.metrics = JsonlMetricSink(run_store.run_dir)
        self.metric_aggregator = MetricAggregator(run_store.run_dir)
        self.trajectories = TrajectoryStore(run_store.run_dir, run_store.run_id)
        self.artifacts = ArtifactStore(run_store.run_dir)

    def log_event(self, event: str, message: str, payload: dict[str, Any] | None = None, *, level: str = "INFO") -> None:
        self.logger.log(level, event, message, payload)

    def log_metric(
        self,
        name: str,
        value: float | int | str | bool | None,
        *,
        step: int | None = None,
        source: str = "trajweave",
        split: str | None = None,
        tags: dict[str, Any] | None = None,
    ) -> None:
        self.metrics.write(
            MetricEvent(
                run_id=self.run_id,
                name=name,
                value=value,
                step=step,
                source=source,
                split=split,
                tags=tags or {},
            )
        )

    def log_metrics(self, metrics: dict[str, Any], *, source: str = "trajweave", step: int | None = None) -> None:
        for name, value in metrics.items():
            if isinstance(value, (int, float, str, bool)) or value is None:
                self.log_metric(name, value, source=source, step=step)

    def log_rollout_result(self, result: RolloutResult, *, source: str = "rollout") -> None:
        self.log_trajectories(result.trajectories)
        self.log_samples(result.samples)
        self.log_metric("success_rate", result.success_rate, source=source)
        self.log_metric("trajectories", len(result.trajectories), source=source)
        self.log_metric("samples", len(result.samples), source=source)

    def log_trajectories(self, trajectories: list[MultiAgentTrajectory]) -> None:
        self.trajectories.write_trajectories(trajectories)

    def log_samples(self, samples: list[TrainingSample]) -> None:
        self.trajectories.write_samples(samples)

    def log_artifact(self, *, name: str, path: str | Path, kind: str, metadata: dict[str, Any] | None = None) -> None:
        self.artifacts.register(name=name, path=path, kind=kind, metadata=metadata)

    def log_verl_result(self, result: dict[str, Any]) -> None:
        stdout_path = result.get("stdout_path")
        stderr_path = result.get("stderr_path")
        if stdout_path:
            stdout_path = self._archive_log(
                source=Path(str(stdout_path)),
                name="verl_stdout.log",
                metadata={"stream": "stdout"},
            )
        if stderr_path:
            stderr_path = self._archive_log(
                source=Path(str(stderr_path)),
                name="verl_stderr.log",
                metadata={"stream": "stderr"},
            )
        stdout_text = str(result.get("stdout") or "")
        if stdout_path and Path(str(stdout_path)).exists():
            stdout_text = Path(str(stdout_path)).read_text(encoding="utf-8", errors="replace")
        if stdout_text:
            for event in parse_verl_console_metrics(stdout_text, run_id=self.run_id):
                self.metrics.write(event)
        self.log_metric("returncode", result.get("returncode"), source="verl")

    def _archive_log(self, *, source: Path, name: str, metadata: dict[str, Any]) -> Path:
        target = self.run_dir / "logs" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.exists() and source.resolve() != target.resolve():
            shutil.copy2(source, target)
        elif not target.exists() and source.exists():
            shutil.copy2(source, target)
        self.log_artifact(name=name, path=target if target.exists() else source, kind="log", metadata=metadata)
        return target if target.exists() else source

    def write_summary(self, output: dict[str, Any]) -> dict[str, Any]:
        metric_summary = self.metric_aggregator.summarize()
        summary = {**output, "run_id": self.run_id, "run_dir": str(self.run_dir), "metric_summary": metric_summary}
        self.run_store.write_summary(summary)
        return summary

    def finalize(self, status: str, *, error: str | None = None) -> None:
        if status == "completed":
            self.log_event("run_completed", "TrajWeave run completed.")
        elif status == "failed":
            self.log_event("run_failed_finalized", "TrajWeave run finalized as failed.", {"error": error}, level="ERROR")
        self.log_artifact(name="events.jsonl", path=self.run_dir / "logs" / "events.jsonl", kind="log")
        self.log_artifact(name="console.log", path=self.run_dir / "logs" / "console.log", kind="log")
        self.metric_aggregator.summarize()
        self.run_store.write_status(status, error=error)
