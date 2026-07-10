from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from trajweave.metrics.events import MetricEvent
from trajweave.storage.jsonl import JsonlWriter
from trajweave.storage.serialization import json_safe


class JsonlMetricSink:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.metrics_dir = self.run_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.metrics_dir / "metrics.jsonl"
        self._writer = JsonlWriter(self.path)

    def write(self, event: MetricEvent) -> None:
        self._writer.write(json_safe(event))

    def write_many(self, events: list[MetricEvent]) -> None:
        for event in events:
            self.write(event)


class MetricAggregator:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.metrics_path = self.run_dir / "metrics" / "metrics.jsonl"
        self.summary_path = self.run_dir / "metrics" / "summary.json"

    def summarize(self) -> dict[str, Any]:
        latest: dict[str, Any] = {}
        counts: dict[str, int] = {}
        if self.metrics_path.exists():
            with self.metrics_path.open("r", encoding="utf-8") as file:
                for line in file:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    name = str(row.get("name"))
                    latest[name] = row.get("value")
                    counts[name] = counts.get(name, 0) + 1
        summary = {"latest": latest, "counts": counts}
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_path = tempfile.mkstemp(prefix=f".{self.summary_path.name}.", dir=self.summary_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary_path, 0o600)
            os.replace(temporary_path, self.summary_path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)
        return summary
