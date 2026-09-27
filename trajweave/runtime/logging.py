from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from trajweave.storage.jsonl import JsonlWriter
from trajweave.storage.run_store import utc_now_iso
from trajweave.storage.serialization import redact_secrets


class StructuredLogger:
    def __init__(self, run_id: str, run_dir: str | Path, *, level: str = "INFO", console: bool = True) -> None:
        self.run_id = run_id
        self.run_dir = Path(run_dir)
        self.events_path = self.run_dir / "logs" / "events.jsonl"
        self.console_path = self.run_dir / "logs" / "console.log"
        self._writer = JsonlWriter(self.events_path)
        self._level = getattr(logging, level.upper(), logging.INFO)
        self._console = console
        self.run_dir.joinpath("logs").mkdir(parents=True, exist_ok=True)

    def log(self, level: str, event: str, message: str, payload: dict[str, Any] | None = None) -> None:
        level_name = level.upper()
        row = redact_secrets(
            {
                "ts": utc_now_iso(),
                "run_id": self.run_id,
                "level": level_name,
                "event": event,
                "message": message,
                "payload": payload or {},
            }
        )
        self._writer.write(row)
        if getattr(logging, level_name, logging.INFO) >= self._level:
            with self.console_path.open("a", encoding="utf-8") as file:
                file.write(f"{row['ts']} {row['level']} {row['event']} {row['message']}\n")
            self.console_path.chmod(0o600)
            if self._console:
                logging.getLogger("trajweave.runtime").log(
                    getattr(logging, level_name, logging.INFO),
                    row["message"],
                )

    def info(self, event: str, message: str, payload: dict[str, Any] | None = None) -> None:
        self.log("INFO", event, message, payload)

    def warning(self, event: str, message: str, payload: dict[str, Any] | None = None) -> None:
        self.log("WARNING", event, message, payload)

    def error(self, event: str, message: str, payload: dict[str, Any] | None = None) -> None:
        self.log("ERROR", event, message, payload)
