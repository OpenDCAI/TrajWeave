from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from trajweave.storage.jsonl import JsonlWriter


def write_online_turn_rows(runtime: Any, rows: list[dict[str, Any]]) -> None:
    """Persist online turn rows for any AgentLoop backend."""
    if not rows or not runtime.capture_online_turns or not runtime.run_dir or not runtime.run_id:
        return
    path = Path(str(runtime.run_dir)) / "trajectories" / "online_turns" / f"worker-{os.getpid()}.jsonl"
    writer = JsonlWriter(path)
    for row in rows:
        writer.write({"run_id": runtime.run_id, **row})
