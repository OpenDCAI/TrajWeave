from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from trajweave.storage.serialization import json_safe


class JsonlWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, row: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")
        self.path.chmod(0o600)

    def write_many(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        with self.path.open("a", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")
        self.path.chmod(0o600)
