from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from trajweave.storage.jsonl import JsonlWriter


class ArtifactStore:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.artifacts_dir = self.run_dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.artifacts_dir / "artifact_index.jsonl"
        self._index = JsonlWriter(self.index_path)

    def path(self, name: str) -> Path:
        output = self.artifacts_dir / name
        output.parent.mkdir(parents=True, exist_ok=True)
        return output

    def register(self, *, name: str, path: str | Path, kind: str, metadata: dict[str, Any] | None = None) -> None:
        artifact_path = Path(path)
        self._index.write(
            {
                "name": name,
                "kind": kind,
                "path": str(artifact_path),
                "exists": artifact_path.exists(),
                "metadata": metadata or {},
            }
        )

    def copy_file(self, *, name: str, source: str | Path, kind: str, metadata: dict[str, Any] | None = None) -> Path:
        source_path = Path(source)
        target_path = self.path(name)
        if source_path.exists() and source_path.resolve() != target_path.resolve():
            shutil.copy2(source_path, target_path)
        self.register(name=name, path=target_path, kind=kind, metadata=metadata)
        return target_path
