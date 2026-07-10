from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from trajweave.storage.jsonl import JsonlWriter
from trajweave.storage.serialization import redact_secrets


class ArtifactStore:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.artifacts_dir = self.run_dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.artifacts_dir / "artifact_index.jsonl"
        self._index = JsonlWriter(self.index_path)

    def path(self, name: str) -> Path:
        candidate = Path(name)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"Artifact name must stay below artifacts_dir: {name!r}")
        output = (self.artifacts_dir / candidate).resolve()
        artifacts_root = self.artifacts_dir.resolve()
        if output == artifacts_root or not output.is_relative_to(artifacts_root):
            raise ValueError(f"Artifact name must identify a file below artifacts_dir: {name!r}")
        output.parent.mkdir(parents=True, exist_ok=True)
        return output

    def register(
        self,
        *,
        name: str,
        path: str | Path,
        kind: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        artifact_path = Path(path).expanduser().resolve()
        if not artifact_path.exists():
            return
        self._index.write(
            redact_secrets(
                {
                    "name": name,
                    "kind": kind,
                    "path": str(artifact_path),
                    "exists": True,
                    "metadata": metadata or {},
                }
            )
        )

    def copy_file(
        self,
        *,
        name: str,
        source: str | Path,
        kind: str,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        source_path = Path(source)
        if not source_path.is_file():
            raise FileNotFoundError(f"Artifact source file does not exist: {source_path}")
        target_path = self.path(name)
        if source_path.resolve() != target_path.resolve():
            shutil.copy2(source_path, target_path)
        self.register(name=name, path=target_path, kind=kind, metadata=metadata)
        return target_path
