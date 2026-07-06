from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from trajweave.storage.serialization import json_safe


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(value: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "-" for char in value)
    return "-".join(part for part in slug.split("-") if part) or "run"


def git_sha(cwd: str | Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


@dataclass(frozen=True)
class RunStoreConfig:
    root_dir: str = "outputs/trajweave/runs"
    run_id: str | None = None
    name: str | None = None
    tags: list[str] = field(default_factory=list)
    resume: bool = False
    overwrite: bool = False

    @classmethod
    def from_config(cls, config: dict[str, Any], *, recipe: str) -> "RunStoreConfig":
        run_cfg = config.get("run", {}) or {}
        if not isinstance(run_cfg, dict):
            raise TypeError("run config must be a mapping.")
        return cls(
            root_dir=str(run_cfg.get("root_dir", "outputs/trajweave/runs")),
            run_id=run_cfg.get("id"),
            name=run_cfg.get("name") or recipe,
            tags=[str(item) for item in run_cfg.get("tags", [])],
            resume=bool(run_cfg.get("resume", False)),
            overwrite=bool(run_cfg.get("overwrite", False)),
        )


class RunStore:
    def __init__(
        self,
        config: RunStoreConfig,
        *,
        recipe: str,
        canonical_recipe: str,
        mode: str,
        config_path: str | None,
        raw_config: dict[str, Any],
        cwd: str | Path | None = None,
    ) -> None:
        self.config = config
        self.recipe = recipe
        self.canonical_recipe = canonical_recipe
        self.mode = mode
        self.config_path = config_path
        self.raw_config = raw_config
        self.cwd = Path(cwd or Path.cwd())
        self.run_id = config.run_id or self._new_run_id()
        self.run_dir = Path(config.root_dir) / self.run_id

    def initialize(self) -> None:
        if self.run_dir.exists() and self.config.overwrite:
            shutil.rmtree(self.run_dir)
        if self.run_dir.exists() and not (self.config.resume or self.config.overwrite):
            raise FileExistsError(f"Run directory already exists: {self.run_dir}")
        for path in [
            self.run_dir,
            self.logs_dir,
            self.metrics_dir,
            self.trajectories_dir,
            self.artifacts_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)
        self.write_manifest()
        self.write_config_snapshot()
        self.write_status("running")

    @property
    def logs_dir(self) -> Path:
        return self.run_dir / "logs"

    @property
    def metrics_dir(self) -> Path:
        return self.run_dir / "metrics"

    @property
    def trajectories_dir(self) -> Path:
        return self.run_dir / "trajectories"

    @property
    def artifacts_dir(self) -> Path:
        return self.run_dir / "artifacts"

    def _new_run_id(self) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        name = slugify(self.config.name or self.recipe)
        return f"{timestamp}-{name}-{uuid4().hex[:8]}"

    def write_manifest(self) -> None:
        manifest = {
            "run_id": self.run_id,
            "name": self.config.name,
            "tags": self.config.tags,
            "recipe": self.recipe,
            "canonical_recipe": self.canonical_recipe,
            "mode": self.mode,
            "config_path": self.config_path,
            "git_sha": git_sha(self.cwd),
            "created_at": utc_now_iso(),
        }
        self._write_json(self.run_dir / "manifest.json", manifest)

    def write_config_snapshot(self) -> None:
        config_path = self.run_dir / "config.yaml"
        config_path.write_text(yaml.safe_dump(json_safe(self.raw_config), allow_unicode=True, sort_keys=False), encoding="utf-8")

    def write_status(self, status: str, *, error: str | None = None) -> None:
        payload = {"run_id": self.run_id, "status": status, "updated_at": utc_now_iso()}
        if error:
            payload["error"] = error
        self._write_json(self.run_dir / "status.json", payload)

    def write_summary(self, summary: dict[str, Any]) -> None:
        self._write_json(self.run_dir / "summary.json", {"run_id": self.run_id, **json_safe(summary)})

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
