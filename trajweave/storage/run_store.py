from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any
from uuid import uuid4

import yaml

from trajweave.storage.serialization import json_safe, redact_secrets


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(value: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "-" for char in value)
    return "-".join(part for part in slug.split("-") if part) or "run"


def git_sha(cwd: str | Path | None = None) -> str | None:
    try:
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stdout:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=cwd,
                check=False,
                stdout=stdout,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            stdout.seek(0)
            sha = stdout.read(64).strip()
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return sha or None


def git_worktree_fingerprint(cwd: str | Path | None = None) -> str | None:
    """Hash tracked changes and every non-ignored untracked file in the worktree."""

    root = Path(cwd or Path.cwd())
    try:
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=root,
            check=False,
            capture_output=True,
        )
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=root,
            check=False,
            capture_output=True,
        )
    except OSError:
        return None
    if diff.returncode != 0 or untracked.returncode != 0:
        return None

    digest = hashlib.sha256()
    _update_fingerprint(digest, b"tracked-diff", diff.stdout)
    for raw_path in sorted(path for path in untracked.stdout.split(b"\0") if path):
        path = root / os.fsdecode(raw_path)
        try:
            content = path.read_bytes()
        except OSError:
            return None
        _update_fingerprint(digest, raw_path, content)
    return digest.hexdigest()


def git_worktree_is_dirty(cwd: str | Path | None = None) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=cwd,
            check=False,
            capture_output=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return bool(result.stdout.strip())


def _update_fingerprint(digest: Any, label: bytes, content: bytes) -> None:
    digest.update(len(label).to_bytes(8, "big"))
    digest.update(label)
    digest.update(len(content).to_bytes(8, "big"))
    digest.update(content)


@dataclass(frozen=True)
class RunStoreConfig:
    root_dir: str = "outputs/trajweave/runs"
    run_id: str | None = None
    name: str | None = None
    tags: list[str] = field(default_factory=list)
    resume: bool = False
    overwrite: bool = False

    @classmethod
    def from_config(cls, config: dict[str, Any], *, recipe: str) -> RunStoreConfig:
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
        self.run_id = config.run_id if config.run_id is not None else self._new_run_id()
        self.root_dir, self.run_dir = _safe_run_dir(config.root_dir, self.run_id)
        lock_digest = hashlib.sha256(self.run_id.encode("utf-8")).hexdigest()[:16]
        self._lock_path = self.root_dir / ".locks" / f"{slugify(self.run_id)}-{lock_digest}.lock"
        self._lock_fd: int | None = None

    def initialize(self) -> None:
        if self.config.resume:
            raise ValueError(
                "run.resume=true is not supported because TrajWeave does not connect run-store reuse "
                "to backend checkpoint recovery."
            )
        self._assert_safe_run_dir()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._acquire_lock()
        try:
            if self.run_dir.exists() and self.config.overwrite:
                shutil.rmtree(self.run_dir)
            if self.run_dir.exists() and not self.config.overwrite:
                raise FileExistsError(f"Run directory already exists: {self.run_dir}")
            self.run_dir.parent.mkdir(parents=True, exist_ok=True)
            self.run_dir.mkdir(parents=False, exist_ok=False)
            for path in [
                self.logs_dir,
                self.metrics_dir,
                self.trajectories_dir,
                self.artifacts_dir,
            ]:
                path.mkdir(parents=False, exist_ok=False)
            self.write_manifest()
            self.write_config_snapshot()
            self.write_status("running")
        except Exception:
            self.release_lock()
            raise

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
            "worktree_diff_sha256": git_worktree_fingerprint(self.cwd),
            "worktree_dirty": git_worktree_is_dirty(self.cwd),
            "created_at": utc_now_iso(),
        }
        self._write_json(self.run_dir / "manifest.json", manifest)

    def write_config_snapshot(self) -> None:
        config_path = self.run_dir / "config.yaml"
        self._atomic_write_text(
            config_path,
            yaml.safe_dump(redact_secrets(self.raw_config), allow_unicode=True, sort_keys=False),
        )

    def write_status(self, status: str, *, error: str | None = None) -> None:
        payload = {"run_id": self.run_id, "status": status, "updated_at": utc_now_iso()}
        if error:
            payload["error"] = error
        self._write_json(self.run_dir / "status.json", payload)

    def write_summary(self, summary: dict[str, Any]) -> None:
        self._write_json(self.run_dir / "summary.json", {"run_id": self.run_id, **json_safe(summary)})

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        self._atomic_write_text(
            path,
            json.dumps(redact_secrets(payload), ensure_ascii=False, indent=2) + "\n",
        )

    def _atomic_write_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary_path, 0o600)
            os.replace(temporary_path, path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def _acquire_lock(self) -> None:
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._lock_fd = os.open(self._lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise RuntimeError(f"Run is already active or has a stale lock: {self._lock_path}") from exc
        os.write(self._lock_fd, f"pid={os.getpid()}\n".encode("ascii"))

    def release_lock(self) -> None:
        if self._lock_fd is None:
            return
        os.close(self._lock_fd)
        self._lock_fd = None
        try:
            self._lock_path.unlink()
        except FileNotFoundError:
            pass

    def _assert_safe_run_dir(self) -> None:
        resolved = self.run_dir.resolve()
        if resolved != self.run_dir or resolved == self.root_dir or not resolved.is_relative_to(self.root_dir):
            raise ValueError(f"Run directory must remain a direct descendant of root_dir: {self.run_dir}")


def _safe_run_dir(root_dir: str | Path, run_id: str) -> tuple[Path, Path]:
    if not isinstance(run_id, str):
        raise TypeError("run.id must be a string.")
    if "\x00" in run_id:
        raise ValueError("run.id must not contain null bytes.")

    run_path = Path(run_id)
    windows_path = PureWindowsPath(run_id)
    if run_path.is_absolute() or windows_path.is_absolute():
        raise ValueError(f"run.id must be relative to root_dir, got absolute path: {run_id!r}")
    if ".." in run_path.parts or ".." in windows_path.parts:
        raise ValueError(f"run.id must not contain '..': {run_id!r}")

    root_path = Path(root_dir).expanduser().resolve()
    candidate = (root_path / run_path).resolve()
    if candidate == root_path:
        raise ValueError("run.id must identify a run below root_dir, not root_dir itself.")
    if not candidate.is_relative_to(root_path):
        raise ValueError(f"run.id escapes root_dir: {run_id!r}")
    return root_path, candidate
