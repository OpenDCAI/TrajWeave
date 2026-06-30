from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class VerlTrainerLaunchConfig:
    python: str = sys.executable
    module: str = "verl.trainer.main_ppo"
    overrides: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    execute: bool = False
    stdout_path: str | None = None
    stderr_path: str | None = None

    def command(self) -> list[str]:
        return [self.python, "-m", self.module, *self.overrides]


@dataclass
class VerlTrainerLauncher:
    config: VerlTrainerLaunchConfig

    def write_command_file(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = " ".join(self.config.command())
        lines = ["#!/usr/bin/env bash", "set -euo pipefail", command]
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        output_path.chmod(0o755)
        return output_path

    def run(self) -> dict:
        command = self.config.command()
        if not self.config.execute:
            return {"status": "dry_run", "command": command}
        result = subprocess.run(
            command,
            cwd=self.config.cwd,
            env=None if not self.config.env else {**os.environ, **self.config.env},
            check=False,
            text=True,
            capture_output=True,
        )
        if self.config.stdout_path:
            stdout_path = Path(self.config.stdout_path)
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
            stdout_path.write_text(result.stdout, encoding="utf-8")
        if self.config.stderr_path:
            stderr_path = Path(self.config.stderr_path)
            stderr_path.parent.mkdir(parents=True, exist_ok=True)
            stderr_path.write_text(result.stderr, encoding="utf-8")
        return {
            "status": "ok" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "command": command,
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-4000:],
            "stdout_path": self.config.stdout_path,
            "stderr_path": self.config.stderr_path,
        }
