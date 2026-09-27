from __future__ import annotations

import codecs
import os
import shlex
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from threading import Thread
from typing import BinaryIO, TextIO

from trajweave.storage.serialization import redact_text

_READ_CHUNK_SIZE = 8192
_MAX_PENDING_TEXT = 64 * 1024
_REDACTION_OVERLAP = 4096
_RESULT_TAIL_SIZE = 4000
_TERMINATE_TIMEOUT_SECONDS = 10


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
        command = " ".join(shlex.quote(part) for part in self.config.command())
        lines = ["#!/usr/bin/env bash", "set -euo pipefail", command]
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        output_path.chmod(0o700)
        return output_path

    def run(self) -> dict:
        command = self.config.command()
        if not self.config.execute:
            return {"status": "dry_run", "command": command}

        stdout_path, stderr_path = self._log_paths()
        stream_errors: list[OSError] = []
        with (
            stdout_path.open("w", encoding="utf-8") as stdout_file,
            stderr_path.open("w", encoding="utf-8") as stderr_file,
        ):
            stdout_path.chmod(0o600)
            stderr_path.chmod(0o600)
            stdout_capture = _RedactingLogCapture(stdout_file, errors=stream_errors)
            stderr_capture = _RedactingLogCapture(stderr_file, errors=stream_errors)
            process = subprocess.Popen(
                command,
                cwd=self.config.cwd,
                env=self._subprocess_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            if process.stdout is None or process.stderr is None:
                process.terminate()
                process.wait()
                raise RuntimeError("VERL subprocess did not expose stdout/stderr pipes.")

            threads = [
                Thread(target=_pump_stream, args=(process.stdout, stdout_capture), name="verl-stdout"),
                Thread(target=_pump_stream, args=(process.stderr, stderr_capture), name="verl-stderr"),
            ]
            for thread in threads:
                thread.start()
            try:
                returncode = process.wait()
            except BaseException:
                _terminate_process_tree(process)
                raise
            finally:
                for thread in threads:
                    thread.join()

        if stream_errors:
            raise OSError(f"Failed to stream VERL subprocess logs: {stream_errors[0]}") from stream_errors[0]
        return {
            "status": "ok" if returncode == 0 else "failed",
            "returncode": returncode,
            "command": command,
            "stdout": stdout_capture.tail,
            "stderr": stderr_capture.tail,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        }

    def _log_paths(self) -> tuple[Path, Path]:
        temp_dir: Path | None = None
        if not self.config.stdout_path or not self.config.stderr_path:
            temp_dir = Path(tempfile.mkdtemp(prefix="trajweave-verl-"))
        if self.config.stdout_path:
            stdout_path = Path(self.config.stdout_path).expanduser()
        else:
            assert temp_dir is not None
            stdout_path = temp_dir / "stdout.log"
        if self.config.stderr_path:
            stderr_path = Path(self.config.stderr_path).expanduser()
        else:
            assert temp_dir is not None
            stderr_path = temp_dir / "stderr.log"
        stdout_path = stdout_path.resolve()
        stderr_path = stderr_path.resolve()
        if stdout_path == stderr_path:
            raise ValueError("VERL stdout_path and stderr_path must be different files.")
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        return stdout_path, stderr_path

    def _subprocess_env(self) -> dict[str, str] | None:
        if not self.config.env:
            return None
        return {**os.environ, **{str(key): str(value) for key, value in self.config.env.items()}}


class _RedactingLogCapture:
    def __init__(self, output: TextIO, *, errors: list[OSError]) -> None:
        self.output: TextIO | None = output
        self.errors = errors
        self.pending = ""
        self.tail = ""

    def feed(self, text: str) -> None:
        self.pending += text
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            self._emit(f"{line}\n")
        if len(self.pending) > _MAX_PENDING_TEXT:
            cutoff = len(self.pending) - _REDACTION_OVERLAP
            self._emit(self.pending[:cutoff])
            self.pending = self.pending[cutoff:]

    def finish(self) -> None:
        if self.pending:
            self._emit(self.pending)
            self.pending = ""

    def _emit(self, text: str) -> None:
        redacted = redact_text(text)
        self.tail = (self.tail + redacted)[-_RESULT_TAIL_SIZE:]
        if self.output is None:
            return
        try:
            self.output.write(redacted)
            self.output.flush()
        except OSError as exc:
            self.errors.append(exc)
            self.output = None


def _pump_stream(stream: BinaryIO, capture: _RedactingLogCapture) -> None:
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    read = getattr(stream, "read1", stream.read)
    try:
        while chunk := read(_READ_CHUNK_SIZE):
            capture.feed(decoder.decode(chunk))
        capture.feed(decoder.decode(b"", final=True))
    except OSError as exc:
        capture.errors.append(exc)
    finally:
        capture.finish()
        stream.close()


def _terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    """Terminate the isolated VERL process group, including Ray descendants."""

    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        process.terminate()
    try:
        process.wait(timeout=_TERMINATE_TIMEOUT_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError:
        process.kill()
    process.wait()
