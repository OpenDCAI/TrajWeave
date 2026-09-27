import json
import subprocess
import sys
from pathlib import Path

import yaml

from trajweave.cli import run as cli_run


def test_cli_returns_nonzero_for_failed_verl_launch(monkeypatch, capsys):
    monkeypatch.setattr(
        cli_run,
        "run_from_config_path",
        lambda _path: {
            "verl_launch": {
                "status": "failed",
                "returncode": 23,
                "command": ["python", "+api_key=cli-secret"],
            }
        },
    )

    exit_code = cli_run.main(["--config", "unused.yaml"])

    assert exit_code == 1
    assert "cli-secret" not in capsys.readouterr().out


def test_cli_returns_zero_for_successful_run(monkeypatch):
    monkeypatch.setattr(
        cli_run,
        "run_from_config_path",
        lambda _path: {"verl_launch": {"status": "ok", "returncode": 0}},
    )

    assert cli_run.main(["--config", "unused.yaml"]) == 0


def test_cli_module_process_exits_nonzero_for_failed_verl(tmp_path: Path):
    config_path = tmp_path / "failed-verl.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "recipe": "agentflow.flow_grpo.planner_tool",
                "mode": "verl_train",
                "run": {"root_dir": str(tmp_path / "runs")},
                "logging": {"console": False},
                "prepare": {"tiny_verl_assets": {"enabled": False}},
                "agentflow": {"agent_loop_backend": "synthetic_tq", "max_steps": 2},
                "verl": {
                    "enabled": True,
                    "execute": True,
                    "module": "trajweave.tests.no_such_verl_module",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    repo_root = Path(__file__).resolve().parents[2]

    completed = subprocess.run(
        [sys.executable, "-m", "trajweave.cli.run", "--config", str(config_path)],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 1
    assert json.loads(completed.stdout)["verl_launch"]["status"] == "failed"
