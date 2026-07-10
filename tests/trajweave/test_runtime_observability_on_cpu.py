import json
import stat
from pathlib import Path

from trajweave.metrics import parse_verl_console_metrics
from trajweave.runner import run_from_config
from trajweave.storage import TrajectoryStore


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_runner_creates_run_store_logs_metrics_and_trajectories(tmp_path):
    result = run_from_config(
        {
            "recipe": "doctor_mas_math",
            "mode": "smoke",
            "run": {"root_dir": str(tmp_path), "name": "unit-smoke"},
            "logging": {"console": False},
            "backend": {"type": "rule"},
            "team": {"max_turns": 2},
            "rollout": {"rollouts_per_task": 1},
        }
    )

    run_dir = Path(result["run_dir"])
    assert result["run_id"]
    assert run_dir.exists()
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "status.json").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "config.yaml").exists()
    assert read_jsonl(run_dir / "logs" / "events.jsonl")[0]["event"] == "run_started"
    assert len(read_jsonl(run_dir / "trajectories" / "trajectories.jsonl")) == 2
    assert len(read_jsonl(run_dir / "trajectories" / "turns.jsonl")) >= 4
    assert len(read_jsonl(run_dir / "trajectories" / "samples.jsonl")) >= 4
    metric_names = {row["name"] for row in read_jsonl(run_dir / "metrics" / "metrics.jsonl")}
    assert {"success_rate", "trajectories", "samples"}.issubset(metric_names)
    assert stat.S_IMODE((run_dir / "metrics" / "summary.json").stat().st_mode) == 0o600


def test_verl_dry_run_is_registered_as_run_artifact(tmp_path):
    result = run_from_config(
        {
            "recipe": "agentflow.flow_grpo.planner_tool",
            "mode": "verl_train",
            "run": {"root_dir": str(tmp_path), "name": "unit-verl-dry-run"},
            "logging": {"console": False},
            "prepare": {"tiny_verl_assets": {"enabled": False}},
            "agentflow": {"agent_loop_backend": "synthetic_tq", "max_steps": 2},
            "verl": {"enabled": True, "execute": False, "overrides": ["trainer.use_v1=true"]},
        }
    )

    run_dir = Path(result["run_dir"])
    command_file = run_dir / "artifacts" / "run_verl_ppo.sh"
    assert result["verl_launch"]["status"] == "dry_run"
    assert command_file.exists()
    command_text = " ".join(result["verl_launch"]["command"])
    assert f'+trajweave.run_id="{result["run_id"]}"' in command_text
    assert "+trajweave.capture_online_turns=true" in command_text
    assert "trainer.rollout_data_dir=" not in command_text
    artifact_rows = read_jsonl(run_dir / "artifacts" / "artifact_index.jsonl")
    assert any(row["name"] == "run_verl_ppo.sh" for row in artifact_rows)


def test_verl_native_rollout_dump_is_explicit_opt_in(tmp_path):
    result = run_from_config(
        {
            "recipe": "agentflow.flow_grpo.planner_tool",
            "mode": "verl_train",
            "run": {"root_dir": str(tmp_path), "name": "unit-native-rollout-dump"},
            "logging": {"console": False},
            "prepare": {"tiny_verl_assets": {"enabled": False}},
            "agentflow": {"agent_loop_backend": "synthetic_tq", "max_steps": 2},
            "verl": {
                "enabled": True,
                "execute": False,
                "capture_native_rollouts": True,
                "overrides": ["trainer.use_v1=true"],
            },
        }
    )

    command_text = " ".join(result["verl_launch"]["command"])
    assert "trainer.rollout_data_dir=" in command_text
    assert "trainer.validation_data_dir=" in command_text


def test_verl_failed_launch_marks_run_failed(tmp_path):
    result = run_from_config(
        {
            "recipe": "agentflow.flow_grpo.planner_tool",
            "mode": "verl_train",
            "run": {"root_dir": str(tmp_path), "name": "unit-verl-failed-launch"},
            "logging": {"console": False},
            "prepare": {"tiny_verl_assets": {"enabled": False}},
            "agentflow": {"agent_loop_backend": "synthetic_tq", "max_steps": 2},
            "verl": {
                "enabled": True,
                "execute": True,
                "module": "trajweave.tests.no_such_verl_module",
                "overrides": ["trainer.use_v1=true"],
            },
        }
    )

    run_dir = Path(result["run_dir"])
    assert result["verl_launch"]["status"] == "failed"
    assert read_jsonl(run_dir / "logs" / "events.jsonl")[-1]["event"] == "run_failed_finalized"
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert "VERL launch failed" in status["error"]


def test_verl_metric_parser_extracts_step_metrics():
    events = parse_verl_console_metrics(
        "step:1 - actor/pg_loss:0.25 - actor/grad_norm:2 - critic/score/mean:0.5",
        run_id="run-test",
    )

    assert {event.name for event in events} == {"actor/pg_loss", "actor/grad_norm", "critic/score/mean"}
    assert {event.step for event in events} == {1}
    assert [event for event in events if event.name == "actor/pg_loss"][0].value == 0.25


def test_trajectory_store_writes_online_turn_shards(tmp_path):
    store = TrajectoryStore(tmp_path, "run-test")

    store.write_online_turn("worker:1", {"agent_name": "solver", "turn_id": 0})

    shard_path = tmp_path / "trajectories" / "online_turns" / "worker_1.jsonl"
    rows = read_jsonl(shard_path)
    assert rows == [{"run_id": "run-test", "agent_name": "solver", "turn_id": 0}]
