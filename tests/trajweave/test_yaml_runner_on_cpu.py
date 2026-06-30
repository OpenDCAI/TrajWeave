from trajweave.runner import run_from_config


def test_yaml_runner_runs_math_smoke_config():
    result = run_from_config(
        {
            "recipe": "doctor_mas_math",
            "mode": "smoke",
            "backend": {"type": "rule"},
            "team": {"max_turns": 2},
            "rollout": {"rollouts_per_task": 1},
        }
    )

    assert result["recipe"] == "doctor_mas_math"
    assert result["trajectories"] == 2
    assert result["success_rate"] == 1.0


def test_yaml_runner_runs_search_smoke_config():
    result = run_from_config(
        {
            "recipe": "doctor_mas_search",
            "mode": "smoke",
            "backend": {"type": "rule"},
            "team": {"max_turns": 2},
            "rollout": {"rollouts_per_task": 1},
        }
    )

    assert result["recipe"] == "doctor_mas_search"
    assert result["trajectories"] == 2
    assert result["success_rate"] == 1.0
