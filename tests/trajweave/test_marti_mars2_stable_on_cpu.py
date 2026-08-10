import pytest

from trajweave.recipes.marti_mars2.stable import run_stable_cpu_fixture
from trajweave.runner import run_from_config


@pytest.mark.parametrize("tis_level", ["token", "sequence"])
def test_stable_cpu_fixture_accepts_gspo_tis_and_overlong(tis_level):
    result = run_stable_cpu_fixture(
        {
            "stable": {
                "loss": "gspo",
                "tis_level": tis_level,
                "tis_mode": "truncate",
                "tis_threshold": 2.0,
                "overlong_buffer_len": 2,
                "overlong_penalty_factor": 1.0,
            }
        }
    )

    assert result["acceptance"]["status"] == "passed"
    assert result["metrics"]["tis_level"] == tis_level
    assert result["metrics"]["tis_finite"] is True
    assert result["metrics"]["ess"] > 0
    assert result["metrics"]["overlong_penalty"] < 0


def test_yaml_runner_runs_stable_cpu_fixture(tmp_path):
    result = run_from_config(
        {
            "recipe": "marti_mars2.stable.gspo_sequence_tis",
            "mode": "smoke",
            "run": {"root_dir": str(tmp_path), "name": "unit-marti-mars2-stable"},
            "logging": {"console": False},
            "stable": {
                "loss": "gspo",
                "tis_level": "sequence",
                "tis_mode": "truncate",
                "overlong_buffer_len": 2,
            },
        }
    )

    assert result["stable_cpu_fixture"]["acceptance"]["status"] == "passed"
