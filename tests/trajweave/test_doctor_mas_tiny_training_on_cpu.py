from pathlib import Path

import pytest

from trajweave.recipes.doctor_mas.train_tiny import TrainConfig, run_training


def test_doctor_mas_tiny_training_writes_metrics_and_checkpoint(tmp_path: Path):
    pytest.importorskip("torch")

    output_dir = tmp_path / "tiny_train"
    history = run_training(
        TrainConfig(
            steps=2,
            rollouts_per_task=2,
            eval_interval=1,
            lr=0.05,
            entropy_coef=0.0,
            seed=7,
            task_mode="fixed",
            output_dir=str(output_dir),
        )
    )

    assert len(history) == 3
    assert (output_dir / "config.json").is_file()
    assert (output_dir / "metrics.jsonl").is_file()
    assert (output_dir / "tiny_policy.pt").is_file()
    assert all(row["trainable_samples"] > 0 for row in history)
    assert history[-1]["eval_success_rate"] is not None
