import json

from trajweave.recipes.marti_mars2.acceptance import audit_fidelity_training_run, checkpoint_dir_from_overrides


def test_fidelity_acceptance_requires_mixed_real_verifier_rewards_and_training_signal(tmp_path):
    run_dir = tmp_path / "run"
    trajectory_dir = run_dir / "trajectories" / "online_turns"
    trajectory_dir.mkdir(parents=True)
    rows = [
        {
            "reward_score": reward,
            "metadata": {
                "tree_id": "tree-0",
                "verification_mode": "local_subprocess_fallback",
                "failure_type": None if reward == 1 else "compile_error",
            },
        }
        for reward in (0.0, 1.0, 5 / 6)
    ]
    (trajectory_dir / "worker.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint = checkpoint_dir / "global_step_1" / "actor" / "model_world_size_1_rank_0.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    audit = audit_fidelity_training_run(
        run_dir,
        metric_summary={
            "latest": {
                "critic/advantages/max": 0.8,
                "critic/advantages/min": -1.1,
                "actor/loss": 0.01,
                "actor/grad_norm": 2.5,
                "rollout_corr/rollout_is_mean": 1.0,
                "rollout_corr/rollout_is_min": 0.9,
                "rollout_corr/rollout_is_max": 1.1,
            }
        },
        checkpoint_dir=checkpoint_dir,
        allow_local_verifier_fallback=True,
    )

    assert audit["status"] == "passed"
    assert all(audit["checks"].values())
    assert audit["reward_groups"]["tree-0"]["rewards"] == [0.0, 1.0, 5 / 6]
    assert audit["tensor_diff_required"] is True


def test_fidelity_acceptance_rejects_local_verifier_fallback_by_default(tmp_path):
    trajectory_dir = tmp_path / "trajectories" / "online_turns"
    trajectory_dir.mkdir(parents=True)
    (trajectory_dir / "worker.jsonl").write_text(
        json.dumps(
            {
                "reward_score": 1.0,
                "metadata": {
                    "tree_id": "tree-0",
                    "verification_mode": "local_subprocess_fallback",
                    "failure_type": None,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    audit = audit_fidelity_training_run(
        tmp_path,
        metric_summary={"latest": {}},
        checkpoint_dir=None,
        require_correction=False,
        require_learning_signal=False,
    )

    assert audit["status"] == "failed"
    assert audit["checks"]["real_verifier_without_errors"] is False
    assert audit["local_verifier_fallback_allowed"] is False


def test_fidelity_acceptance_rejects_uniform_rewards_and_zero_gradient(tmp_path):
    trajectory_dir = tmp_path / "run" / "trajectories" / "online_turns"
    trajectory_dir.mkdir(parents=True)
    (trajectory_dir / "worker.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "reward_score": 1.0,
                    "metadata": {
                        "tree_id": "tree-0",
                        "verification_mode": "local_subprocess_fallback",
                        "failure_type": None,
                    },
                }
            )
            for _ in range(3)
        ),
        encoding="utf-8",
    )

    audit = audit_fidelity_training_run(
        tmp_path / "run",
        metric_summary={"latest": {"actor/grad_norm": 0.0}},
        checkpoint_dir=None,
    )

    assert audit["status"] == "failed"
    assert audit["checks"]["mixed_rewards_within_tree"] is False
    assert audit["checks"]["finite_nonzero_gradient"] is False
    assert audit["checks"]["checkpoint_present"] is False


def test_fidelity_acceptance_uses_nonzero_learning_signal_from_any_completed_step(tmp_path):
    run_dir = tmp_path / "run"
    trajectory_dir = run_dir / "trajectories" / "online_turns"
    trajectory_dir.mkdir(parents=True)
    rows = [
        {
            "reward_score": reward,
            "metadata": {
                "tree_id": "tree-0",
                "verification_mode": "local_subprocess_fallback",
                "failure_type": None if reward else "wrong_answer",
            },
        }
        for reward in (0.0, 1.0)
    ]
    (trajectory_dir / "worker.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir()
    metric_rows = [
        {"name": "critic/advantages/max", "value": 1.2, "step": 1},
        {"name": "critic/advantages/min", "value": -0.8, "step": 1},
        {"name": "actor/loss", "value": 0.05, "step": 1},
        {"name": "actor/grad_norm", "value": 3.0, "step": 1},
        {"name": "critic/advantages/max", "value": 0.0, "step": 2},
        {"name": "critic/advantages/min", "value": 0.0, "step": 2},
        {"name": "actor/loss", "value": 0.0, "step": 2},
        {"name": "actor/grad_norm", "value": 0.0, "step": 2},
    ]
    (metrics_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in metric_rows),
        encoding="utf-8",
    )
    checkpoint = run_dir / "checkpoints" / "global_step_2" / "actor" / "model_world_size_1_rank_0.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    audit = audit_fidelity_training_run(
        run_dir,
        metric_summary={
            "latest": {
                "critic/advantages/max": 0.0,
                "critic/advantages/min": 0.0,
                "actor/loss": 0.0,
                "actor/grad_norm": 0.0,
            }
        },
        checkpoint_dir=run_dir / "checkpoints",
        require_correction=False,
        allow_local_verifier_fallback=True,
    )

    assert audit["status"] == "passed"
    assert audit["metrics"]["advantage_max"] == 1.2
    assert audit["metrics"]["actor_loss"] == 0.05
    assert audit["metrics"]["actor_grad_norm"] == 3.0


def test_vanilla_grpo_acceptance_does_not_require_rollout_correction(tmp_path):
    run_dir = tmp_path / "run"
    trajectory_dir = run_dir / "trajectories" / "online_turns"
    trajectory_dir.mkdir(parents=True)
    rows = [
        {
            "reward_score": reward,
            "metadata": {
                "tree_id": "group-0",
                "verification_mode": "local_subprocess_fallback",
                "failure_type": None if reward else "wrong_answer",
            },
        }
        for reward in (0.0, 0.5, 1.0)
    ]
    (trajectory_dir / "worker.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint = checkpoint_dir / "global_step_1" / "actor" / "model_world_size_1_rank_0.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    audit = audit_fidelity_training_run(
        run_dir,
        metric_summary={
            "latest": {
                "critic/advantages/max": 1.0,
                "critic/advantages/min": -1.0,
                "actor/loss": 0.1,
                "actor/grad_norm": 1.0,
            }
        },
        checkpoint_dir=checkpoint_dir,
        require_correction=False,
        allow_local_verifier_fallback=True,
    )

    assert audit["status"] == "passed"
    assert audit["correction_required"] is False


def test_fidelity_acceptance_finds_multi_actor_checkpoints(tmp_path):
    checkpoint = tmp_path / "global_step_1" / "actors" / "policy_a" / "model_world_size_1_rank_0.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    audit = audit_fidelity_training_run(
        tmp_path,
        metric_summary={
            "latest": {
                "critic/advantages/max": 1.0,
                "critic/advantages/min": -1.0,
                "actor/loss": 0.1,
                "actor/grad_norm": 1.0,
            }
        },
        checkpoint_dir=tmp_path,
        require_correction=False,
    )
    assert audit["checks"]["checkpoint_present"] is True


def test_fidelity_acceptance_reads_multi_actor_learning_metrics(tmp_path):
    online = tmp_path / "trajectories" / "online_turns"
    online.mkdir(parents=True)
    rows = [
        {
            "policy_group": "policy_a" if reward else "policy_b",
            "agent_id": "generator" if reward else "critic",
            "reward_score": reward,
            "metadata": {
                "tree_id": "tree-0",
                "verification_mode": "local_subprocess_fallback",
                "failure_type": None if reward else "wrong_answer",
            },
        }
        for reward in (1.0, 0.0)
    ]
    (online / "worker.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    for policy_group in ("policy_a", "policy_b"):
        checkpoint = tmp_path / "global_step_1" / "actors" / policy_group / "model_world_size_1_rank_0.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()

    audit = audit_fidelity_training_run(
        tmp_path,
        metric_summary={
            "latest": {
                "critic/advantages/max": 0.8,
                "critic/advantages/min": -0.8,
                "actor/policy_a/loss": -0.2,
                "actor/policy_a/grad_norm": 3.0,
                "actor/policy_b/loss": 0.4,
                "actor/policy_b/grad_norm": 5.0,
            }
        },
        checkpoint_dir=tmp_path,
        require_correction=False,
        require_multi_agent_routing=True,
        allow_local_verifier_fallback=True,
    )

    assert audit["status"] == "passed"
    assert audit["metrics"]["actor_loss"] == 0.4
    assert audit["metrics"]["actor_grad_norm"] == 5.0
    assert audit["metrics"]["actor_losses"] == {"policy_a": -0.2, "policy_b": 0.4}
    assert audit["metrics"]["actor_grad_norms"] == {"policy_a": 3.0, "policy_b": 5.0}


def test_fidelity_acceptance_requires_acknowledged_multi_actor_weight_sync(tmp_path):
    checkpoint_root = tmp_path / "checkpoints"
    manifest = checkpoint_root / "global_step_2" / "multi_actor_weight_sync.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "group_ids": ["policy_a", "policy_b"],
                "synchronized": True,
                "pending_groups": [],
                "versions": {
                    "policy_a": {"global_step": 2, "rollout_synced_step": 2},
                    "policy_b": {"global_step": 2, "rollout_synced_step": 2},
                },
            }
        ),
        encoding="utf-8",
    )

    audit = audit_fidelity_training_run(
        tmp_path,
        metric_summary={"latest": {}},
        checkpoint_dir=checkpoint_root,
        require_correction=False,
        require_learning_signal=False,
        require_multi_actor_weight_sync=True,
    )

    assert audit["checks"]["multi_actor_weight_sync"] is True
    assert audit["weight_sync_manifest"]["pending_groups"] == []


def test_fidelity_acceptance_rejects_pending_multi_actor_weight_sync(tmp_path):
    checkpoint_root = tmp_path / "checkpoints"
    manifest = checkpoint_root / "global_step_1" / "multi_actor_weight_sync.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "group_ids": ["policy_a", "policy_b"],
                "synchronized": False,
                "pending_groups": ["policy_b"],
            }
        ),
        encoding="utf-8",
    )

    audit = audit_fidelity_training_run(
        tmp_path,
        metric_summary={"latest": {}},
        checkpoint_dir=checkpoint_root,
        require_correction=False,
        require_learning_signal=False,
        require_multi_actor_weight_sync=True,
    )

    assert audit["status"] == "failed"
    assert audit["checks"]["multi_actor_weight_sync"] is False


def test_multi_actor_routing_acceptance_can_separate_zero_signal_from_route_failure(tmp_path):
    online = tmp_path / "trajectories" / "online_turns"
    online.mkdir(parents=True)
    rows = [
        {
            "policy_group": "policy_a",
            "agent_id": "generator",
            "reward_score": 0.0,
            "metadata": {"tree_id": "t", "verification_mode": "local_subprocess_fallback"},
        },
        {
            "policy_group": "policy_b",
            "agent_id": "critic",
            "reward_score": 0.0,
            "metadata": {"tree_id": "t", "verification_mode": "local_subprocess_fallback"},
        },
    ]
    (online / "worker.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    checkpoint = tmp_path / "global_step_1" / "actors" / "policy_a" / "model_world_size_1_rank_0.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    audit = audit_fidelity_training_run(
        tmp_path,
        metric_summary={"latest": {}},
        checkpoint_dir=tmp_path,
        require_correction=False,
        require_learning_signal=False,
        require_multi_agent_routing=True,
        allow_local_verifier_fallback=True,
    )
    assert audit["status"] == "passed"
    assert audit["learning_signal_required"] is False
    assert audit["checks"]["multi_agent_routing"] is True
    assert "finite_rollout_correction" not in audit["checks"]


def test_checkpoint_dir_is_read_from_verl_overrides():
    config = {"verl": {"overrides": ["trainer.default_local_dir=outputs/checkpoints"]}}

    assert checkpoint_dir_from_overrides(config).as_posix() == "outputs/checkpoints"


def test_run_scoped_checkpoint_dir_resolves_from_runtime_directory(tmp_path):
    config = {
        "verl": {
            "overrides": ["trainer.default_local_dir=${oc.env:TRAJWEAVE_RUN_DIR}/checkpoints"],
        }
    }

    assert checkpoint_dir_from_overrides(config, run_dir=tmp_path / "run") == tmp_path / "run" / "checkpoints"
