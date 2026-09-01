from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def audit_fidelity_training_run(
    run_dir: str | Path,
    *,
    metric_summary: dict[str, Any],
    checkpoint_dir: str | Path | None,
    require_correction: bool = True,
    require_learning_signal: bool = True,
    require_multi_agent_routing: bool = False,
    require_multi_actor_weight_sync: bool = False,
    allow_local_verifier_fallback: bool = False,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    rewards_by_tree: dict[str, list[float]] = defaultdict(list)
    verifier_modes: list[str] = []
    failure_types: list[str | None] = []
    trajectory_files = sorted((run_path / "trajectories" / "online_turns").glob("*.jsonl"))
    for path in trajectory_files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row.get("metadata") or {}
            tree_id = metadata.get("tree_id")
            reward = row.get("reward_score")
            if tree_id is None or not isinstance(reward, int | float):
                continue
            rewards_by_tree[str(tree_id)].append(float(reward))
            verifier_modes.append(str(metadata.get("verification_mode") or ""))
            failure_types.append(metadata.get("failure_type"))

    latest = metric_summary.get("latest", {}) or {}
    metric_history = _metric_history(run_path)
    advantage_max = _history_extreme(
        metric_history,
        "critic/advantages/max",
        fallback=latest.get("critic/advantages/max"),
        largest=True,
    )
    advantage_min = _history_extreme(
        metric_history,
        "critic/advantages/min",
        fallback=latest.get("critic/advantages/min"),
        largest=False,
    )
    actor_losses = _actor_group_metrics_over_run(metric_history, latest, metric_suffix="loss")
    actor_grad_norms = _actor_group_metrics_over_run(metric_history, latest, metric_suffix="grad_norm")
    actor_loss = _largest_magnitude(actor_losses.values())
    actor_grad_norm = _largest_magnitude(actor_grad_norms.values())
    correction_values = [
        _finite_float(latest.get("rollout_corr/rollout_is_mean")),
        _finite_float(latest.get("rollout_corr/rollout_is_min")),
        _finite_float(latest.get("rollout_corr/rollout_is_max")),
    ]
    reward_ranges = {
        tree_id: {"rewards": rewards, "min": min(rewards), "max": max(rewards)}
        for tree_id, rewards in rewards_by_tree.items()
        if rewards
    }
    checkpoint_files = _checkpoint_model_files(checkpoint_dir)
    weight_sync_manifest = _latest_weight_sync_manifest(checkpoint_dir)
    policy_groups = {
        str(row.get("policy_group"))
        for path in trajectory_files
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
        if row.get("policy_group")
    }
    agent_ids = {
        str(row.get("agent_id"))
        for path in trajectory_files
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
        if row.get("agent_id")
    }
    accepted_verifier_modes = {"prime_code_subprocess"}
    if allow_local_verifier_fallback:
        accepted_verifier_modes.add("local_subprocess_fallback")
    checks: dict[str, bool] = {
        "real_verifier_without_errors": bool(verifier_modes)
        and all(mode in accepted_verifier_modes for mode in verifier_modes)
        and all(failure_type != "verifier_error" for failure_type in failure_types),
        "nonzero_verifier_reward": any(reward > 0 for rewards in rewards_by_tree.values() for reward in rewards),
        "mixed_rewards_within_tree": any(max(rewards) > min(rewards) for rewards in rewards_by_tree.values()),
        "finite_nonzero_advantage": advantage_max is not None
        and advantage_min is not None
        and max(abs(advantage_max), abs(advantage_min)) > 0,
        "finite_nonzero_actor_loss": actor_loss is not None and abs(actor_loss) > 0,
        "finite_nonzero_gradient": actor_grad_norm is not None and actor_grad_norm > 0,
        "checkpoint_present": bool(checkpoint_files),
    }
    if require_multi_agent_routing:
        checks["multi_agent_routing"] = len(policy_groups) >= 2 and len(agent_ids) >= 2
    if require_multi_actor_weight_sync:
        checks["multi_actor_weight_sync"] = (
            bool(weight_sync_manifest)
            and bool(weight_sync_manifest.get("synchronized"))
            and not weight_sync_manifest.get("pending_groups")
            and len(weight_sync_manifest.get("group_ids", [])) >= 2
        )
    if not require_learning_signal:
        for name in (
            "nonzero_verifier_reward",
            "mixed_rewards_within_tree",
            "finite_nonzero_advantage",
            "finite_nonzero_actor_loss",
            "finite_nonzero_gradient",
        ):
            checks.pop(name, None)
    if require_correction:
        checks["finite_rollout_correction"] = all(value is not None for value in correction_values)
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "reward_groups": reward_ranges,
        "metrics": {
            "advantage_max": advantage_max,
            "advantage_min": advantage_min,
            "actor_loss": actor_loss,
            "actor_grad_norm": actor_grad_norm,
            "actor_losses": actor_losses,
            "actor_grad_norms": actor_grad_norms,
            "rollout_is_mean": correction_values[0],
            "rollout_is_min": correction_values[1],
            "rollout_is_max": correction_values[2],
        },
        "trajectory_files": [str(path) for path in trajectory_files],
        "checkpoint_files": [str(path) for path in checkpoint_files],
        "weight_sync_manifest": weight_sync_manifest,
        "correction_required": require_correction,
        "learning_signal_required": require_learning_signal,
        "multi_agent_routing_required": require_multi_agent_routing,
        "multi_actor_weight_sync_required": require_multi_actor_weight_sync,
        "local_verifier_fallback_allowed": allow_local_verifier_fallback,
        "tensor_diff_required": True,
    }


def checkpoint_dir_from_overrides(config: dict[str, Any], *, run_dir: str | Path | None = None) -> Path | None:
    for override in config.get("verl", {}).get("overrides", []):
        text = str(override)
        if text.startswith("trainer.default_local_dir="):
            value = text.split("=", 1)[1]
            if value.startswith("${oc.env:TRAJWEAVE_RUN_DIR}"):
                if run_dir is None:
                    return None
                value = f"{run_dir}{value.removeprefix('${oc.env:TRAJWEAVE_RUN_DIR}')}"
            return Path(value)
    return None


def _checkpoint_model_files(checkpoint_dir: str | Path | None) -> list[Path]:
    if checkpoint_dir is None:
        return []
    root = Path(checkpoint_dir)
    single_actor = root.glob("global_step_*/actor/model_world_size_*_rank_*.pt")
    multi_actor = root.glob("global_step_*/actors/*/model_world_size_*_rank_*.pt")
    return sorted([*single_actor, *multi_actor])


def _latest_weight_sync_manifest(checkpoint_dir: str | Path | None) -> dict[str, Any] | None:
    if checkpoint_dir is None:
        return None
    manifests = list(Path(checkpoint_dir).glob("global_step_*/multi_actor_weight_sync.json"))
    if not manifests:
        return None

    def global_step(path: Path) -> int:
        try:
            return int(path.parent.name.removeprefix("global_step_"))
        except ValueError:
            return -1

    latest = max(manifests, key=global_step)
    data = json.loads(latest.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def _finite_float(value: Any) -> float | None:
    if not isinstance(value, int | float):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _actor_group_metrics(latest: dict[str, Any], *, metric_suffix: str) -> dict[str, float]:
    generic_key = f"actor/{metric_suffix}"
    metrics: dict[str, float] = {}
    generic = _finite_float(latest.get(generic_key))
    if generic is not None:
        metrics["shared"] = generic
    for key, value in latest.items():
        parts = str(key).split("/")
        if len(parts) != 3 or parts[0] != "actor" or parts[2] != metric_suffix:
            continue
        numeric = _finite_float(value)
        if numeric is not None:
            metrics[parts[1]] = numeric
    return metrics


def _metric_history(run_path: Path) -> dict[str, list[float]]:
    history: dict[str, list[float]] = defaultdict(list)
    path = run_path / "metrics" / "metrics.jsonl"
    if not path.exists():
        return history
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        numeric = _finite_float(row.get("value"))
        if numeric is not None:
            history[str(row.get("name"))].append(numeric)
    return history


def _history_extreme(
    history: dict[str, list[float]],
    name: str,
    *,
    fallback: Any,
    largest: bool,
) -> float | None:
    values = history.get(name, [])
    if values:
        return max(values) if largest else min(values)
    return _finite_float(fallback)


def _actor_group_metrics_over_run(
    history: dict[str, list[float]], latest: dict[str, Any], *, metric_suffix: str
) -> dict[str, float]:
    metrics = _actor_group_metrics(latest, metric_suffix=metric_suffix)
    generic_key = f"actor/{metric_suffix}"
    if history.get(generic_key):
        metrics["shared"] = _largest_magnitude(history[generic_key])
    for key, values in history.items():
        parts = key.split("/")
        if len(parts) == 3 and parts[0] == "actor" and parts[2] == metric_suffix and values:
            metrics[parts[1]] = _largest_magnitude(values)
    return metrics


def _largest_magnitude(values: Any) -> float | None:
    values = list(values)
    return max(values, key=abs) if values else None
