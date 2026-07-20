from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from trajweave.backends.verl.routing import safe_actor_role_key
from trajweave.backends.verl.runtime_config import TrajWeaveAgentLoopRuntimeConfig

logger = logging.getLogger(__name__)


def sync_hf_local_rollout_weights(trainer: Any) -> dict[str, str]:
    """把训练 Worker Group 的当前权重同步到 HF local AgentLoop。"""

    runtime = TrajWeaveAgentLoopRuntimeConfig.from_verl_config(trainer.config)
    if runtime.agent_loop_backend != "hf_local_tq":
        return {}
    manager = getattr(trainer, "agent_loop_manager", None)
    if manager is None:
        raise RuntimeError("HF local rollout weight sync requires trainer.agent_loop_manager.")

    global_step = int(trainer.global_steps)
    release_results = manager.release_local_models()
    if not release_results:
        raise RuntimeError("HF local rollout weight sync did not release any AgentLoop worker model cache.")
    root = Path(str(trainer.config.trainer.default_local_dir)) / "rollout_sync" / f"global_step_{global_step}"
    actor_wgs = getattr(trainer, "actor_rollout_wgs", None)
    if actor_wgs:
        model_paths = _export_multi_actor_snapshots(
            actor_wgs=actor_wgs,
            trainable_group_ids=set(
                getattr(
                    trainer,
                    "multi_actor_trainable_group_ids",
                    getattr(trainer, "maporl_trainable_group_ids", actor_wgs),
                )
            ),
            root=root,
            global_step=global_step,
        )
    else:
        model_paths = _export_single_actor_snapshot(
            actor_wg=trainer.actor_rollout_wg,
            root=root,
            global_step=global_step,
        )

    results = manager.reload_local_models(model_paths, policy_version=global_step)
    if not results:
        raise RuntimeError("HF local rollout weight sync did not reload any AgentLoop worker.")
    _prune_old_snapshots(root.parent, keep=2)
    logger.info(
        "TrajWeave HF local rollout weights synchronized at global_step=%d: %s",
        global_step,
        model_paths,
    )
    return model_paths


def _export_single_actor_snapshot(*, actor_wg: Any, root: Path, global_step: int) -> dict[str, str]:
    actor_path = root / "actor"
    actor_wg.export_hf_rollout_snapshot(str(actor_path), global_step, max_ckpt_to_keep=2)
    hf_path = _require_hf_snapshot(actor_path)
    return {"__default__": str(hf_path)}


def _export_multi_actor_snapshots(
    *,
    actor_wgs: dict[str, Any],
    trainable_group_ids: set[str],
    root: Path,
    global_step: int,
) -> dict[str, str]:
    model_paths: dict[str, str] = {}
    for group_id, actor_wg in actor_wgs.items():
        if group_id not in trainable_group_ids:
            continue
        group_path = root / safe_actor_role_key(group_id)
        actor_wg.export_hf_rollout_snapshot(str(group_path), global_step, max_ckpt_to_keep=2)
        model_paths[str(group_id)] = str(_require_hf_snapshot(group_path))
    if not model_paths:
        raise RuntimeError("No trainable TrajWeave worker group produced an HF rollout snapshot.")
    return model_paths


def _require_hf_snapshot(checkpoint_path: Path) -> Path:
    hf_path = checkpoint_path / "huggingface"
    if not (hf_path / "config.json").is_file():
        raise RuntimeError(f"HF rollout snapshot is incomplete: {hf_path}")
    weight_files = tuple(hf_path.glob("*.safetensors")) + tuple(hf_path.glob("pytorch_model*.bin"))
    if not weight_files:
        raise RuntimeError(f"HF rollout snapshot contains no model weights: {hf_path}")
    return hf_path


def _prune_old_snapshots(sync_root: Path, *, keep: int) -> None:
    snapshots = sorted(
        (path for path in sync_root.glob("global_step_*") if path.is_dir()),
        key=lambda path: _snapshot_step(path.name),
    )
    for path in snapshots[:-keep]:
        shutil.rmtree(path)


def _snapshot_step(name: str) -> int:
    try:
        return int(name.rsplit("_", 1)[-1])
    except ValueError:
        return -1
