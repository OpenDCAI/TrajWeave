from __future__ import annotations

from trajweave.backends.verl.trainers.multi_actor_sync import (
    TrajWeaveMultiActorSyncTrainer,
    WorkerGroupConfig,
    _actor_rollout_ref_config_for_group,
    _batch_len,
    _positive_gpu_count,
    _safe_metric_name,
    _safe_path_name,
    _worker_groups_from_config,
)
from verl.trainer.ppo.v1.trainer_base import register_trainer


@register_trainer("trajweave_maporl_multi_actor_sync")
class TrajWeaveMAPoRLMultiActorSyncTrainer(TrajWeaveMultiActorSyncTrainer):
    """MAPoRL 旧配置的兼容入口；实际能力由通用多 Actor 训练器提供。"""


__all__ = [
    "TrajWeaveMAPoRLMultiActorSyncTrainer",
    "WorkerGroupConfig",
    "_actor_rollout_ref_config_for_group",
    "_batch_len",
    "_positive_gpu_count",
    "_safe_metric_name",
    "_safe_path_name",
    "_worker_groups_from_config",
]
