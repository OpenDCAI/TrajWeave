"""Canonical import surface for TrajWeave's generic multi-actor trainer."""

from trajweave.backends.verl.trainers.maporl_multi_actor import (
    TrajWeaveMAPoRLMultiActorSyncTrainer,
    TrajWeaveMultiActorSyncTrainer,
    WorkerGroupConfig,
)

__all__ = [
    "TrajWeaveMultiActorSyncTrainer",
    "TrajWeaveMAPoRLMultiActorSyncTrainer",
    "WorkerGroupConfig",
]
