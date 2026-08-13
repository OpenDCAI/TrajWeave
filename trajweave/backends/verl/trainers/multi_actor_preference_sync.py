from __future__ import annotations

from trajweave.backends.verl.trainers.joint_preference_sync import (
    MADPOLossConfig,
    TrajWeaveJointPreferenceSyncTrainer,
    validate_joint_preference_training_contract,
)
from verl.trainer.ppo.v1.trainer_base import register_trainer


@register_trainer("trajweave_multi_actor_preference_sync")
class TrajWeaveMultiActorPreferenceSyncTrainer(TrajWeaveJointPreferenceSyncTrainer):
    """Compatibility trainer name for the joint sequential MADPO implementation."""


__all__ = [
    "MADPOLossConfig",
    "TrajWeaveMultiActorPreferenceSyncTrainer",
    "validate_joint_preference_training_contract",
]
