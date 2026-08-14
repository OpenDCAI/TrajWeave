from __future__ import annotations

from trajweave.backends.verl.trainers.comlrl_iterative import (
    IterationState,
    IterationStateStore,
    IterativeConfig,
    MADPOIterWorkflow,
    MARLHFIterWorkflow,
    PolicyLifecycle,
    PromptBatch,
    RewardScorer,
    RewardScores,
)


def register_trajweave_trainers() -> None:
    from trajweave.backends.verl.trainers import (
        c3_critic_sync,  # noqa: F401
        comlrl_staged,  # noqa: F401
        joint_preference_sync,  # noqa: F401
        maporl_multi_actor,  # noqa: F401
        mrlx_async,  # noqa: F401
        multi_actor_critic_sync,  # noqa: F401
        multi_actor_preference_sync,  # noqa: F401
        multi_actor_sync,  # noqa: F401
    )


__all__ = [
    "IterationState",
    "IterationStateStore",
    "IterativeConfig",
    "MADPOIterWorkflow",
    "MARLHFIterWorkflow",
    "PolicyLifecycle",
    "PromptBatch",
    "RewardScorer",
    "RewardScores",
    "register_trajweave_trainers",
]
