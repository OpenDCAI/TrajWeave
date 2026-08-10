def _ensure_torch_dtensor_import_compat() -> None:
    """Expose DTensor on the path expected by this vendored VERL snapshot."""

    try:
        import builtins

        import torch.distributed.tensor as tensor_mod
        from torch.distributed._tensor import DTensor, Shard
        from torch.distributed._tensor.placement_types import DTensorSpec

        if not hasattr(tensor_mod, "DTensor"):
            tensor_mod.DTensor = DTensor
        if not hasattr(tensor_mod, "Shard"):
            tensor_mod.Shard = Shard
        for name, value in {
            "DTensor": DTensor,
            "Shard": Shard,
            "DTensorSpec": DTensorSpec,
        }.items():
            if not hasattr(builtins, name):
                setattr(builtins, name, value)
    except Exception:
        return


_ensure_torch_dtensor_import_compat()


__all__ = [
    "TRAJWEAVE_AGENT_LOOP_MANAGER_FQN",
    "TrajWeaveAgentLoopManager",
    "TrajWeaveAgentLoopRuntimeConfig",
    "VerlDataProtoAdapter",
    "VerlTrainerLaunchConfig",
    "VerlTrainerLauncher",
    "export_dataproto",
]


def __getattr__(name: str):
    if name == "VerlDataProtoAdapter":
        from trajweave.backends.verl.dataproto import VerlDataProtoAdapter

        return VerlDataProtoAdapter
    if name == "export_dataproto":
        from trajweave.backends.verl.export import export_dataproto

        return export_dataproto
    if name in {"VerlTrainerLaunchConfig", "VerlTrainerLauncher"}:
        from trajweave.backends.verl.launcher import VerlTrainerLaunchConfig, VerlTrainerLauncher

        return {"VerlTrainerLaunchConfig": VerlTrainerLaunchConfig, "VerlTrainerLauncher": VerlTrainerLauncher}[name]
    if name in {
        "TRAJWEAVE_AGENT_LOOP_MANAGER_FQN",
        "TrajWeaveAgentLoopManager",
        "TrajWeaveAgentLoopRuntimeConfig",
    }:
        from trajweave.backends.verl.agent_loop import (
            TRAJWEAVE_AGENT_LOOP_MANAGER_FQN,
            TrajWeaveAgentLoopManager,
            TrajWeaveAgentLoopRuntimeConfig,
        )

        return {
            "TRAJWEAVE_AGENT_LOOP_MANAGER_FQN": TRAJWEAVE_AGENT_LOOP_MANAGER_FQN,
            "TrajWeaveAgentLoopManager": TrajWeaveAgentLoopManager,
            "TrajWeaveAgentLoopRuntimeConfig": TrajWeaveAgentLoopRuntimeConfig,
        }[name]
    raise AttributeError(name)
from trajweave.backends.verl.async_buffer import (
    PerPolicyBufferCoordinator,
    PolicyBufferCoordinator,
    PolicyBufferState,
    TransferQueueBufferCoordinator,
    run_asymmetric_three_step_fixture,
)

__all__ = [
    "PerPolicyBufferCoordinator",
    "PolicyBufferCoordinator",
    "PolicyBufferState",
    "TransferQueueBufferCoordinator",
    "run_asymmetric_three_step_fixture",
]
