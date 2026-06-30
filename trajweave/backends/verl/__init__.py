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
