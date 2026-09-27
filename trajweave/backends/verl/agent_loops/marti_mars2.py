from __future__ import annotations

from typing import Any

from trajweave.backends.verl.runtime_config import as_bool, config_get

SUPPORTED_RECIPES = ("marti_mars2_single_mcts",)


def load_worker_class():
    """Load the native VERL/vLLM MARTI worker lazily to avoid import cycles."""
    from trajweave.backends.verl.agent_loop import TrajWeaveVLLMAgentLoopWorkerTQ

    return TrajWeaveVLLMAgentLoopWorkerTQ


def vllm_sampling_params(config: Any, sampling_params: dict[str, Any], *, policy_group: str) -> dict[str, Any]:
    """Attach the grouped-policy marker only when the vLLM router consumes it."""
    params = dict(sampling_params)
    trajweave = config_get(config, "trajweave", {})
    multi_actor = config_get(trajweave, "multi_actor", {})
    vllm = config_get(multi_actor, "vllm", {})
    if as_bool(config_get(vllm, "enabled", False)):
        params["trajweave_policy_group"] = str(policy_group)
    return params
