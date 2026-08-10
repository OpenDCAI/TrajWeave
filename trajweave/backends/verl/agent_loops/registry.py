from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from trajweave.backends.verl.emitters.registry import supported_emitter_recipes


WorkerLoader = Callable[[], Any]


@dataclass(frozen=True)
class AgentLoopBackendSpec:
    name: str
    worker_loader: WorkerLoader | None
    supported_recipes: tuple[str, ...] | None


def _synthetic_worker():
    from trajweave.backends.verl.agent_loop import TrajWeaveSyntheticAgentLoopWorkerTQ

    return TrajWeaveSyntheticAgentLoopWorkerTQ


def _marti_worker():
    from trajweave.backends.verl.agent_loops.marti_mars2 import load_worker_class

    return load_worker_class()


def _registry() -> dict[str, AgentLoopBackendSpec]:
    recipes = tuple(sorted(supported_emitter_recipes()))
    return {
        "verl_tq": AgentLoopBackendSpec("verl_tq", None, ()),
        "synthetic_tq": AgentLoopBackendSpec("synthetic_tq", _synthetic_worker, recipes),
        "hf_local_tq": AgentLoopBackendSpec("hf_local_tq", _synthetic_worker, recipes),
        "vllm_marti_tq": AgentLoopBackendSpec("vllm_marti_tq", _marti_worker, ("marti_mars2_single_mcts",)),
    }


def agent_loop_backend_names() -> tuple[str, ...]:
    return tuple(_registry())


def validate_agent_loop_backend(recipe: str | None, backend: str) -> None:
    registry = _registry()
    if backend not in registry:
        raise ValueError(f"Unsupported TrajWeave AgentLoop backend: {backend}")
    if recipe and recipe not in supported_emitter_recipes():
        raise ValueError(f"Unsupported TrajWeave recipe for VERL AgentLoopManager: {recipe}")
    supported = registry[backend].supported_recipes
    if recipe and supported is not None and recipe not in supported:
        if backend == "verl_tq":
            raise ValueError(
                "TrajWeave MASRL recipes require a metadata-emitting AgentLoop backend; "
                "native verl_tq does not emit the fields required by recipe credit assignment."
            )
        raise ValueError(f"AgentLoop backend {backend!r} does not support recipe {recipe!r}.")


def load_agent_loop_worker(backend: str):
    spec = _registry().get(backend)
    if spec is None:
        raise ValueError(f"Unsupported TrajWeave AgentLoop backend: {backend}")
    return spec.worker_loader() if spec.worker_loader is not None else None
