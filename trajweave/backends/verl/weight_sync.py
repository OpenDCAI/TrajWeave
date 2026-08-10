from __future__ import annotations

import asyncio
import hashlib
import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class PolicyWeightVersion:
    """Immutable record for one actor policy version."""

    group_id: str
    global_step: int
    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None
    rollout_synced_step: int | None = None


class PolicyWeightTransport(Protocol):
    """Per-policy-group rollout transport implemented by a backend adapter."""

    def load_policy(self, version: PolicyWeightVersion) -> bool:
        """Load a policy version and return true only after engine acknowledgement."""


@dataclass(frozen=True)
class CheckpointEnginePolicyEndpoint:
    """Adapt one VERL checkpoint engine to the grouped transport contract.

    VERL's ``CheckpointEngineManager.update_weights`` currently signals success
    by returning normally rather than returning a boolean acknowledgement. The
    adapter treats that completed call as an acknowledgement and preserves the
    exact policy-group routing at the ``GroupedPolicyWeightTransport`` layer.
    """

    manager: Any

    def load_policy(self, version: PolicyWeightVersion) -> bool:
        update = self.manager.update_weights(global_steps=version.global_step)
        if inspect.isawaitable(update):
            update = _run_awaitable(update)
        return True if update is None else bool(update)


@dataclass(frozen=True)
class GroupedPolicyWeightTransport:
    """Route each policy version to its explicitly bound rollout endpoint.

    Endpoint objects intentionally own backend-specific details such as Ray
    handles, HTTP clients, or vLLM IPC. This adapter only enforces that a
    policy cannot silently fall back to another group's server.
    """

    endpoints: Mapping[str, PolicyWeightTransport]

    def __post_init__(self) -> None:
        normalized = {str(group_id): endpoint for group_id, endpoint in self.endpoints.items()}
        if not normalized or any(not group_id for group_id in normalized):
            raise ValueError("Grouped policy transport requires non-empty endpoint bindings.")
        object.__setattr__(self, "endpoints", normalized)

    def load_policy(self, version: PolicyWeightVersion) -> bool:
        endpoint = self.endpoints.get(version.group_id)
        if endpoint is None:
            raise KeyError(
                f"No rollout endpoint is bound to policy group {version.group_id!r}; "
                f"known groups: {tuple(self.endpoints)}."
            )
        return bool(endpoint.load_policy(version))


def _run_awaitable(value: Any) -> Any:
    """Resolve an async checkpoint-engine call from the synchronous contract."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(value)
    raise RuntimeError("Checkpoint engine returned an awaitable while an event loop is already running.")


@dataclass(frozen=True)
class WeightSyncResult:
    group_id: str
    global_step: int
    acknowledged: bool
    error: str | None = None


@dataclass
class MultiActorWeightSyncContract:
    """Track actor updates separately from rollout-engine synchronization.

    The contract deliberately does not perform transport. A future vLLM
    implementation can consume ``pending_groups`` and call
    ``mark_rollout_sync`` only after the corresponding policy is loaded.
    """

    group_ids: tuple[str, ...]
    _versions: dict[str, PolicyWeightVersion] = field(default_factory=dict, init=False)
    _sampling_ranges: dict[str, tuple[int, int]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        normalized = tuple(str(group_id) for group_id in self.group_ids)
        if not normalized or any(not group_id for group_id in normalized):
            raise ValueError("Multi-actor weight sync requires non-empty group ids.")
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"Duplicate multi-actor weight sync group ids: {normalized}")
        self.group_ids = normalized

    def record_actor_update(self, group_id: str, global_step: int) -> PolicyWeightVersion:
        group_id = self._require_group(group_id)
        global_step = int(global_step)
        if global_step < 0:
            raise ValueError("global_step must be non-negative.")
        previous = self._versions.get(group_id)
        if previous is not None and global_step < previous.global_step:
            raise ValueError(
                f"Weight version for {group_id!r} moved backwards from "
                f"{previous.global_step} to {global_step}."
            )
        same_version = previous is not None and global_step == previous.global_step
        version = PolicyWeightVersion(
            group_id=group_id,
            global_step=global_step,
            checkpoint_path=previous.checkpoint_path if same_version else None,
            checkpoint_sha256=previous.checkpoint_sha256 if same_version else None,
            rollout_synced_step=previous.rollout_synced_step if previous else None,
        )
        self._versions[group_id] = version
        return version

    def record_checkpoint(self, group_id: str, checkpoint_path: str | Path) -> PolicyWeightVersion:
        group_id = self._require_group(group_id)
        current = self._versions.get(group_id)
        if current is None:
            raise ValueError(f"Cannot attach a checkpoint before recording actor update for {group_id!r}.")
        path = Path(checkpoint_path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        version = PolicyWeightVersion(
            group_id=group_id,
            global_step=current.global_step,
            checkpoint_path=str(path),
            checkpoint_sha256=digest,
            rollout_synced_step=current.rollout_synced_step,
        )
        self._versions[group_id] = version
        return version

    def mark_rollout_sync(self, group_id: str, global_step: int) -> PolicyWeightVersion:
        group_id = self._require_group(group_id)
        current = self._versions.get(group_id)
        if current is None:
            raise ValueError(f"Cannot sync {group_id!r} before recording an actor update.")
        global_step = int(global_step)
        if global_step != current.global_step:
            raise ValueError(
                f"Rollout sync for {group_id!r} targets step {global_step}, "
                f"but actor version is {current.global_step}."
            )
        version = PolicyWeightVersion(
            group_id=group_id,
            global_step=current.global_step,
            checkpoint_path=current.checkpoint_path,
            checkpoint_sha256=current.checkpoint_sha256,
            rollout_synced_step=global_step,
        )
        self._versions[group_id] = version
        return version

    def record_sampling_range(self, group_id: str, min_step: int, max_step: int) -> None:
        """Record the rollout policy-version range consumed by one update."""
        self._require_group(group_id)
        bounds = (int(min_step), int(max_step))
        if bounds[0] > bounds[1]:
            raise ValueError("sampling range min_step must not exceed max_step")
        self._sampling_ranges[str(group_id)] = bounds

    def sync_pending(self, transport: PolicyWeightTransport) -> tuple[WeightSyncResult, ...]:
        """Attempt each pending group and advance only acknowledged versions."""
        results = []
        for group_id in self.pending_groups:
            version = self._versions.get(group_id)
            if version is None:
                results.append(
                    WeightSyncResult(
                        group_id=group_id,
                        global_step=-1,
                        acknowledged=False,
                        error="actor version is not available",
                    )
                )
                continue
            try:
                acknowledged = bool(transport.load_policy(version))
            except Exception as exc:  # transport failures remain observable state
                results.append(
                    WeightSyncResult(
                        group_id=group_id,
                        global_step=version.global_step,
                        acknowledged=False,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
            if acknowledged:
                self.mark_rollout_sync(group_id, version.global_step)
            results.append(
                WeightSyncResult(
                    group_id=group_id,
                    global_step=version.global_step,
                    acknowledged=acknowledged,
                )
            )
        return tuple(results)

    @property
    def pending_groups(self) -> tuple[str, ...]:
        return tuple(
            group_id
            for group_id in self.group_ids
            if group_id not in self._versions
            or self._versions[group_id].rollout_synced_step != self._versions[group_id].global_step
        )

    @property
    def synchronized(self) -> bool:
        return not self.pending_groups and len(self._versions) == len(self.group_ids)

    def metric_fields(self) -> dict[str, int]:
        return {
            "trajweave/maporl/weight_sync/actor_versions": len(self._versions),
            "trajweave/maporl/weight_sync/rollout_synced": sum(
                version.rollout_synced_step == version.global_step for version in self._versions.values()
            ),
            "trajweave/maporl/weight_sync/pending": len(self.pending_groups),
            "trajweave/maporl/weight_sync/synchronized": int(self.synchronized),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "group_ids": list(self.group_ids),
            "synchronized": self.synchronized,
            "pending_groups": list(self.pending_groups),
            "versions": {
                group_id: {
                    "global_step": version.global_step,
                    "checkpoint_path": version.checkpoint_path,
                    "checkpoint_sha256": version.checkpoint_sha256,
                    "rollout_synced_step": version.rollout_synced_step,
                }
                for group_id, version in self._versions.items()
            },
            "sampling_ranges": {
                group_id: {"min_step": bounds[0], "max_step": bounds[1]}
                for group_id, bounds in self._sampling_ranges.items()
            },
            "sync_confirmation": {
                group_id: bool(version.rollout_synced_step == version.global_step)
                for group_id, version in self._versions.items()
            },
        }

    def _require_group(self, group_id: str) -> str:
        group_id = str(group_id)
        if group_id not in self.group_ids:
            raise KeyError(f"Unknown weight sync group {group_id!r}; known groups: {self.group_ids}.")
        return group_id
