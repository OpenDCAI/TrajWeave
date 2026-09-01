"""Per-policy trajectory ownership for asynchronous multi-actor updates.

This module is deliberately independent of TransferQueue's storage backend.  A
trainer can feed completed tree records into :class:`PolicyBufferCoordinator`
after reading them from TQ, while CPU fixtures can exercise ownership,
filtering, and staleness without Ray or a running metadata server.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


@dataclass
class PolicyBufferState:
    group_id: str
    actor_step: int = 0
    rollout_synced_step: int = 0
    queue: deque[dict[str, Any]] = field(default_factory=deque, repr=False)
    accepted: int = 0
    filtered: int = 0
    stale: int = 0
    dropped: int = 0

    @property
    def buffer_depth(self) -> int:
        return len(self.queue)

    @property
    def readiness(self) -> bool:
        return bool(self.queue)


class PolicyBufferCoordinator:
    """Own one FIFO buffer and version clock per policy group.

    ``add_tree`` is atomic at tree level: reward filtering is evaluated on the
    completed tree mean before any sample is assigned to a policy queue.  A
    sample whose rollout version is older than ``max_policy_lag`` is dropped,
    never silently trained.
    """

    def __init__(
        self,
        policy_groups: Iterable[str],
        *,
        min_batch_size: int = 1,
        reward_range: tuple[float, float] | None = None,
        max_policy_lag: int = 1,
    ) -> None:
        groups = tuple(str(group) for group in policy_groups)
        if not groups or len(set(groups)) != len(groups):
            raise ValueError("policy_groups must contain unique non-empty ids")
        if min_batch_size <= 0:
            raise ValueError("min_batch_size must be positive")
        if max_policy_lag < 0:
            raise ValueError("max_policy_lag must be non-negative")
        if reward_range is not None and reward_range[0] >= reward_range[1]:
            raise ValueError("reward_range must be strictly increasing")
        self.policy_groups = groups
        self.min_batch_size = int(min_batch_size)
        self.reward_range = reward_range
        self.max_policy_lag = int(max_policy_lag)
        self.states = {group: PolicyBufferState(group) for group in groups}
        self.metrics: dict[str, int] = {
            "trees_seen": 0,
            "trees_filtered": 0,
            "samples_accepted": 0,
            "samples_filtered": 0,
            "samples_stale": 0,
            "samples_unknown_group": 0,
        }

    def update_actor_step(self, group_id: str, step: int) -> None:
        state = self._state(group_id)
        step = int(step)
        if step < state.actor_step:
            raise ValueError(f"actor step for {group_id!r} moved backwards")
        state.actor_step = step

    def mark_rollout_sync(self, group_id: str, step: int) -> None:
        state = self._state(group_id)
        step = int(step)
        if step > state.actor_step:
            raise ValueError(f"rollout sync step {step} exceeds actor step {state.actor_step}")
        if step < state.rollout_synced_step:
            raise ValueError(f"rollout sync step for {group_id!r} moved backwards")
        state.rollout_synced_step = step

    def add_tree(
        self, samples: Iterable[Mapping[str, Any]], *, global_step: int | None = None
    ) -> dict[str, int | float | bool]:
        rows = [dict(sample) for sample in samples]
        self.metrics["trees_seen"] += 1
        if not rows:
            return {"accepted": 0, "filtered": 0, "stale": 0, "tree_filtered": True, "tree_mean_reward": 0.0}
        rewards = [float(row.get("raw_score", row.get("reward", 0.0))) for row in rows]
        mean_reward = sum(rewards) / len(rewards)
        if self.reward_range is not None:
            lower, upper = self.reward_range
            keep_tree = lower < mean_reward < upper
        else:
            keep_tree = True
        if not keep_tree:
            self.metrics["trees_filtered"] += 1
            self.metrics["samples_filtered"] += len(rows)
            for row in rows:
                group = str(row.get("policy_group", row.get("worker_group", "")))
                if group in self.states:
                    self.states[group].filtered += 1
            return {
                "accepted": 0,
                "filtered": len(rows),
                "stale": 0,
                "tree_filtered": True,
                "tree_mean_reward": mean_reward,
            }

        accepted = stale = filtered = 0
        for row in rows:
            group = str(row.get("policy_group", row.get("worker_group", "")))
            if group not in self.states:
                self.metrics["samples_unknown_group"] += 1
                filtered += 1
                continue
            state = self.states[group]
            rollout_step = int(
                row.get(
                    "rollout_policy_step",
                    row.get(
                        "policy_step",
                        row.get(
                            "rollout_global_step", global_step if global_step is not None else state.rollout_synced_step
                        ),
                    ),
                )
            )
            rollout_global = int(
                row.get("rollout_global_step", global_step if global_step is not None else rollout_step)
            )
            lag = max(0, state.actor_step - rollout_step)
            row.update({"rollout_policy_step": rollout_step, "rollout_global_step": rollout_global, "policy_lag": lag})
            if lag > self.max_policy_lag:
                stale += 1
                state.stale += 1
                self.metrics["samples_stale"] += 1
                continue
            state.queue.append(row)
            state.accepted += 1
            accepted += 1
        self.metrics["samples_accepted"] += accepted
        self.metrics["samples_filtered"] += filtered
        return {
            "accepted": accepted,
            "filtered": filtered,
            "stale": stale,
            "tree_filtered": False,
            "tree_mean_reward": mean_reward,
        }

    def ready_groups(self) -> tuple[str, ...]:
        return tuple(group for group in self.policy_groups if self.states[group].buffer_depth >= self.min_batch_size)

    def pop_batch(self, group_id: str, batch_size: int | None = None) -> list[dict[str, Any]]:
        state = self._state(group_id)
        size = self.min_batch_size if batch_size is None else int(batch_size)
        if size <= 0:
            raise ValueError("batch_size must be positive")
        if len(state.queue) < size:
            return []
        return [state.queue.popleft() for _ in range(size)]

    def discard_pending(self) -> int:
        count = sum(len(state.queue) for state in self.states.values())
        for state in self.states.values():
            state_count = len(state.queue)
            state.queue.clear()
            state.dropped += state_count
        return count

    @property
    def pending(self) -> int:
        return sum(state.buffer_depth for state in self.states.values())

    def snapshot(self) -> dict[str, Any]:
        return {
            "policy_groups": list(self.policy_groups),
            "min_batch_size": self.min_batch_size,
            "reward_range": list(self.reward_range) if self.reward_range is not None else None,
            "max_policy_lag": self.max_policy_lag,
            "states": {
                group: {
                    "actor_step": state.actor_step,
                    "rollout_synced_step": state.rollout_synced_step,
                    "buffer_depth": state.buffer_depth,
                    "accepted": state.accepted,
                    "filtered": state.filtered,
                    "stale": state.stale,
                    "dropped": state.dropped,
                }
                for group, state in self.states.items()
            },
            "metrics": dict(self.metrics),
            "pending": self.pending,
        }

    def restore_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        """Restore version/counter metadata while intentionally dropping queues."""
        if tuple(str(group) for group in snapshot.get("policy_groups", ())) != self.policy_groups:
            raise ValueError("buffer snapshot policy groups do not match coordinator")
        for group, values in (snapshot.get("states", {}) or {}).items():
            state = self._state(group)
            state.actor_step = int(values.get("actor_step", 0))
            state.rollout_synced_step = int(values.get("rollout_synced_step", 0))
            state.accepted = int(values.get("accepted", 0))
            state.filtered = int(values.get("filtered", 0))
            state.stale = int(values.get("stale", 0))
            state.dropped = int(values.get("dropped", 0))
            state.queue.clear()
        self.metrics.update({key: int(value) for key, value in (snapshot.get("metrics", {}) or {}).items()})

    def metric_fields(self) -> dict[str, float | int]:
        fields: dict[str, float | int] = {f"buffer/{key}": value for key, value in self.metrics.items()}
        fields["buffer/pending"] = self.pending
        for group, state in self.states.items():
            prefix = f"buffer/{group}"
            fields[f"{prefix}/depth"] = state.buffer_depth
            fields[f"{prefix}/actor_step"] = state.actor_step
            fields[f"{prefix}/rollout_synced_step"] = state.rollout_synced_step
            fields[f"{prefix}/pass_rate"] = state.accepted / max(1, state.accepted + state.filtered + state.stale)
            fields[f"{prefix}/policy_lag"] = max(0, state.actor_step - state.rollout_synced_step)
        return fields

    def _state(self, group_id: str) -> PolicyBufferState:
        group = str(group_id)
        if group not in self.states:
            raise KeyError(f"Unknown policy group {group!r}; known groups: {self.policy_groups}")
        return self.states[group]


PerPolicyBufferCoordinator = PolicyBufferCoordinator
TransferQueueBufferCoordinator = PolicyBufferCoordinator

__all__ = [
    "PolicyBufferCoordinator",
    "PerPolicyBufferCoordinator",
    "TransferQueueBufferCoordinator",
    "PolicyBufferState",
    "async_buffer_acceptance",
    "run_asymmetric_three_step_fixture",
]


def async_buffer_acceptance(
    result: Mapping[str, Any],
    *,
    require_stale_sample: bool = True,
    require_two_actors: bool = True,
    max_policy_lag: int = 1,
) -> dict[str, Any]:
    """Audit the mechanism-level async-buffer fixture.

    This intentionally checks ownership/version semantics only.  It does not
    claim that an optimizer or a live vLLM endpoint ran.
    """

    updates = {str(group): int(value) for group, value in (result.get("updates", {}) or {}).items()}
    metrics = {str(key): value for key, value in (result.get("metrics", {}) or {}).items()}
    pending = int(result.get("pending", metrics.get("buffer/pending", 0)))
    stale = int(metrics.get("buffer/samples_stale", 0))
    observed_lag = max(
        (int(value) for key, value in metrics.items() if key.endswith("/policy_lag")),
        default=0,
    )
    checks = {
        "pending_empty": pending == 0,
        "stale_samples_accounted": stale > 0 if require_stale_sample else True,
        "policy_lag_within_threshold": observed_lag <= int(max_policy_lag),
        "two_actors_updated": min(updates.values(), default=0) > 0 if require_two_actors else True,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "updates": updates,
        "stale_samples": stale,
        "max_policy_lag": observed_lag,
        "pending": pending,
    }


def run_asymmetric_three_step_fixture() -> dict[str, Any]:
    """Deterministic CPU fixture for two independently updated actors.

    ``policy_b`` intentionally receives a sample two versions behind at the
    final wave.  That sample must be rejected as stale while earlier delayed
    samples remain trainable, proving that the lag boundary is strict.
    """
    coordinator = PolicyBufferCoordinator(("policy_a", "policy_b"), min_batch_size=1, max_policy_lag=1)
    updates = {"policy_a": 0, "policy_b": 0}
    for global_step in range(3):
        coordinator.update_actor_step("policy_a", updates["policy_a"])
        coordinator.update_actor_step("policy_b", updates["policy_b"])
        coordinator.add_tree(
            [
                {
                    "policy_group": "policy_a",
                    "reward": 0.75,
                    "rollout_policy_step": global_step,
                    "rollout_global_step": global_step,
                },
                {
                    "policy_group": "policy_b",
                    "reward": 0.25,
                    "rollout_policy_step": 0 if global_step == 2 else max(0, global_step - 1),
                    "rollout_global_step": global_step,
                },
            ],
            global_step=global_step,
        )
        for group in coordinator.ready_groups():
            batch = coordinator.pop_batch(group)
            if batch:
                updates[group] += 1
                coordinator.update_actor_step(group, updates[group])
                coordinator.mark_rollout_sync(group, updates[group])
    result = {
        "updates": updates,
        "pending": coordinator.pending,
        "metrics": coordinator.metric_fields(),
        "snapshot": coordinator.snapshot(),
    }
    result["acceptance"] = async_buffer_acceptance(result)
    return result
