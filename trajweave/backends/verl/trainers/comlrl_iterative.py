"""Orchestration-only controller for canonical CoMLRL iterative training.

The resumable workflow manifest is a TrajWeave extension. CoMLRL v1.4.1's
MADPOIter/MARLHFIter trainers do not provide optimizer or RNG resume support.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol

from trajweave.backends.verl.trainers.comlrl_staged import resolve_marlhf_rl_dispatch
from trajweave.core.preference import JointPreferencePair
from trajweave.credit.comlrl.iterative import compare_policy_candidates_by_index, select_policy_comparisons
from trajweave.orchestration.comlrl.comparator import (
    Comparator,
    FrozenPolicySnapshot,
    PolicyComparator,
    RemoteComparator,
    canonical_comparator_policy,
    canonical_generation_mode,
)
from trajweave.storage.preference_replay import (
    PolicySnapshotStore,
    PreferenceReplayRecord,
    PreferenceReplayStore,
    canonical_replay_mode,
)
from trajweave.storage.serialization import redact_secrets

Algorithm = Literal["MADPOIter", "MARLHFIter"]
RunMode = Literal["fresh", "resume"]
_WORKFLOW_SCHEMA_VERSION = 1
_RESUME_EXTENSION = "TrajWeave extension; absent from upstream CoMLRL v1.4.1"
_SAFE_STATES = {"ready", "replay_ready", "completed"}
_UNSAFE_STATES = {"training", "snapshot"}
_UNSAFE_LOWER_STAGES = {
    "madpo_training",
    "reward_model_training",
    "online_rl_running",
    "snapshot_running",
}


class IterativeWorkflowError(RuntimeError):
    """The iterative workflow cannot proceed without violating its contract."""


class UnsafeResumeError(IterativeWorkflowError):
    """The manifest stopped inside a stage that cannot be resumed safely."""


class ReplayStore(Protocol):
    @property
    def available_iterations(self) -> tuple[int, ...]: ...

    def write_iteration(
        self,
        iteration: int,
        records: Sequence[PreferenceReplayRecord],
    ) -> Any: ...

    def load_iteration(self, iteration: int) -> tuple[PreferenceReplayRecord, ...]: ...

    def select(
        self,
        mode: str,
        *,
        current_iteration: int | None = None,
        sample_size: int | None = None,
        nearest_k: int | None = None,
        replay_lambda: float | None = None,
        seed: int | None = None,
        replacement: bool = True,
    ) -> list[PreferenceReplayRecord]: ...


class SnapshotStore(Protocol):
    @property
    def available_iterations(self) -> tuple[int, ...]: ...

    def commit_iteration(
        self,
        iteration: int,
        writer: Callable[[Path], Mapping[str, Any] | None],
        metadata: Mapping[str, Any] | None = None,
    ) -> Path: ...


ComparatorFactory = Callable[[int], Comparator]
PreferenceCollector = Callable[
    [int, Comparator, int, int | None, str],
    Sequence[PreferenceReplayRecord],
]
MADPOTrainer = Callable[[int, tuple[JointPreferencePair, ...]], Any]
RewardModelTrainer = Callable[[int, tuple[JointPreferencePair, ...]], Any]
OnlineRLTrainer = Callable[[int, tuple[JointPreferencePair, ...], Any], Any]
PolicySnapshotSaver = Callable[[int, Path], Any]


@dataclass(frozen=True)
class CoMLRLIterativeConfig:
    algorithm: Algorithm = "MADPOIter"
    num_iterations: int = 6
    num_train_epochs: int | None = None
    num_target_candidates: int = 20
    pairs_per_sample: int | None = 4
    pair_selection: str = "comparator_reward"
    replay_mode: str = "current"
    nearest_k: int | None = None
    replay_lambda: float | None = None
    replay_sample_size: int | None = None
    replay_seed: int | None = None
    replay_replacement: bool = True
    max_turns: int = 1
    joint_mode: str = "aligned"

    def __post_init__(self) -> None:
        if self.algorithm not in {"MADPOIter", "MARLHFIter"}:
            raise ValueError("algorithm must be 'MADPOIter' or 'MARLHFIter'")
        _positive_int(self.num_iterations, "num_iterations")
        train_epochs = self.num_train_epochs
        if train_epochs is None:
            train_epochs = 2 if self.algorithm == "MARLHFIter" else 1
            object.__setattr__(self, "num_train_epochs", train_epochs)
        _positive_int(train_epochs, "num_train_epochs")
        if _positive_int(self.num_target_candidates, "num_target_candidates") < 2:
            raise ValueError("num_target_candidates must be >= 2")
        if self.max_turns != 1:
            raise ValueError("CoMLRL iterative training supports max_turns=1 only")
        if self.joint_mode != "aligned":
            raise ValueError("CoMLRL iterative training supports joint_mode='aligned' only")
        selections = {"comparator_reward", "reward_gap", "random", "all"}
        if self.pair_selection not in selections:
            raise ValueError("pair_selection must be one of: comparator_reward, reward_gap, random, all")
        if self.pair_selection == "all":
            if self.pairs_per_sample is not None:
                raise ValueError("pairs_per_sample must be None when pair_selection='all'")
        elif self.pairs_per_sample is None:
            raise ValueError("pairs_per_sample is required unless pair_selection='all'")
        else:
            _positive_int(self.pairs_per_sample, "pairs_per_sample")

        replay_mode = canonical_replay_mode(self.replay_mode)
        object.__setattr__(self, "replay_mode", replay_mode)
        if self.replay_sample_size is not None:
            _positive_int(self.replay_sample_size, "replay_sample_size")
        if self.replay_seed is not None and (
            isinstance(self.replay_seed, bool) or not isinstance(self.replay_seed, int)
        ):
            raise TypeError("replay_seed must be an integer or None")
        if not isinstance(self.replay_replacement, bool):
            raise TypeError("replay_replacement must be a boolean")

        if replay_mode == "current":
            if self.nearest_k not in {None, 1}:
                raise ValueError("nearest_k must be None or 1 for current replay")
            if self.replay_lambda is not None or self.replay_sample_size is not None:
                raise ValueError("current replay does not accept replay_lambda or replay_sample_size")
        elif replay_mode == "nearest_k":
            if self.nearest_k is None:
                raise ValueError("nearest_k is required for nearest_k replay")
            _positive_int(self.nearest_k, "nearest_k")
            if self.replay_lambda is not None:
                raise ValueError("replay_lambda is only valid for lambda_decay replay")
        elif replay_mode == "all_history":
            if self.nearest_k is not None or self.replay_lambda is not None:
                raise ValueError("all_history replay does not accept nearest_k or replay_lambda")
        else:
            if self.nearest_k is not None:
                raise ValueError("nearest_k is only valid for nearest_k replay")
            valid_lambda = (
                self.replay_lambda is not None
                and not isinstance(self.replay_lambda, bool)
                and isinstance(self.replay_lambda, int | float)
                and 0.0 <= float(self.replay_lambda) <= 1.0
            )
            if not valid_lambda:
                raise ValueError("replay_lambda must be in [0, 1] for lambda_decay replay")


class CoMLRLIterativeController:
    """Coordinate iterative collection, replay, training, and snapshots."""

    def __init__(
        self,
        config: CoMLRLIterativeConfig,
        *,
        agent_ids: Sequence[str],
        workflow_dir: str | Path,
        comparator_factory: ComparatorFactory,
        collect_preferences: PreferenceCollector,
        save_policy_snapshot: PolicySnapshotSaver,
        replay_store: ReplayStore | None = None,
        snapshot_store: SnapshotStore | None = None,
        train_madpo: MADPOTrainer | None = None,
        train_reward_model: RewardModelTrainer | None = None,
        train_online_rl: OnlineRLTrainer | None = None,
        workflow_identity: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(config, CoMLRLIterativeConfig):
            raise TypeError("config must be CoMLRLIterativeConfig")
        self.config = config
        self.agent_ids = _agent_ids(agent_ids)
        self.workflow_dir = Path(workflow_dir).expanduser().resolve()
        self.workflow_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.workflow_dir / "workflow_manifest.json"
        self.replay_store = replay_store or PreferenceReplayStore(self.workflow_dir / "replay")
        self.snapshot_store = snapshot_store or PolicySnapshotStore(self.workflow_dir / "policy_snapshots")
        self.comparator_factory = _callable(comparator_factory, "comparator_factory")
        self.collect_preferences = _callable(collect_preferences, "collect_preferences")
        self.save_policy_snapshot = _callable(save_policy_snapshot, "save_policy_snapshot")
        self.train_madpo = train_madpo
        self.train_reward_model = train_reward_model
        self.train_online_rl = train_online_rl
        if workflow_identity is not None and not isinstance(workflow_identity, Mapping):
            raise TypeError("workflow_identity must be a mapping or None")
        self.workflow_identity = redact_secrets(dict(workflow_identity or {}))
        self._validate_trainers()

    def run(self, mode: RunMode = "fresh") -> None:
        if mode not in {"fresh", "resume"}:
            raise ValueError("mode must be 'fresh' or 'resume'")
        manifest = self._open_workflow(mode)
        if manifest["status"] == "completed":
            return

        while manifest["next_iteration"] < self.config.num_iterations:
            iteration = manifest["next_iteration"]
            manifest, _ = self._ensure_replay_ready(manifest, iteration)
            selected = tuple(
                self.replay_store.select(
                    self.config.replay_mode,
                    current_iteration=iteration,
                    sample_size=self.config.replay_sample_size,
                    nearest_k=self.config.nearest_k,
                    replay_lambda=self.config.replay_lambda,
                    seed=self.config.replay_seed,
                    replacement=self.config.replay_replacement,
                )
            )
            pairs = (
                records_to_joint_preference_pairs(
                    selected,
                    agent_ids=self.agent_ids,
                    expected_max_iteration=iteration,
                )
                if selected
                else ()
            )

            if pairs:
                manifest = self._write_state("training", iteration=iteration, next_iteration=iteration)
                if self.config.algorithm == "MADPOIter":
                    assert self.train_madpo is not None
                    self.train_madpo(iteration, pairs)
                else:
                    assert self.train_reward_model is not None
                    assert self.train_online_rl is not None
                    reward_artifact = self.train_reward_model(iteration, pairs)
                    self.train_online_rl(iteration, pairs, reward_artifact)

            manifest = self._write_state("snapshot", iteration=iteration, next_iteration=iteration)
            metadata = {
                "algorithm": self.config.algorithm,
                "policy_iteration": iteration,
            }
            commit = getattr(self.snapshot_store, "commit_iteration", None)
            if callable(commit):
                commit(
                    iteration,
                    lambda target_path, iteration=iteration: _snapshot_writer_result(
                        self.save_policy_snapshot(iteration, target_path)
                    ),
                    metadata,
                )
            else:
                target_path = self.snapshot_store.register_iteration(iteration, metadata)
                self.save_policy_snapshot(iteration, target_path)

            next_iteration = iteration + 1
            status = "completed" if next_iteration == self.config.num_iterations else "ready"
            manifest = self._write_state(status, iteration=None, next_iteration=next_iteration)

    def _open_workflow(self, mode: RunMode) -> dict[str, Any]:
        if mode == "fresh":
            has_state = (
                self.manifest_path.exists()
                or bool(self.replay_store.available_iterations)
                or bool(self.snapshot_store.available_iterations)
            )
            if has_state:
                raise FileExistsError(
                    "fresh CoMLRL iterative workflow requires no existing workflow, replay, or snapshot state"
                )
            return self._write_state("ready", iteration=None, next_iteration=0)

        if not self.manifest_path.exists():
            raise FileNotFoundError("resume requires an existing CoMLRL iterative workflow manifest")
        manifest = self._read_manifest()
        self._validate_manifest(manifest)
        if manifest["status"] in _UNSAFE_STATES:
            raise UnsafeResumeError(
                f"cannot safely resume from {manifest['status']!r}: optimizer/RNG "
                "state is not resumable; restart the interrupted policy iteration "
                "from a clean workflow"
            )
        if manifest["status"] not in _SAFE_STATES:
            raise IterativeWorkflowError(f"unknown workflow state {manifest['status']!r}")
        self._validate_persisted_boundaries(manifest)
        return manifest

    def _ensure_replay_ready(
        self,
        manifest: dict[str, Any],
        iteration: int,
    ) -> tuple[dict[str, Any], tuple[PreferenceReplayRecord, ...]]:
        replay_iteration = iteration
        available = set(self.replay_store.available_iterations)
        if manifest["status"] == "replay_ready":
            if manifest["current_iteration"] != iteration or manifest["replay_iteration"] != replay_iteration:
                raise IterativeWorkflowError("replay_ready manifest does not match next_iteration")
            records = self.replay_store.load_iteration(replay_iteration)
            _validate_replay_records(
                records,
                expected_iteration=replay_iteration,
                agent_count=len(self.agent_ids),
            )
            return manifest, records

        if manifest["status"] != "ready":
            raise IterativeWorkflowError(f"cannot prepare replay while workflow state is {manifest['status']!r}")
        if replay_iteration in available:
            records = self.replay_store.load_iteration(replay_iteration)
            _validate_replay_records(
                records,
                expected_iteration=replay_iteration,
                agent_count=len(self.agent_ids),
            )
            manifest = self._write_state("replay_ready", iteration=iteration, next_iteration=iteration)
            return manifest, records

        comparator = self.comparator_factory(iteration)
        if not callable(getattr(comparator, "generate", None)):
            raise TypeError("comparator_factory must return a Comparator implementing generate()")
        collected = self.collect_preferences(
            iteration,
            comparator,
            self.config.num_target_candidates,
            self.config.pairs_per_sample,
            self.config.pair_selection,
        )
        try:
            records = tuple(collected)
        except TypeError as exc:
            raise TypeError("collect_preferences must return a sequence of PreferenceReplayRecord values") from exc
        _validate_replay_records(
            records,
            expected_iteration=replay_iteration,
            agent_count=len(self.agent_ids),
        )
        self.replay_store.write_iteration(replay_iteration, records)
        manifest = self._write_state("replay_ready", iteration=iteration, next_iteration=iteration)
        return manifest, records

    def _validate_trainers(self) -> None:
        if self.config.algorithm == "MADPOIter":
            self.train_madpo = _callable(self.train_madpo, "train_madpo")
            if self.train_reward_model is not None or self.train_online_rl is not None:
                raise ValueError("MADPOIter does not accept MARLHF training callbacks")
            return
        self.train_reward_model = _callable(self.train_reward_model, "train_reward_model")
        self.train_online_rl = _callable(self.train_online_rl, "train_online_rl")
        if self.train_madpo is not None:
            raise ValueError("MARLHFIter does not accept train_madpo")

    def _validate_persisted_boundaries(self, manifest: Mapping[str, Any]) -> None:
        next_iteration = manifest["next_iteration"]
        expected_replay = set(range(next_iteration))
        if manifest["status"] == "replay_ready":
            expected_replay.add(next_iteration)
        actual_replay = set(self.replay_store.available_iterations)
        if manifest["status"] == "ready" and next_iteration in actual_replay:
            expected_replay.add(next_iteration)
        if actual_replay != expected_replay:
            raise IterativeWorkflowError(
                "replay shards do not match workflow boundary: expected "
                f"{sorted(expected_replay)}, got {sorted(actual_replay)}"
            )
        expected_snapshots = set(range(next_iteration))
        actual_snapshots = set(self.snapshot_store.available_iterations)
        if actual_snapshots != expected_snapshots:
            raise IterativeWorkflowError(
                "policy snapshots do not match workflow boundary: expected "
                f"{sorted(expected_snapshots)}, got {sorted(actual_snapshots)}"
            )

    def _read_manifest(self) -> dict[str, Any]:
        try:
            with self.manifest_path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise IterativeWorkflowError(f"cannot read workflow manifest: {exc}") from exc
        if not isinstance(payload, dict):
            raise IterativeWorkflowError("workflow manifest must contain a JSON object")
        return payload

    def _validate_manifest(self, manifest: Mapping[str, Any]) -> None:
        if manifest.get("schema_version") != _WORKFLOW_SCHEMA_VERSION:
            raise IterativeWorkflowError("unsupported workflow manifest schema_version")
        if manifest.get("resume_extension") != _RESUME_EXTENSION:
            raise IterativeWorkflowError("workflow manifest is not a recognized TrajWeave resume extension")
        expected = {
            "algorithm": self.config.algorithm,
            "num_iterations": self.config.num_iterations,
            "num_train_epochs": self.config.num_train_epochs,
            "agent_ids": list(self.agent_ids),
            "workflow_identity": self.workflow_identity,
        }
        drift = {key: (manifest.get(key), value) for key, value in expected.items() if manifest.get(key) != value}
        if drift:
            raise IterativeWorkflowError(f"workflow configuration drift detected: {drift}")
        status = manifest.get("status")
        next_iteration = manifest.get("next_iteration")
        current_iteration = manifest.get("current_iteration")
        replay_iteration = manifest.get("replay_iteration")
        if status not in _SAFE_STATES | _UNSAFE_STATES:
            raise IterativeWorkflowError(f"invalid workflow status {status!r}")
        if isinstance(next_iteration, bool) or not isinstance(next_iteration, int):
            raise IterativeWorkflowError("manifest.next_iteration must be an integer")
        if not 0 <= next_iteration <= self.config.num_iterations:
            raise IterativeWorkflowError("manifest.next_iteration is outside the configured iteration range")
        if status in {"ready", "completed"}:
            if current_iteration is not None or replay_iteration is not None:
                raise IterativeWorkflowError(f"{status} manifest must not contain an active iteration")
            if status == "completed" and next_iteration != self.config.num_iterations:
                raise IterativeWorkflowError("completed manifest has an incomplete next_iteration")
        elif current_iteration != next_iteration or replay_iteration != current_iteration:
            raise IterativeWorkflowError(f"{status} manifest has inconsistent active iteration fields")

    def _write_state(
        self,
        status: str,
        *,
        iteration: int | None,
        next_iteration: int,
    ) -> dict[str, Any]:
        payload = {
            "schema_version": _WORKFLOW_SCHEMA_VERSION,
            "resume_extension": _RESUME_EXTENSION,
            "algorithm": self.config.algorithm,
            "num_iterations": self.config.num_iterations,
            "num_train_epochs": self.config.num_train_epochs,
            "agent_ids": list(self.agent_ids),
            "workflow_identity": self.workflow_identity,
            "status": status,
            "next_iteration": next_iteration,
            "current_iteration": iteration,
            "replay_iteration": iteration if iteration is not None else None,
        }
        self._validate_manifest(payload)
        _atomic_write_json(self.manifest_path, payload)
        return payload


def records_to_joint_preference_pairs(
    records: Sequence[PreferenceReplayRecord],
    *,
    agent_ids: Sequence[str],
    expected_max_iteration: int | None = None,
) -> tuple[JointPreferencePair, ...]:
    """Strictly map replay tuple positions to the caller-provided agent order."""

    ordered_agents = _agent_ids(agent_ids)
    stable_records = tuple(records)
    if not stable_records:
        raise ValueError("at least one replay record is required")
    pairs: list[JointPreferencePair] = []
    occurrence_by_pair_id: dict[str, int] = {}
    for record in stable_records:
        if not isinstance(record, PreferenceReplayRecord):
            raise TypeError("records must contain PreferenceReplayRecord values")
        if len(record.prompts) != len(ordered_agents):
            raise ValueError(
                f"replay record {record.pair_id!r} has {len(record.prompts)} agents; expected {len(ordered_agents)}"
            )
        if expected_max_iteration is not None and record.iteration > expected_max_iteration:
            raise ValueError(
                f"replay record {record.pair_id!r} is from future shard "
                f"{record.iteration}; current shard is {expected_max_iteration}"
            )
        occurrence = occurrence_by_pair_id.get(record.pair_id, 0)
        occurrence_by_pair_id[record.pair_id] = occurrence + 1
        training_pair_id = record.pair_id if occurrence == 0 else f"{record.pair_id}:draw:{occurrence}"
        _validate_preference_order(record)

        prompts = dict(zip(ordered_agents, record.prompts, strict=True))
        chosen = dict(zip(ordered_agents, record.chosen, strict=True))
        rejected = dict(zip(ordered_agents, record.rejected, strict=True))
        pairs.append(
            JointPreferencePair(
                preference_pair_id=training_pair_id,
                episode_id=f"replay-iteration-{record.iteration:04d}",
                tree_node_id=f"replay-{training_pair_id}",
                chosen_joint_action_id=f"{training_pair_id}:chosen",
                rejected_joint_action_id=f"{training_pair_id}:rejected",
                prompts_by_agent=prompts,
                chosen_by_agent=chosen,
                rejected_by_agent=rejected,
                chosen_reward=record.processed_chosen_reward,
                rejected_reward=record.processed_rejected_reward,
                candidate_mean=record.candidate_mean,
                metadata={
                    "source": "preference_replay",
                    "source_preference_pair_id": record.pair_id,
                    "replay_occurrence": occurrence,
                    "replay_iteration": record.iteration,
                    "policy_provenance": dict(record.policy_provenance),
                    "comparator_provenance": dict(record.comparator_provenance),
                },
            )
        )
    return tuple(pairs)


def _validate_replay_records(
    records: Sequence[PreferenceReplayRecord],
    *,
    expected_iteration: int,
    agent_count: int,
) -> None:
    pair_ids: set[str] = set()
    for record in records:
        if not isinstance(record, PreferenceReplayRecord):
            raise TypeError("preference collection must contain PreferenceReplayRecord values")
        if record.iteration != expected_iteration:
            raise ValueError(
                f"replay records must use zero-based shard iteration {expected_iteration}; got {record.iteration}"
            )
        if len(record.prompts) != agent_count:
            raise ValueError(f"replay record {record.pair_id!r} has the wrong agent count")
        if record.pair_id in pair_ids:
            raise ValueError(f"duplicate replay pair_id {record.pair_id!r}")
        pair_ids.add(record.pair_id)
        _validate_preference_order(record)


def _validate_preference_order(record: PreferenceReplayRecord) -> None:
    if record.processed_chosen_reward == record.processed_rejected_reward:
        raise ValueError(f"replay record {record.pair_id!r} is tied")
    if record.processed_chosen_reward < record.processed_rejected_reward:
        raise ValueError(f"replay record {record.pair_id!r} has chosen reward below rejected reward")


def _agent_ids(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("agent_ids must be a sequence of strings")
    normalized = tuple(values)
    if not normalized or any(not isinstance(value, str) or not value for value in normalized):
        raise ValueError("agent_ids must contain non-empty strings")
    if len(normalized) != len(set(normalized)):
        raise ValueError("agent_ids must be unique")
    return normalized


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _snapshot_writer_result(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _callable(value: Any, name: str) -> Any:
    if not callable(value):
        raise TypeError(f"{name} must be callable")
    return value


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                redact_secrets(dict(payload)),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


@dataclass(frozen=True)
class PromptBatch:
    prompts: tuple[str, ...]
    context: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.prompts or any(not isinstance(value, str) or not value for value in self.prompts):
            raise ValueError("PromptBatch.prompts must contain non-empty strings")
        if self.context is not None and not isinstance(self.context, Mapping):
            raise TypeError("PromptBatch.context must be a mapping or None")


@dataclass(frozen=True)
class RewardScores:
    raw: tuple[float, ...]
    processed: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.raw) != len(self.processed):
            raise ValueError("RewardScores raw and processed values must align")
        for value in (*self.raw, *self.processed):
            if isinstance(value, bool) or not isinstance(value, int | float) or not _is_finite(float(value)):
                raise FloatingPointError("RewardScores values must be finite")


class PolicyLifecycle(Protocol):
    def freeze_current(self, *, policy: Any, iteration: int) -> FrozenPolicySnapshot: ...

    def save(
        self,
        *,
        policy: Any,
        path: Path,
        iteration: int | None,
        purpose: str,
    ) -> Mapping[str, Any] | None: ...

    def load(self, *, path: Path, snapshot_id: str) -> FrozenPolicySnapshot: ...

    def release(self, snapshot: FrozenPolicySnapshot) -> None: ...


class RewardScorer(Protocol):
    def __call__(
        self,
        *,
        prompts: Sequence[str],
        candidates_by_agent: Sequence[Sequence[str]],
        context: Mapping[str, Any],
        iteration: int,
    ) -> RewardScores: ...


@dataclass(frozen=True)
class IterativeConfig:
    num_iterations: int = 6
    num_agents: int = 2
    num_turns: int = 1
    joint_mode: str = "aligned"
    num_train_epochs: int | None = None
    preference_num_candidates: int = 20
    preference_pairs_per_sample: int | None = 4
    pair_selection: str = "comparator_reward"
    preference_replay_mode: str = "current"
    preference_replay_k: int | None = None
    preference_replay_lambda: float | None = None
    preference_replay_sample_size: int | None = None
    preference_replay_seed: int = 0
    preference_replay_dir: str | None = None
    policy_checkpoint_dir: str | None = None
    comparator_policy: str = "current"
    comparator_generation_mode: str = "decentralized"
    comparator_centralized_agent_index: int = 0
    comparator_num_candidates: int | None = None
    comparator_history_k: int | None = None
    comparator_model_name: str | None = None
    comparator_agents: Sequence[str] | None = None
    comparator_devices: str | Sequence[str] | None = None
    comparator_model_path: str | None = None
    comparator_api_url: str | None = None
    comparator_api_format: str = "generic"
    comparator_api_model: str | None = None
    comparator_api_timeout: float = 120.0
    comparator_api_headers: Mapping[str, str] | None = None
    comparator_api_key: str | None = None
    comparator_api_key_env: str | None = None
    comparator_api_key_header: str = "Authorization"
    comparator_api_key_prefix: str = "Bearer"
    comparator_api_response_field: str = "completions"
    comparator_api_extra_body: Mapping[str, Any] | None = None
    comparator_api_max_n_per_request: int | None = None
    preference_scoring_reward: str = "task"
    reward_num_train_epochs: int = 2
    rl_algorithm: str = "magrpo"

    def __post_init__(self) -> None:
        _positive_int(self.num_iterations, "num_iterations")
        _positive_int(self.num_agents, "num_agents")
        if self.num_turns != 1:
            raise ValueError("iterative MADPO/MARLHF currently supports num_turns=1 only")
        if str(self.joint_mode).strip().lower() not in {"align", "aligned"}:
            raise ValueError("iterative MADPO/MARLHF requires aligned joint_mode")
        object.__setattr__(self, "joint_mode", "aligned")
        if self.num_train_epochs is not None:
            _positive_int(self.num_train_epochs, "num_train_epochs")
        if _positive_int(self.preference_num_candidates, "preference_num_candidates") < 2:
            raise ValueError("preference_num_candidates must be >= 2")
        selection = str(self.pair_selection).strip().lower()
        if selection not in {"comparator_reward", "reward_gap", "random", "all"}:
            raise ValueError("pair_selection must be comparator_reward, reward_gap, random, or all")
        object.__setattr__(self, "pair_selection", selection)
        if selection == "all":
            if self.preference_pairs_per_sample is not None:
                raise ValueError("preference_pairs_per_sample must be None for pair_selection='all'")
        elif self.preference_pairs_per_sample is None:
            raise ValueError("preference_pairs_per_sample is required outside pair_selection='all'")
        else:
            _positive_int(self.preference_pairs_per_sample, "preference_pairs_per_sample")
        replay_mode = canonical_replay_mode(self.preference_replay_mode)
        object.__setattr__(self, "preference_replay_mode", replay_mode)
        if replay_mode == "current":
            if self.preference_replay_k not in {None, 1}:
                raise ValueError("preference_replay_k must be None or 1 for current replay")
            if self.preference_replay_lambda is not None or self.preference_replay_sample_size is not None:
                raise ValueError("current replay does not accept replay_lambda or replay_sample_size")
        elif replay_mode == "nearest_k":
            if self.preference_replay_k is None:
                raise ValueError("preference_replay_k is required for nearest_k replay")
            _positive_int(self.preference_replay_k, "preference_replay_k")
            if self.preference_replay_lambda is not None:
                raise ValueError("preference_replay_lambda is only valid for lambda_decay replay")
        elif replay_mode == "all_history":
            if self.preference_replay_k is not None or self.preference_replay_lambda is not None:
                raise ValueError("all_history replay does not accept replay_k or replay_lambda")
        else:
            if self.preference_replay_k is not None:
                raise ValueError("preference_replay_k is only valid for nearest_k replay")
            if self.preference_replay_lambda is None or not 0 <= float(self.preference_replay_lambda) <= 1:
                raise ValueError("preference_replay_lambda must be in [0, 1]")
        if self.preference_replay_sample_size is not None:
            _positive_int(self.preference_replay_sample_size, "preference_replay_sample_size")
        if isinstance(self.preference_replay_seed, bool) or not isinstance(self.preference_replay_seed, int):
            raise TypeError("preference_replay_seed must be an integer")
        policy = canonical_comparator_policy(self.comparator_policy)
        mode = canonical_generation_mode(self.comparator_generation_mode)
        object.__setattr__(self, "comparator_policy", policy)
        object.__setattr__(self, "comparator_generation_mode", mode)
        if mode == "centralized":
            if self.num_agents != 2:
                raise ValueError("centralized comparator generation requires num_agents=2")
            if not 0 <= self.comparator_centralized_agent_index < self.num_agents:
                raise ValueError("comparator_centralized_agent_index must index an agent")
        if self.comparator_num_candidates is not None:
            _positive_int(self.comparator_num_candidates, "comparator_num_candidates")
        if self.comparator_api_max_n_per_request is not None:
            _positive_int(self.comparator_api_max_n_per_request, "comparator_api_max_n_per_request")
        if self.comparator_api_timeout <= 0:
            raise ValueError("comparator_api_timeout must be positive")
        model_configured = bool(self.comparator_model_path or self.comparator_model_name or self.comparator_agents)
        if policy == "model":
            if not model_configured:
                raise ValueError("model comparator requires comparator_model_name or comparator_agents")
            if self.comparator_history_k is not None:
                raise ValueError("comparator_history_k is only valid for history comparator")
        elif model_configured:
            raise ValueError("comparator model fields are only valid for model comparator")
        if policy == "history":
            object.__setattr__(self, "comparator_history_k", self.comparator_history_k or 1)
            _positive_int(self.comparator_history_k, "comparator_history_k")
        elif self.comparator_history_k is not None:
            raise ValueError("comparator_history_k is only valid for history comparator")
        if policy == "api" and not self.comparator_api_url:
            raise ValueError("comparator_api_url is required for API comparator")
        scoring = str(self.preference_scoring_reward).strip().lower()
        if scoring not in {"task", "reward_model"}:
            raise ValueError("preference_scoring_reward must be task or reward_model")
        object.__setattr__(self, "preference_scoring_reward", scoring)
        _positive_int(self.reward_num_train_epochs, "reward_num_train_epochs")
        resolve_marlhf_rl_dispatch(self.rl_algorithm)


@dataclass(frozen=True)
class IterationState:
    iteration: int
    completed_stages: tuple[str, ...] = ()
    replay_shard: str | None = None
    comparator_snapshot: str | None = None
    reward_model_checkpoint: str | None = None
    active_reward_model_version: str | None = None
    policy_snapshot: str | None = None
    rng_seed: int = 0

    @property
    def completed(self) -> bool:
        return "completed" in self.completed_stages

    def with_stage(self, stage: str, **changes: Any) -> IterationState:
        stages = self.completed_stages if stage in self.completed_stages else (*self.completed_stages, stage)
        return replace(self, completed_stages=stages, **changes)

    def transition_stage(self, running: str, completed: str, **changes: Any) -> IterationState:
        stages = tuple(stage for stage in self.completed_stages if stage != running)
        if completed not in stages:
            stages = (*stages, completed)
        return replace(self, completed_stages=stages, **changes)

    def to_dict(self) -> dict[str, Any]:
        return redact_secrets(
            {
                "iteration": self.iteration,
                "completed_stages": list(self.completed_stages),
                "replay_shard": self.replay_shard,
                "comparator_snapshot": self.comparator_snapshot,
                "reward_model_checkpoint": self.reward_model_checkpoint,
                "active_reward_model_version": self.active_reward_model_version,
                "policy_snapshot": self.policy_snapshot,
                "rng_seed": self.rng_seed,
            }
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> IterationState:
        return cls(
            iteration=int(payload["iteration"]),
            completed_stages=tuple(str(value) for value in payload.get("completed_stages", [])),
            replay_shard=payload.get("replay_shard"),
            comparator_snapshot=payload.get("comparator_snapshot"),
            reward_model_checkpoint=payload.get("reward_model_checkpoint"),
            active_reward_model_version=payload.get("active_reward_model_version"),
            policy_snapshot=payload.get("policy_snapshot"),
            rng_seed=int(payload.get("rng_seed", 0)),
        )


class IterationStateStore:
    def __init__(self, root: str | Path, *, base_seed: int = 0) -> None:
        self.root = Path(root).expanduser().resolve()
        self.base_seed = base_seed
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self._states: dict[int, IterationState] = {}
        for path in sorted(self.root.glob("iteration_*.json")):
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            state = IterationState.from_dict(payload)
            self._states[state.iteration] = state
        self._write_manifest()

    @property
    def iteration_cursor(self) -> int:
        cursor = 0
        while self._states.get(cursor) is not None and self._states[cursor].completed:
            cursor += 1
        return cursor

    def get_or_create(self, iteration: int) -> IterationState:
        if iteration not in self._states:
            self._states[iteration] = IterationState(iteration=iteration, rng_seed=self.base_seed + iteration)
            self.save(self._states[iteration])
        return self._states[iteration]

    def save(self, state: IterationState) -> None:
        self._states[state.iteration] = state
        _atomic_write_json(self.root / f"iteration_{state.iteration:04d}.json", state.to_dict())
        self._write_manifest()

    def states(self) -> tuple[IterationState, ...]:
        return tuple(self._states[index] for index in sorted(self._states))

    def _write_manifest(self) -> None:
        completed = [index for index in sorted(self._states) if self._states[index].completed]
        _atomic_write_json(
            self.manifest_path,
            {
                "schema_version": 1,
                "iteration_cursor": self.iteration_cursor,
                "completed_iterations": completed,
            },
        )


class _IterativeWorkflowBase:
    def __init__(
        self,
        *,
        config: IterativeConfig,
        artifact_dir: str | Path,
        current_policy: Any,
        prompt_source: Any,
        task_reward_scorer: Any,
        policy_lifecycle: Any,
        comparator_model: FrozenPolicySnapshot | None = None,
        centralized_prompt_adapter: Callable[..., str] | None = None,
        centralized_response_parser: Callable[[str], Sequence[str]] | None = None,
        remote_comparator_opener: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config
        self.artifact_dir = Path(artifact_dir).expanduser().resolve()
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.current_policy = current_policy
        self.prompt_source = _callable(prompt_source, "prompt_source")
        self.task_reward_scorer = _callable(task_reward_scorer, "task_reward_scorer")
        self.policy_lifecycle = policy_lifecycle
        self.comparator_model = comparator_model
        self.centralized_prompt_adapter = centralized_prompt_adapter
        self.centralized_response_parser = centralized_response_parser
        self.remote_comparator_opener = remote_comparator_opener
        self.replay_store = PreferenceReplayStore(self.config.preference_replay_dir or self.artifact_dir / "replay")
        self.snapshot_store = PolicySnapshotStore(
            self.config.policy_checkpoint_dir or self.artifact_dir / "policy_snapshots"
        )
        self.state_store = IterationStateStore(
            self.artifact_dir / "iteration_state", base_seed=self.config.preference_replay_seed
        )
        self.num_agents = len(current_policy)
        if self.num_agents != self.config.num_agents:
            raise ValueError("current_policy agent count must match config.num_agents")

    def _ensure_initial_snapshot(self) -> None:
        if self.snapshot_store.initial_path.is_dir():
            return
        self.snapshot_store.commit_initial(
            lambda path: self.policy_lifecycle.save(
                policy=self.current_policy,
                path=path,
                iteration=None,
                purpose="initial",
            )
        )

    @staticmethod
    def _raise_if_unsafe(state: IterationState) -> None:
        running = sorted(_UNSAFE_LOWER_STAGES.intersection(state.completed_stages))
        if running:
            raise UnsafeResumeError(
                f"cannot safely resume iteration {state.iteration} from running stage(s) {running}; "
                "the callback may already have produced side effects"
            )

    def _prompt_batches(self, iteration: int) -> tuple[PromptBatch, ...]:
        values = tuple(self.prompt_source(iteration=iteration))
        if not values or any(not isinstance(value, PromptBatch) for value in values):
            raise ValueError("prompt_source must return one or more PromptBatch values")
        if any(len(value.prompts) != self.num_agents for value in values):
            raise ValueError("PromptBatch agent count must match current_policy")
        return values

    @contextmanager
    def _comparator_context(self, iteration: int, state: IterationState):
        policy = self.config.comparator_policy
        owned: list[FrozenPolicySnapshot] = []
        try:
            if policy == "api":
                comparator: Comparator = RemoteComparator(
                    url=str(self.config.comparator_api_url),
                    num_agents=self.num_agents,
                    api_format=self.config.comparator_api_format,
                    generation_mode=self.config.comparator_generation_mode,
                    model=self.config.comparator_api_model,
                    timeout=self.config.comparator_api_timeout,
                    headers=self.config.comparator_api_headers,
                    api_key=self.config.comparator_api_key,
                    api_key_env=self.config.comparator_api_key_env,
                    api_key_header=self.config.comparator_api_key_header,
                    api_key_prefix=self.config.comparator_api_key_prefix,
                    response_field=self.config.comparator_api_response_field,
                    extra_body=self.config.comparator_api_extra_body,
                    max_candidates_per_request=self.config.comparator_api_max_n_per_request,
                    centralized_prompt_adapter=self.centralized_prompt_adapter,
                    centralized_response_parser=self.centralized_response_parser,
                    opener=self.remote_comparator_opener,
                )
            elif policy == "current_copy":
                if state.comparator_snapshot and "comparator_prepared" in state.completed_stages:
                    snapshot = self.policy_lifecycle.load(
                        path=Path(state.comparator_snapshot), snapshot_id=f"current-copy-{iteration:04d}"
                    )
                else:
                    snapshot = self.policy_lifecycle.freeze_current(policy=self.current_policy, iteration=iteration)
                if not isinstance(snapshot, FrozenPolicySnapshot):
                    raise TypeError("current_copy requires a real FrozenPolicySnapshot")
                owned.append(snapshot)
                comparator = PolicyComparator(
                    policy=policy,
                    current_policy=self.current_policy,
                    current_copy=snapshot,
                    num_agents=self.num_agents,
                    generation_mode=self.config.comparator_generation_mode,
                    centralized_prompt_adapter=self.centralized_prompt_adapter,
                    centralized_response_parser=self.centralized_response_parser,
                    centralized_agent_index=self.config.comparator_centralized_agent_index,
                )
            elif policy == "history":
                path = self.snapshot_store.resolve_history(iteration, history_k=self.config.comparator_history_k)
                snapshot = self.policy_lifecycle.load(path=path, snapshot_id=f"history-{path.name}")
                if not isinstance(snapshot, FrozenPolicySnapshot):
                    raise TypeError("history policy loader must return FrozenPolicySnapshot")
                owned.append(snapshot)
                comparator = PolicyComparator(
                    policy=policy,
                    current_policy=self.current_policy,
                    history_resolver=lambda _iteration: snapshot,
                    num_agents=self.num_agents,
                    generation_mode=self.config.comparator_generation_mode,
                    centralized_prompt_adapter=self.centralized_prompt_adapter,
                    centralized_response_parser=self.centralized_response_parser,
                    centralized_agent_index=self.config.comparator_centralized_agent_index,
                )
            elif policy == "model":
                snapshot = self.comparator_model
                if snapshot is None:
                    source = self.config.comparator_model_path or self.config.comparator_model_name
                    if source is None:
                        raise ValueError("comparator_agents requires an injected comparator_model snapshot")
                    path = Path(source).expanduser().resolve()
                    snapshot = self.policy_lifecycle.load(path=path, snapshot_id="model")
                    owned.append(snapshot)
                if not isinstance(snapshot, FrozenPolicySnapshot):
                    raise TypeError("model comparator requires FrozenPolicySnapshot")
                comparator = PolicyComparator(
                    policy=policy,
                    current_policy=self.current_policy,
                    model=snapshot,
                    num_agents=self.num_agents,
                    generation_mode=self.config.comparator_generation_mode,
                    centralized_prompt_adapter=self.centralized_prompt_adapter,
                    centralized_response_parser=self.centralized_response_parser,
                    centralized_agent_index=self.config.comparator_centralized_agent_index,
                )
            else:
                comparator = PolicyComparator(
                    policy="current",
                    current_policy=self.current_policy,
                    num_agents=self.num_agents,
                    generation_mode=self.config.comparator_generation_mode,
                    centralized_prompt_adapter=self.centralized_prompt_adapter,
                    centralized_response_parser=self.centralized_response_parser,
                    centralized_agent_index=self.config.comparator_centralized_agent_index,
                )
            yield comparator, tuple(owned)
        finally:
            close = locals().get("comparator")
            close = getattr(close, "close", None)
            if callable(close):
                close()
            release = getattr(self.policy_lifecycle, "release", None)
            if callable(release):
                for snapshot in reversed(owned):
                    release(snapshot)

    def _collect_records(
        self,
        *,
        iteration: int,
        replay_iteration: int,
        state: IterationState,
        scorer: Any,
    ) -> tuple[list[PreferenceReplayRecord], IterationState]:
        current_generator = PolicyComparator(
            policy="current",
            current_policy=self.current_policy,
            num_agents=self.num_agents,
        )
        records: list[PreferenceReplayRecord] = []
        with self._comparator_context(iteration, state) as (comparator, owned_snapshots):
            if "comparator_prepared" not in state.completed_stages:
                comparator_snapshot = owned_snapshots[0].path if owned_snapshots else None
                state = state.with_stage("comparator_prepared", comparator_snapshot=comparator_snapshot)
                self.state_store.save(state)
            for batch_index, batch in enumerate(self._prompt_batches(iteration)):
                current = current_generator.generate(
                    batch.prompts,
                    num_candidates=self.config.preference_num_candidates,
                    iteration=iteration,
                    context=batch.context,
                )
                comparator_result = comparator.generate(
                    batch.prompts,
                    num_candidates=self.config.comparator_num_candidates or self.config.preference_num_candidates,
                    iteration=iteration,
                    context=batch.context,
                )
                current_scores = _score_candidates(scorer, current, batch.context, iteration)
                comparator_scores = _score_candidates(scorer, comparator_result, batch.context, iteration)
                records.extend(
                    _records_from_comparison(
                        policy_iteration=iteration,
                        replay_iteration=replay_iteration,
                        batch_index=batch_index,
                        current=current,
                        comparator=comparator_result,
                        current_scores=current_scores,
                        comparator_scores=comparator_scores,
                        pair_selection=self.config.pair_selection,
                        pairs_per_sample=self.config.preference_pairs_per_sample,
                        seed=state.rng_seed + batch_index,
                    )
                )
        return records, state

    def _write_or_load_replay(
        self,
        *,
        iteration: int,
        state: IterationState,
        scorer: Any,
    ) -> tuple[tuple[PreferenceReplayRecord, ...], IterationState]:
        replay_iteration = iteration
        available = set(self.replay_store.available_iterations)
        if "replay_written" in state.completed_stages:
            if replay_iteration not in available:
                raise IterativeWorkflowError(
                    f"iteration {iteration} state declares replay_written but shard is missing"
                )
            return self.replay_store.load_iteration(replay_iteration), state
        if replay_iteration in available:
            records = self.replay_store.load_iteration(replay_iteration)
            _validate_replay_records(
                records,
                expected_iteration=replay_iteration,
                agent_count=self.num_agents,
            )
            shard = self.replay_store.shard_path(replay_iteration)
            state = state.with_stage("replay_written", replay_shard=str(shard))
            self.state_store.save(state)
            return records, state
        records, state = self._collect_records(
            iteration=iteration,
            replay_iteration=replay_iteration,
            state=state,
            scorer=scorer,
        )
        shard = self.replay_store.write_iteration(replay_iteration, records)
        state = state.with_stage("replay_written", replay_shard=str(shard.path))
        self.state_store.save(state)
        return tuple(records), state

    def _select_replay(
        self,
        iteration: int,
        current_records: Sequence[PreferenceReplayRecord],
        state: IterationState,
    ) -> tuple[list[PreferenceReplayRecord], IterationState]:
        sample_size = self.config.preference_replay_sample_size
        if sample_size is None and self.config.preference_replay_mode != "current":
            sample_size = len(current_records) or None
        selected = self.replay_store.select(
            self.config.preference_replay_mode,
            current_iteration=iteration,
            sample_size=sample_size,
            nearest_k=self.config.preference_replay_k,
            replay_lambda=self.config.preference_replay_lambda,
            seed=state.rng_seed,
        )
        if "replay_selected" not in state.completed_stages:
            state = state.with_stage("replay_selected")
            self.state_store.save(state)
        return selected, state

    def _snapshot_iteration(self, iteration: int, state: IterationState) -> IterationState:
        if "policy_snapshotted" in state.completed_stages:
            return state
        state = state.with_stage("snapshot_running")
        self.state_store.save(state)
        path = self.snapshot_store.commit_iteration(
            iteration,
            lambda temporary: self.policy_lifecycle.save(
                policy=self.current_policy,
                path=temporary,
                iteration=iteration,
                purpose="iteration",
            ),
        )
        state = state.transition_stage("snapshot_running", "policy_snapshotted", policy_snapshot=str(path))
        self.state_store.save(state)
        return state

    def _complete(self, state: IterationState) -> IterationState:
        if not state.completed:
            state = state.with_stage("completed")
            self.state_store.save(state)
        return state


class MADPOIterWorkflow(_IterativeWorkflowBase):
    def __init__(self, *, madpo_runner: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.madpo_runner = _callable(madpo_runner, "madpo_runner")

    def run(self) -> tuple[IterationState, ...]:
        self._ensure_initial_snapshot()
        for iteration in range(self.state_store.iteration_cursor, self.config.num_iterations):
            state = self.state_store.get_or_create(iteration)
            self._raise_if_unsafe(state)
            current_records, state = self._write_or_load_replay(
                iteration=iteration,
                state=state,
                scorer=self.task_reward_scorer,
            )
            selected, state = self._select_replay(iteration, current_records, state)
            if "madpo_trained" not in state.completed_stages and "training_skipped" not in state.completed_stages:
                if selected:
                    state = state.with_stage("madpo_training")
                    self.state_store.save(state)
                    self.madpo_runner(
                        iteration=iteration,
                        preference_records=tuple(selected),
                        num_train_epochs=self.config.num_train_epochs or 1,
                    )
                    state = state.transition_stage("madpo_training", "madpo_trained")
                else:
                    state = state.with_stage("training_skipped")
                self.state_store.save(state)
            state = self._snapshot_iteration(iteration, state)
            self._complete(state)
        return self.state_store.states()


class MARLHFIterWorkflow(_IterativeWorkflowBase):
    def __init__(
        self,
        *,
        reward_model_factory: Any,
        rl_runner: Any,
        reward_model_loader: Callable[..., Any] | None = None,
        active_reward_model_setter: Callable[[Any | None], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not callable(reward_model_factory) and not callable(getattr(reward_model_factory, "create", None)):
            raise TypeError("reward_model_factory must be callable or implement create()")
        self.reward_model_factory = reward_model_factory
        self.reward_model_loader = reward_model_loader
        self.rl_runner = rl_runner
        self.active_reward_model_setter = active_reward_model_setter
        self._active_reward_scorer: Any | None = None
        self._active_reward_model_version: str | None = None
        self._active_reward_model_checkpoint: str | None = None
        self._reward_model_instances: set[int] = set()

    def run(self) -> tuple[IterationState, ...]:
        self._ensure_initial_snapshot()
        cursor = self.state_store.iteration_cursor
        self._restore_previous_reward_model(cursor)
        for iteration in range(cursor, self.config.num_iterations):
            state = self.state_store.get_or_create(iteration)
            self._raise_if_unsafe(state)
            if state.completed:
                continue
            previous_scorer = self._active_reward_scorer
            self._set_active_reward_model(None)
            scoring = (
                previous_scorer
                if self.config.preference_scoring_reward == "reward_model" and previous_scorer is not None
                else self.task_reward_scorer
            )
            current_records, state = self._write_or_load_replay(
                iteration=iteration,
                state=state,
                scorer=scoring,
            )
            selected, state = self._select_replay(iteration, current_records, state)
            if not selected:
                if "training_skipped" not in state.completed_stages:
                    state = state.with_stage(
                        "training_skipped",
                        reward_model_checkpoint=self._active_reward_model_checkpoint,
                        active_reward_model_version=self._active_reward_model_version,
                    )
                    self.state_store.save(state)
                state = self._snapshot_iteration(iteration, state)
                self._complete(state)
                continue

            frozen = self._reward_scorer_for_iteration(iteration, tuple(selected), state)
            state = self.state_store.get_or_create(iteration)
            self._active_reward_scorer = frozen
            self._active_reward_model_version = state.active_reward_model_version
            self._active_reward_model_checkpoint = state.reward_model_checkpoint
            self._set_active_reward_model(frozen)
            if "online_rl_completed" not in state.completed_stages:
                state = state.with_stage("online_rl_running")
                self.state_store.save(state)
                _run_stage(
                    self.rl_runner,
                    iteration=iteration,
                    dispatch=resolve_marlhf_rl_dispatch(self.config.rl_algorithm),
                    reward_scorer=frozen,
                    task_reward_scorer=self.task_reward_scorer,
                    preference_records=tuple(selected),
                    num_train_epochs=self.config.num_train_epochs or 2,
                )
                state = state.transition_stage("online_rl_running", "online_rl_completed")
                self.state_store.save(state)
            state = self._snapshot_iteration(iteration, state)
            self._complete(state)
        return self.state_store.states()

    def _reward_scorer_for_iteration(
        self,
        iteration: int,
        selected: tuple[PreferenceReplayRecord, ...],
        state: IterationState,
    ) -> Any:
        if "reward_model_frozen" in state.completed_stages:
            return self._load_reward_model(state.reward_model_checkpoint, state.active_reward_model_version)
        state = state.with_stage("reward_model_training")
        self.state_store.save(state)
        factory = (
            self.reward_model_factory.create
            if hasattr(self.reward_model_factory, "create")
            else self.reward_model_factory
        )
        worker = factory(iteration=iteration)
        if id(worker) in self._reward_model_instances:
            raise RuntimeError("reward_model_factory reused a reward model instance")
        self._reward_model_instances.add(id(worker))
        for method in ("train", "freeze", "save_checkpoint"):
            if not callable(getattr(worker, method, None)):
                raise TypeError("reward model instances must implement train(), freeze(), and save_checkpoint()")
        worker.train(
            preference_records=selected,
            iteration=iteration,
            num_train_epochs=self.config.reward_num_train_epochs,
        )
        frozen = worker.freeze()
        if frozen is None:
            frozen = getattr(worker, "scorer", None)
        if frozen is None or not bool(getattr(frozen, "is_frozen", False)):
            raise RuntimeError("reward model freeze() must produce a frozen scorer")
        checkpoint = self.artifact_dir / "reward_models" / f"iteration_{iteration:04d}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        worker.save_checkpoint(checkpoint)
        state = state.transition_stage(
            "reward_model_training",
            "reward_model_frozen",
            reward_model_checkpoint=str(checkpoint),
            active_reward_model_version=f"iteration_{iteration:04d}",
        )
        self.state_store.save(state)
        close = getattr(worker, "close", None)
        if callable(close):
            close()
        return frozen

    def _load_reward_model(self, checkpoint: str | None, version: str | None) -> Any:
        if not checkpoint or not version:
            raise RuntimeError("frozen reward model state is missing checkpoint/version")
        loader = self.reward_model_loader or getattr(self.reward_model_factory, "load_frozen", None)
        if not callable(loader):
            raise RuntimeError("resuming a frozen reward model requires a frozen scorer loader")
        scorer = loader(checkpoint=Path(checkpoint), version=version)
        if scorer is None or not bool(getattr(scorer, "is_frozen", False)):
            raise RuntimeError("reward model loader must return a frozen scorer")
        return scorer

    def _restore_previous_reward_model(self, cursor: int) -> None:
        previous = [state for state in self.state_store.states() if state.iteration < cursor and state.completed]
        for state in reversed(previous):
            if state.reward_model_checkpoint and state.active_reward_model_version:
                self._active_reward_scorer = self._load_reward_model(
                    state.reward_model_checkpoint, state.active_reward_model_version
                )
                self._active_reward_model_checkpoint = state.reward_model_checkpoint
                self._active_reward_model_version = state.active_reward_model_version
                return

    def _set_active_reward_model(self, scorer: Any | None) -> None:
        if self.active_reward_model_setter is not None:
            self.active_reward_model_setter(scorer)


def _run_stage(stage: Any, **kwargs: Any) -> Any:
    target = stage.run if hasattr(stage, "run") else stage
    if not callable(target):
        raise TypeError("runner must be callable or implement run()")
    return target(**kwargs)


def _score_candidates(scorer: Any, result: Any, context: Mapping[str, Any] | None, iteration: int) -> RewardScores:
    target = scorer.score if hasattr(scorer, "score") else scorer
    scores = target(
        prompts=result.prompts,
        candidates_by_agent=result.candidates_by_agent,
        context=context or {},
        iteration=iteration,
    )
    if not isinstance(scores, RewardScores):
        raise TypeError("reward scorer must return RewardScores")
    if len(scores.processed) != result.num_candidates:
        raise ValueError("reward scorer must return one score per candidate")
    return scores


def _records_from_comparison(
    *,
    policy_iteration: int,
    replay_iteration: int,
    batch_index: int,
    current: Any,
    comparator: Any,
    current_scores: RewardScores,
    comparator_scores: RewardScores,
    pair_selection: str,
    pairs_per_sample: int | None,
    seed: int,
) -> list[PreferenceReplayRecord]:
    comparisons = compare_policy_candidates_by_index(current_scores.processed, comparator_scores.processed)
    selected = select_policy_comparisons(
        comparisons,
        mode=pair_selection,
        limit=pairs_per_sample,
        seed=seed,
    )
    all_rewards = (*current_scores.processed, *comparator_scores.processed)
    candidate_mean = sum(all_rewards) / len(all_rewards)
    records = []
    for comparison in selected:
        index = comparison.candidate_index
        current_text = tuple(values[index] for values in current.candidates_by_agent)
        comparator_text = tuple(values[index] for values in comparator.candidates_by_agent)
        chosen = current_text if comparison.winner_source == "current" else comparator_text
        rejected = comparator_text if comparison.loser_source == "comparator" else current_text
        chosen_reward = max(comparison.current_reward, comparison.comparator_reward)
        rejected_reward = min(comparison.current_reward, comparison.comparator_reward)
        records.append(
            PreferenceReplayRecord(
                pair_id=f"iter-{policy_iteration:04d}-batch-{batch_index:04d}-candidate-{index:04d}",
                iteration=replay_iteration,
                prompts=current.prompts,
                chosen=chosen,
                rejected=rejected,
                processed_chosen_reward=chosen_reward,
                processed_rejected_reward=rejected_reward,
                raw_policy_reward=current_scores.raw[index],
                raw_comparator_reward=comparator_scores.raw[index],
                raw_candidate_rewards=(*current_scores.raw, *comparator_scores.raw),
                candidate_mean=candidate_mean,
                policy_provenance={"policy": "current", **dict(current.provenance)},
                comparator_provenance=dict(comparator.provenance),
            )
        )
    return records


def _is_finite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}


__all__ = [
    "CoMLRLIterativeConfig",
    "CoMLRLIterativeController",
    "IterationState",
    "IterationStateStore",
    "IterativeConfig",
    "IterativeWorkflowError",
    "MADPOIterWorkflow",
    "MARLHFIterWorkflow",
    "PolicyLifecycle",
    "PromptBatch",
    "RewardScorer",
    "RewardScores",
    "UnsafeResumeError",
    "records_to_joint_preference_pairs",
]
