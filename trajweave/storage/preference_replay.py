from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from trajweave.storage.serialization import redact_secrets

REPLAY_SCHEMA_VERSION = 1
SNAPSHOT_MANIFEST_SCHEMA_VERSION = 1
_ITERATION_FILE_RE = re.compile(r"^iteration_(\d+)\.json$")
_ITERATION_DIR_RE = re.compile(r"^iteration_(\d+)$")
_REPLAY_MODE_ALIASES = {
    "current": "current",
    "latest": "current",
    "new": "current",
    "nearest_k": "nearest_k",
    "recent_k": "nearest_k",
    "last_k": "nearest_k",
    "k": "nearest_k",
    "all_history": "all_history",
    "lambda": "lambda_decay",
    "lambda_decay": "lambda_decay",
    "td_lambda": "lambda_decay",
    "td-lambda": "lambda_decay",
}


class PreferenceReplayError(ValueError):
    pass


class PreferenceReplaySchemaError(PreferenceReplayError):
    pass


class DuplicateIterationError(PreferenceReplayError):
    pass


def canonical_replay_mode(mode: str | None) -> str:
    value = str(mode or "current").strip().lower()
    try:
        return _REPLAY_MODE_ALIASES[value]
    except KeyError as exc:
        raise ValueError("replay mode must be one of: current, nearest_k, all_history, lambda_decay") from exc


@dataclass(frozen=True)
class PreferenceReplayRecord:
    iteration: int
    prompts: tuple[str, ...]
    chosen: tuple[str, ...]
    rejected: tuple[str, ...]
    processed_chosen_reward: float
    processed_rejected_reward: float
    raw_policy_reward: float | None
    raw_comparator_reward: float | None
    candidate_mean: float
    policy_provenance: Mapping[str, Any]
    comparator_provenance: Mapping[str, Any]
    pair_id: str = ""
    raw_candidate_rewards: tuple[float, ...] = ()
    schema_version: int = field(default=REPLAY_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        if isinstance(self.iteration, bool) or not isinstance(self.iteration, int) or self.iteration < 0:
            raise ValueError("iteration must be a non-negative integer")
        prompts = _text_tuple(self.prompts, "prompts", allow_empty_text=False)
        chosen = _text_tuple(self.chosen, "chosen", allow_empty_text=False)
        rejected = _text_tuple(self.rejected, "rejected", allow_empty_text=False)
        if not prompts or len(prompts) != len(chosen) or len(prompts) != len(rejected):
            raise ValueError("prompts, chosen, and rejected must contain the same non-zero number of agents")
        object.__setattr__(self, "prompts", prompts)
        object.__setattr__(self, "chosen", chosen)
        object.__setattr__(self, "rejected", rejected)

        for name in ("processed_chosen_reward", "processed_rejected_reward", "candidate_mean"):
            _finite_float(getattr(self, name), name)
        for name in ("raw_policy_reward", "raw_comparator_reward"):
            value = getattr(self, name)
            if value is not None:
                _finite_float(value, name)
        raw_candidates = tuple(_finite_float(value, "raw_candidate_rewards") for value in self.raw_candidate_rewards)
        object.__setattr__(self, "raw_candidate_rewards", raw_candidates)
        if not isinstance(self.policy_provenance, Mapping) or not isinstance(self.comparator_provenance, Mapping):
            raise TypeError("policy_provenance and comparator_provenance must be mappings")
        object.__setattr__(self, "policy_provenance", dict(self.policy_provenance))
        object.__setattr__(self, "comparator_provenance", dict(self.comparator_provenance))

        stable_id = self.stable_pair_id(
            iteration=self.iteration,
            prompts=prompts,
            chosen=chosen,
            rejected=rejected,
        )
        if self.pair_id:
            if not isinstance(self.pair_id, str):
                raise TypeError("pair_id must be a string")
        else:
            object.__setattr__(self, "pair_id", stable_id)

    @staticmethod
    def stable_pair_id(
        *,
        iteration: int,
        prompts: Sequence[str],
        chosen: Sequence[str],
        rejected: Sequence[str],
    ) -> str:
        canonical = json.dumps(
            {
                "iteration": iteration,
                "prompts": list(prompts),
                "chosen": list(chosen),
                "rejected": list(rejected),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"pref-{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:24]}"

    @property
    def processed_rewards(self) -> dict[str, float]:
        return {
            "chosen": self.processed_chosen_reward,
            "rejected": self.processed_rejected_reward,
        }

    @property
    def raw_rewards(self) -> dict[str, Any]:
        return {
            "policy": self.raw_policy_reward,
            "comparator": self.raw_comparator_reward,
            "candidates": list(self.raw_candidate_rewards),
        }

    def to_dict(self) -> dict[str, Any]:
        return redact_secrets(
            {
                "schema_version": self.schema_version,
                "pair_id": self.pair_id,
                "iteration": self.iteration,
                "prompts": list(self.prompts),
                "chosen": list(self.chosen),
                "rejected": list(self.rejected),
                "processed_rewards": self.processed_rewards,
                "raw_rewards": self.raw_rewards,
                "candidate_mean": self.candidate_mean,
                "policy_provenance": dict(self.policy_provenance),
                "comparator_provenance": dict(self.comparator_provenance),
            }
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PreferenceReplayRecord:
        if not isinstance(payload, Mapping):
            raise PreferenceReplaySchemaError("preference replay record must be an object")
        if payload.get("schema_version") != REPLAY_SCHEMA_VERSION:
            raise PreferenceReplaySchemaError(
                f"unsupported preference replay record schema_version: {payload.get('schema_version')!r}"
            )
        processed = payload.get("processed_rewards")
        raw = payload.get("raw_rewards")
        if not isinstance(processed, Mapping):
            raise PreferenceReplaySchemaError("record.processed_rewards must be an object")
        if not isinstance(raw, Mapping):
            raise PreferenceReplaySchemaError("record.raw_rewards must be an object")
        try:
            return cls(
                pair_id=payload["pair_id"],
                iteration=payload["iteration"],
                prompts=tuple(payload["prompts"]),
                chosen=tuple(payload["chosen"]),
                rejected=tuple(payload["rejected"]),
                processed_chosen_reward=processed["chosen"],
                processed_rejected_reward=processed["rejected"],
                raw_policy_reward=raw.get("policy"),
                raw_comparator_reward=raw.get("comparator"),
                raw_candidate_rewards=tuple(raw.get("candidates", [])),
                candidate_mean=payload["candidate_mean"],
                policy_provenance=payload["policy_provenance"],
                comparator_provenance=payload["comparator_provenance"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PreferenceReplaySchemaError(f"invalid preference replay record: {exc}") from exc


@dataclass(frozen=True)
class PreferenceReplayShard:
    iteration: int
    path: Path
    num_records: int


class PreferenceReplayStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self._shards: dict[int, PreferenceReplayShard] = {}
        self._scan_existing_shards()
        self._validate_existing_manifest()
        self._write_manifest()

    @property
    def available_iterations(self) -> tuple[int, ...]:
        return tuple(sorted(self._shards))

    @property
    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": REPLAY_SCHEMA_VERSION,
            "available_iterations": list(self.available_iterations),
            "shards": [
                {
                    "iteration": shard.iteration,
                    "path": shard.path.name,
                    "num_records": shard.num_records,
                }
                for shard in (self._shards[index] for index in self.available_iterations)
            ],
        }

    def shard_path(self, iteration: int) -> Path:
        iteration = _iteration(iteration)
        return self.root / f"iteration_{iteration:04d}.json"

    def write_iteration(
        self,
        iteration: int,
        records: Sequence[PreferenceReplayRecord],
    ) -> PreferenceReplayShard:
        iteration = _iteration(iteration)
        if iteration in self._shards:
            raise DuplicateIterationError(f"preference replay iteration {iteration} already exists")
        normalized = tuple(records)
        if any(not isinstance(record, PreferenceReplayRecord) for record in normalized):
            raise TypeError("records must contain PreferenceReplayRecord values")
        if any(record.iteration != iteration for record in normalized):
            raise ValueError("every replay record iteration must match the shard iteration")
        pair_ids = [record.pair_id for record in normalized]
        if len(pair_ids) != len(set(pair_ids)):
            raise ValueError("pair_id values must be unique within an iteration shard")

        path = self.shard_path(iteration)
        lock_path = self.root / f".iteration_{iteration:04d}.lock"
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise DuplicateIterationError(f"preference replay iteration {iteration} is being written") from exc
        try:
            os.close(lock_fd)
            if path.exists():
                raise DuplicateIterationError(f"preference replay iteration {iteration} already exists")
            payload = {
                "schema_version": REPLAY_SCHEMA_VERSION,
                "iteration": iteration,
                "num_records": len(normalized),
                "records": [record.to_dict() for record in normalized],
            }
            _atomic_write_json(path, payload)
            shard = PreferenceReplayShard(iteration=iteration, path=path, num_records=len(normalized))
            self._shards[iteration] = shard
            self._write_manifest()
            return shard
        finally:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass

    def load_iteration(self, iteration: int) -> tuple[PreferenceReplayRecord, ...]:
        iteration = _iteration(iteration)
        try:
            shard = self._shards[iteration]
        except KeyError as exc:
            raise KeyError(f"preference replay iteration {iteration} is unavailable") from exc
        return self._load_shard(shard)

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
    ) -> list[PreferenceReplayRecord]:
        canonical_mode = canonical_replay_mode(mode)
        if not self._shards:
            return []
        if current_iteration is None:
            current_iteration = max(self._shards)
        current_iteration = _iteration(current_iteration)
        if current_iteration not in self._shards:
            raise KeyError(f"preference replay iteration {current_iteration} is unavailable")
        if sample_size is not None:
            if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < 0:
                raise ValueError("sample_size must be a non-negative integer or None")
        if not isinstance(replacement, bool):
            raise TypeError("replacement must be a boolean")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise TypeError("seed must be an integer or None")

        eligible = [self._shards[index] for index in sorted(self._shards) if index <= current_iteration]
        current_shard = self._shards[current_iteration]
        if canonical_mode == "current":
            records = list(self._load_shard(current_shard))
            if sample_size is None:
                return records
            return self._sample_records_from_one_shard(
                records,
                sample_size,
                random.Random(seed),
                replacement=replacement,
            )

        if canonical_mode == "nearest_k":
            if isinstance(nearest_k, bool) or not isinstance(nearest_k, int) or nearest_k < 1:
                raise ValueError("nearest_k must be a positive integer for nearest_k replay")
            eligible = eligible[-nearest_k:]
        elif nearest_k is not None:
            raise ValueError("nearest_k is only valid for nearest_k replay")

        weights: list[float] | None = None
        if canonical_mode == "lambda_decay":
            if replay_lambda is None or isinstance(replay_lambda, bool) or not isinstance(replay_lambda, int | float):
                raise ValueError("replay_lambda must be provided for lambda_decay replay")
            replay_lambda = float(replay_lambda)
            if not 0.0 <= replay_lambda <= 1.0:
                raise ValueError("replay_lambda must be in [0, 1]")
            weights = _lambda_decay_weights(len(eligible), replay_lambda)
        elif replay_lambda is not None:
            raise ValueError("replay_lambda is only valid for lambda_decay replay")

        if sample_size is None:
            sample_size = current_shard.num_records
            if sample_size == 0:
                sample_size = next((shard.num_records for shard in reversed(eligible) if shard.num_records), 0)
        return self._sample_shards(
            eligible,
            sample_size,
            weights=weights,
            rng=random.Random(seed),
            replacement=replacement,
        )

    def _sample_shards(
        self,
        shards: Sequence[PreferenceReplayShard],
        sample_size: int,
        *,
        weights: Sequence[float] | None,
        rng: random.Random,
        replacement: bool,
    ) -> list[PreferenceReplayRecord]:
        if sample_size <= 0:
            return []
        eligible: list[PreferenceReplayShard] = []
        eligible_weights: list[float] = []
        for index, shard in enumerate(shards):
            if shard.num_records <= 0:
                continue
            weight = float(weights[index]) if weights is not None else 1.0
            if weight < 0 or not math.isfinite(weight):
                raise ValueError("shard weights must be finite and non-negative")
            if weight == 0:
                continue
            eligible.append(shard)
            eligible_weights.append(weight)
        if not eligible:
            return []

        records_by_shard = [list(self._load_shard(shard)) for shard in eligible]
        if not replacement:
            total = sum(len(records) for records in records_by_shard)
            if sample_size > total:
                raise ValueError("sample_size exceeds available records when replacement=False")
            output: list[PreferenceReplayRecord] = []
            for _ in range(sample_size):
                active = [index for index, records in enumerate(records_by_shard) if records]
                active_weights = [eligible_weights[index] for index in active]
                shard_index = rng.choices(active, weights=active_weights, k=1)[0]
                record_index = rng.randrange(len(records_by_shard[shard_index]))
                output.append(records_by_shard[shard_index].pop(record_index))
            rng.shuffle(output)
            return output

        shard_counts = [0] * len(eligible)
        for _ in range(sample_size):
            shard_index = rng.choices(range(len(eligible)), weights=eligible_weights, k=1)[0]
            shard_counts[shard_index] += 1
        output = []
        for records, count in zip(records_by_shard, shard_counts, strict=True):
            output.extend(self._sample_records_from_one_shard(records, count, rng, replacement=True))
        rng.shuffle(output)
        return output

    @staticmethod
    def _sample_records_from_one_shard(
        records: Sequence[PreferenceReplayRecord],
        count: int,
        rng: random.Random,
        *,
        replacement: bool,
    ) -> list[PreferenceReplayRecord]:
        if count <= 0 or not records:
            return []
        values = list(records)
        if not replacement:
            if count > len(values):
                raise ValueError("sample_size exceeds available records when replacement=False")
            return rng.sample(values, k=count)
        if count <= len(values):
            return rng.sample(values, k=count)
        output: list[PreferenceReplayRecord] = []
        remaining = count
        while remaining > 0:
            if remaining >= len(values):
                batch = list(values)
                rng.shuffle(batch)
                output.extend(batch)
                remaining -= len(batch)
            else:
                output.extend(rng.sample(values, k=remaining))
                remaining = 0
        return output

    def _scan_existing_shards(self) -> None:
        discovered: dict[int, Path] = {}
        for path in sorted(self.root.glob("iteration_*.json")):
            match = _ITERATION_FILE_RE.fullmatch(path.name)
            if match is None:
                continue
            iteration = int(match.group(1))
            if iteration in discovered:
                raise DuplicateIterationError(
                    f"duplicate preference replay files for iteration {iteration}: "
                    f"{discovered[iteration].name}, {path.name}"
                )
            discovered[iteration] = path
        for iteration, path in discovered.items():
            shard = self._read_shard_metadata(path, expected_iteration=iteration)
            self._shards[iteration] = shard

    def _read_shard_metadata(self, path: Path, *, expected_iteration: int) -> PreferenceReplayShard:
        payload = _read_json_object(path, description="preference replay shard")
        if payload.get("schema_version") != REPLAY_SCHEMA_VERSION:
            raise PreferenceReplaySchemaError(
                f"unsupported schema_version in {path.name}: {payload.get('schema_version')!r}"
            )
        if payload.get("iteration") != expected_iteration:
            raise PreferenceReplaySchemaError(
                f"shard {path.name} declares iteration {payload.get('iteration')!r}, expected {expected_iteration}"
            )
        records = payload.get("records")
        if not isinstance(records, list):
            raise PreferenceReplaySchemaError(f"shard {path.name} records must be a list")
        if payload.get("num_records") != len(records):
            raise PreferenceReplaySchemaError(f"shard {path.name} num_records does not match records")
        parsed = tuple(PreferenceReplayRecord.from_dict(record) for record in records)
        if any(record.iteration != expected_iteration for record in parsed):
            raise PreferenceReplaySchemaError(f"shard {path.name} contains a record from another iteration")
        pair_ids = [record.pair_id for record in parsed]
        if len(pair_ids) != len(set(pair_ids)):
            raise PreferenceReplaySchemaError(f"shard {path.name} contains duplicate pair_id values")
        return PreferenceReplayShard(expected_iteration, path, len(parsed))

    def _load_shard(self, shard: PreferenceReplayShard) -> tuple[PreferenceReplayRecord, ...]:
        payload = _read_json_object(shard.path, description="preference replay shard")
        records = payload.get("records")
        if payload.get("schema_version") != REPLAY_SCHEMA_VERSION or not isinstance(records, list):
            raise PreferenceReplaySchemaError(f"invalid replay schema in {shard.path.name}")
        try:
            parsed = tuple(PreferenceReplayRecord.from_dict(record) for record in records)
        except PreferenceReplaySchemaError:
            raise
        if len(parsed) != shard.num_records or payload.get("num_records") != shard.num_records:
            raise PreferenceReplaySchemaError(f"replay shard changed after startup: {shard.path.name}")
        return parsed

    def _validate_existing_manifest(self) -> None:
        if not self.manifest_path.exists():
            return
        payload = _read_json_object(self.manifest_path, description="preference replay manifest")
        if payload.get("schema_version") != REPLAY_SCHEMA_VERSION:
            raise PreferenceReplaySchemaError(
                f"unsupported replay manifest schema_version: {payload.get('schema_version')!r}"
            )
        iterations = payload.get("available_iterations")
        invalid_iterations = not isinstance(iterations, list) or any(
            isinstance(value, bool) or not isinstance(value, int) for value in iterations
        )
        if invalid_iterations:
            raise PreferenceReplaySchemaError("manifest.available_iterations must be an integer list")

    def _write_manifest(self) -> None:
        _atomic_write_json(self.manifest_path, self.manifest)


class PolicySnapshotStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self._initial_metadata: dict[str, Any] | None = None
        self._iteration_metadata: dict[int, dict[str, Any]] = {}
        self._load_manifest()
        self._validate_committed_snapshots()
        self._write_manifest()

    @property
    def initial_path(self) -> Path:
        return self.root / "initial"

    @property
    def available_iterations(self) -> tuple[int, ...]:
        return tuple(sorted(self._iteration_metadata))

    @property
    def manifest(self) -> dict[str, Any]:
        snapshots: dict[str, Any] = {
            "initial": (
                {"path": "initial", "metadata": redact_secrets(self._initial_metadata)}
                if self._initial_metadata is not None
                else None
            ),
            "iterations": {
                str(iteration): {
                    "path": self.iteration_path(iteration).name,
                    "metadata": redact_secrets(self._iteration_metadata[iteration]),
                }
                for iteration in self.available_iterations
            },
        }
        return {"schema_version": SNAPSHOT_MANIFEST_SCHEMA_VERSION, "snapshots": snapshots}

    def iteration_path(self, iteration: int) -> Path:
        iteration = _iteration(iteration)
        return self.root / f"iteration_{iteration:04d}"

    def register_initial(self, metadata: Mapping[str, Any] | None = None) -> Path:
        self.initial_path.mkdir(parents=False, exist_ok=True)
        self._initial_metadata = dict(metadata or {})
        self._write_manifest()
        return self.initial_path

    def register_iteration(self, iteration: int, metadata: Mapping[str, Any] | None = None) -> Path:
        iteration = _iteration(iteration)
        path = self.iteration_path(iteration)
        path.mkdir(parents=False, exist_ok=True)
        self._iteration_metadata[iteration] = dict(metadata or {})
        self._write_manifest()
        return path

    create_initial = register_initial
    create_iteration = register_iteration

    def commit_initial(
        self,
        writer: Callable[[Path], Mapping[str, Any] | None],
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        if self._initial_metadata is not None or self.initial_path.exists():
            raise DuplicateIterationError("initial policy snapshot already exists")
        committed_metadata = self._commit_snapshot_directory(
            self.initial_path,
            writer=writer,
            metadata=metadata,
        )
        self._initial_metadata = committed_metadata
        self._write_manifest()
        return self.initial_path

    def commit_iteration(
        self,
        iteration: int,
        writer: Callable[[Path], Mapping[str, Any] | None],
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        iteration = _iteration(iteration)
        path = self.iteration_path(iteration)
        if iteration in self._iteration_metadata or path.exists():
            raise DuplicateIterationError(f"policy snapshot iteration {iteration} already exists")
        committed_metadata = self._commit_snapshot_directory(
            path,
            writer=writer,
            metadata=metadata,
        )
        self._iteration_metadata[iteration] = committed_metadata
        self._write_manifest()
        return path

    def _commit_snapshot_directory(
        self,
        path: Path,
        *,
        writer: Callable[[Path], Mapping[str, Any] | None],
        metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if not callable(writer):
            raise TypeError("snapshot writer must be callable")
        temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}.", suffix=".tmp", dir=self.root))
        try:
            writer_metadata = writer(temporary)
            if writer_metadata is not None and not isinstance(writer_metadata, Mapping):
                raise TypeError("snapshot writer must return a mapping or None")
            if not any(temporary.iterdir()):
                raise RuntimeError("policy snapshot writer produced an empty directory")
            committed_metadata = dict(metadata or {})
            committed_metadata.update(dict(writer_metadata or {}))
            os.replace(temporary, path)
            return committed_metadata
        finally:
            if temporary.exists():
                import shutil

                shutil.rmtree(temporary)

    def metadata_for(self, iteration: int | None) -> Mapping[str, Any]:
        if iteration is None:
            if self._initial_metadata is None:
                raise KeyError("initial policy snapshot is unavailable")
            return dict(self._initial_metadata)
        iteration = _iteration(iteration)
        try:
            return dict(self._iteration_metadata[iteration])
        except KeyError as exc:
            raise KeyError(f"policy snapshot iteration {iteration} is unavailable") from exc

    def resolve_history(self, iteration: int, *, history_k: int = 1) -> Path:
        iteration = _iteration(iteration)
        if isinstance(history_k, bool) or not isinstance(history_k, int) or history_k < 1:
            raise ValueError("history_k must be a positive integer")
        if self._initial_metadata is None or not self.initial_path.is_dir():
            raise FileNotFoundError("initial policy snapshot is unavailable")
        target_iteration = iteration - history_k
        if target_iteration < 0:
            return self.initial_path
        candidates = [value for value in self.available_iterations if value <= target_iteration]
        if not candidates:
            return self.initial_path
        path = self.iteration_path(max(candidates))
        if not path.is_dir():
            raise FileNotFoundError(f"policy snapshot directory is missing: {path}")
        return path

    def _load_manifest(self) -> None:
        if not self.manifest_path.exists():
            return
        payload = _read_json_object(self.manifest_path, description="policy snapshot manifest")
        if payload.get("schema_version") != SNAPSHOT_MANIFEST_SCHEMA_VERSION:
            raise PreferenceReplaySchemaError(
                f"unsupported policy snapshot manifest schema_version: {payload.get('schema_version')!r}"
            )
        snapshots = payload.get("snapshots")
        if not isinstance(snapshots, Mapping):
            raise PreferenceReplaySchemaError("policy snapshot manifest.snapshots must be an object")
        initial = snapshots.get("initial")
        if initial is not None:
            if not isinstance(initial, Mapping) or not isinstance(initial.get("metadata", {}), Mapping):
                raise PreferenceReplaySchemaError("invalid initial policy snapshot manifest entry")
            self._initial_metadata = dict(initial.get("metadata", {}))
        iterations = snapshots.get("iterations", {})
        if not isinstance(iterations, Mapping):
            raise PreferenceReplaySchemaError("policy snapshot iterations must be an object")
        for raw_iteration, entry in iterations.items():
            if not str(raw_iteration).isdigit() or not isinstance(entry, Mapping):
                raise PreferenceReplaySchemaError("invalid policy snapshot iteration manifest entry")
            metadata = entry.get("metadata", {})
            if not isinstance(metadata, Mapping):
                raise PreferenceReplaySchemaError("policy snapshot metadata must be an object")
            self._iteration_metadata[int(raw_iteration)] = dict(metadata)

    def _validate_committed_snapshots(self) -> None:
        if self._initial_metadata is not None and not self.initial_path.is_dir():
            raise FileNotFoundError(f"policy snapshot directory is missing: {self.initial_path}")
        if self.initial_path.is_dir() and self._initial_metadata is None:
            raise PreferenceReplaySchemaError(
                f"uncommitted policy snapshot directory is not manifest-visible: {self.initial_path}"
            )
        missing = [iteration for iteration in self._iteration_metadata if not self.iteration_path(iteration).is_dir()]
        if missing:
            raise FileNotFoundError(f"policy snapshot directories are missing for iterations: {sorted(missing)}")
        committed = set(self._iteration_metadata)
        orphaned = []
        for path in self.root.iterdir():
            if not path.is_dir():
                continue
            match = _ITERATION_DIR_RE.fullmatch(path.name)
            if match is not None and int(match.group(1)) not in committed:
                orphaned.append(path.name)
        if orphaned:
            raise PreferenceReplaySchemaError(
                f"uncommitted policy snapshot directories are not manifest-visible: {sorted(orphaned)}"
            )

    def _write_manifest(self) -> None:
        _atomic_write_json(self.manifest_path, self.manifest)


def _lambda_decay_weights(num_shards: int, replay_lambda: float) -> list[float]:
    if num_shards < 1:
        return []
    if replay_lambda == 1.0:
        return [1.0 / num_shards] * num_shards
    newest_first = [(1.0 - replay_lambda) * (replay_lambda**age) for age in range(num_shards)]
    normalizer = sum(newest_first)
    if normalizer <= 0:
        newest_first = [1.0] + [0.0] * (num_shards - 1)
        normalizer = 1.0
    return list(reversed([weight / normalizer for weight in newest_first]))


def _iteration(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("iteration must be a non-negative integer")
    return value


def _text_tuple(values: Sequence[str], name: str, *, allow_empty_text: bool) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of strings")
    normalized = tuple(values)
    if any(not isinstance(value, str) or (not allow_empty_text and not value) for value in normalized):
        raise ValueError(f"{name} must contain non-empty strings")
    return normalized


def _finite_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{name} must contain finite numbers")
    return float(value)


def _read_json_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreferenceReplaySchemaError(f"cannot read {description} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PreferenceReplaySchemaError(f"{description} {path} must contain a JSON object")
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(redact_secrets(dict(payload)), handle, ensure_ascii=False, indent=2, sort_keys=True)
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


__all__ = [
    "DuplicateIterationError",
    "PolicySnapshotStore",
    "PreferenceReplayError",
    "PreferenceReplayRecord",
    "PreferenceReplaySchemaError",
    "PreferenceReplayShard",
    "PreferenceReplayStore",
    "REPLAY_SCHEMA_VERSION",
    "canonical_replay_mode",
]
