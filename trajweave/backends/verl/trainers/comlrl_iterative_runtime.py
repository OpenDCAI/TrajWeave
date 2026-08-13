from __future__ import annotations

import random
import shutil
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import transfer_queue as tq
from omegaconf import DictConfig, OmegaConf, open_dict
from transfer_queue import KVBatchMeta

from trajweave.backends.verl.routing import safe_actor_role_key
from trajweave.backends.verl.schema import to_python
from trajweave.backends.verl.trainers.comlrl_iterative import (
    CoMLRLIterativeConfig,
    CoMLRLIterativeController,
)
from trajweave.backends.verl.trainers.comlrl_staged import (
    preference_pairs_from_tq_batch,
    train_marlhf_reward_model,
)
from trajweave.core.preference import JointPreferencePair
from trajweave.orchestration.comlrl.comparator import ComparatorResult
from trajweave.storage.preference_replay import PolicySnapshotStore, PreferenceReplayRecord
from verl.utils.tensordict_utils import list_of_dict_to_tensordict


class _ControllerComparator:
    """The AgentLoop owns generation; this value satisfies the controller protocol."""

    def __init__(self, iteration: int) -> None:
        self.iteration = iteration

    def generate(self, prompts, *, num_candidates, iteration=None, context=None):
        return ComparatorResult(
            prompts=tuple(prompts),
            candidates_by_agent=tuple(tuple("unused" for _ in range(num_candidates)) for _ in prompts),
            policy="current",
            generation_mode="decentralized",
            iteration=iteration,
            provenance={"owned_by": "agent_loop"},
        )


def run_comlrl_iterative_training(*, trainer: Any, agent_loop_manager: Any, config: Any) -> None:
    settings = iterative_settings(config)
    algorithm = str(settings.get("algorithm", "")).strip().lower()
    canonical = {"madpo_iter": "MADPOIter", "marlhf_iter": "MARLHFIter"}.get(algorithm)
    if canonical is None:
        raise ValueError("iterative algorithm must be madpo_iter or marlhf_iter")
    preference_scoring = str(settings.get("preference_scoring_reward", "task")).strip().lower()
    if preference_scoring not in {"task", "reward_model"}:
        raise ValueError("iterative preference_scoring_reward must be task or reward_model")
    replay = settings.get("replay", {}) or {}
    comparator = settings.get("comparator", {}) or {}
    default_train_epochs = 1 if canonical == "MADPOIter" else 2
    num_train_epochs = int(settings.get("num_train_epochs", default_train_epochs))
    if num_train_epochs < 1:
        raise ValueError("iterative num_train_epochs must be positive")
    collection_batches = int(settings.get("preference_collection_batches", len(trainer.train_dataloader)))
    if collection_batches < 1:
        raise ValueError("iterative preference_collection_batches must be positive")
    if not isinstance(replay, dict) or not isinstance(comparator, dict):
        raise TypeError("iterative replay and comparator settings must be mappings")
    controller_config = CoMLRLIterativeConfig(
        algorithm=canonical,
        num_iterations=int(settings.get("num_iterations", 6)),
        num_train_epochs=settings.get("num_train_epochs"),
        num_target_candidates=int(settings.get("num_target_candidates", settings.get("preference_num_candidates", 20))),
        pairs_per_sample=settings.get("pairs_per_sample", 4),
        pair_selection=str(settings.get("pair_selection", "comparator_reward")),
        replay_mode=str(replay.get("mode", "current")),
        nearest_k=replay.get("k"),
        replay_lambda=replay.get("lambda"),
        replay_sample_size=replay.get("sample_size"),
        replay_seed=replay.get("seed"),
        replay_replacement=bool(replay.get("replacement", True)),
    )
    agent_ids = tuple(str(value) for value in to_python(config.agent.agent_ids))
    model_ids = tuple(str(value) for value in to_python(config.agent.model_ids))
    if len(agent_ids) != len(model_ids):
        raise ValueError("iterative agent_ids and model_ids must align")
    state_dir = settings.get("state_dir")
    if not state_dir:
        state_dir = Path(str(config.trainer.default_local_dir)) / "comlrl_iterative"
    trainer.agent_loop_manager = agent_loop_manager
    controller_ref: dict[str, CoMLRLIterativeController] = {}

    def collect(iteration, _controller_comparator, candidates, pairs_per_sample, pair_selection):
        context = {
            "phase": "preference",
            "iteration": iteration,
            "pair_selection": pair_selection,
            "pairs_per_sample": pairs_per_sample,
            "num_target_candidates": int(candidates),
            "random_seed": int(settings.get("random_seed", 0)),
            "preference_scoring_reward": preference_scoring,
            "comparator": _comparator_context(
                comparator,
                iteration=iteration,
                controller=controller_ref["controller"],
                trainer=trainer,
                model_ids=model_ids,
            ),
        }
        if preference_scoring == "reward_model" and canonical == "MARLHFIter":
            marlhf_context = to_python(config.trajweave.comlrl.marlhf)
            if isinstance(marlhf_context, dict) and marlhf_context.get("reward_model_active"):
                context["marlhf"] = dict(marlhf_context)
        agent_loop_manager.set_comlrl_iterative_context(context)
        pairs: list[JointPreferencePair] = []
        for _ in range(collection_batches):
            trainer._add_batch_to_generate()
            batch = trainer.replay_buffer.sample(
                partition_id="train",
                batch_size=int(config.data.train_batch_size),
            )
            try:
                try:
                    pairs.extend(preference_pairs_from_tq_batch(batch, expected_worker_groups=model_ids))
                except ValueError as error:
                    if "no active preference rows" not in str(error):
                        raise
            finally:
                tq.kv_clear(keys=batch.keys, partition_id=batch.partition_id)
        return _pairs_to_replay_records(tuple(pairs), iteration=iteration, agent_ids=agent_ids)

    def save_snapshot(iteration, target_path):
        metadata = _export_actor_snapshots(trainer, target_path, iteration)
        latest_paths = dict(getattr(trainer, "_trajweave_rollout_model_paths", {}))
        if latest_paths:
            agent_loop_manager.reload_local_models(latest_paths, policy_version=trainer.global_steps)
        return metadata

    kwargs = {
        "config": controller_config,
        "agent_ids": agent_ids,
        "workflow_dir": state_dir,
        "comparator_factory": _ControllerComparator,
        "collect_preferences": collect,
        "save_policy_snapshot": save_snapshot,
        "workflow_identity": {key: value for key, value in settings.items() if key != "run_mode"},
    }
    if canonical == "MADPOIter":
        preference_train_batch_size = _iterative_preference_train_batch_size(settings, config)

        def train_madpo(iteration, pairs):
            return _train_madpo_pair_batches(
                trainer,
                pairs,
                agent_ids=agent_ids,
                model_ids=model_ids,
                iteration=iteration,
                num_train_epochs=num_train_epochs,
                batch_size=preference_train_batch_size,
            )

        kwargs["train_madpo"] = train_madpo
    else:

        def train_reward_model(iteration, pairs):
            checkpoint_path = Path(str(state_dir)) / "reward_models" / f"iteration_{iteration:04d}" / "reward_model.pt"
            with open_dict(config) if isinstance(config, DictConfig) else nullcontext():
                if isinstance(config, DictConfig):
                    OmegaConf.update(
                        config,
                        "trajweave.comlrl.marlhf.reward_model_checkpoint",
                        str(checkpoint_path),
                        merge=False,
                        force_add=True,
                    )
                    OmegaConf.update(
                        config,
                        "trajweave.comlrl.marlhf.reward_model_version",
                        f"iteration-{iteration}",
                        merge=False,
                        force_add=True,
                    )
                else:
                    marlhf_config = config["trajweave"]["comlrl"].setdefault("marlhf", {})
                    marlhf_config["reward_model_checkpoint"] = str(checkpoint_path)
                    marlhf_config["reward_model_version"] = f"iteration-{iteration}"
            workflow = train_marlhf_reward_model(config, pairs)
            marlhf = to_python(config.trajweave.comlrl.marlhf)
            return {"workflow": workflow, "marlhf": dict(marlhf)}

        def train_online_rl(iteration, _pairs, reward_artifact):
            agent_loop_manager.set_comlrl_iterative_context(
                {
                    "phase": "online",
                    "iteration": iteration,
                    "marlhf": reward_artifact["marlhf"],
                }
            )
            step_metrics = []
            for _ in range(num_train_epochs * len(trainer.train_dataloader)):
                trainer.global_steps += 1
                timing_raw: dict[str, Any] = {}
                trainer.timing_raw = timing_raw
                trainer.on_step_begin()
                metrics: dict[str, Any] = {}
                batch = trainer.step(metrics, timing_raw)
                try:
                    trainer.on_step_end()
                finally:
                    tq.kv_clear(keys=batch.keys, partition_id=batch.partition_id)
                step_metrics.append(metrics)
            return tuple(step_metrics)

        kwargs["train_reward_model"] = train_reward_model
        kwargs["train_online_rl"] = train_online_rl
    controller = CoMLRLIterativeController(**kwargs)
    controller_ref["controller"] = controller
    if not controller.snapshot_store.initial_path.is_dir():
        controller.snapshot_store.commit_initial(
            lambda initial_path: _export_actor_snapshots(trainer, initial_path, -1),
            {"iteration": None},
        )
    mode = str(settings.get("run_mode", "fresh"))
    controller.run("resume" if mode == "resume" else "fresh")
    agent_loop_manager.set_comlrl_iterative_context(None)


def apply_iterative_resume_model_paths(config: Any) -> None:
    settings = iterative_settings(config)
    if not settings or str(settings.get("run_mode", "fresh")) != "resume":
        return
    trajweave = to_python(config.get("trajweave", {}))
    comlrl = trajweave.get("comlrl", {}) if isinstance(trajweave, dict) else {}
    online_algorithm = str(comlrl.get("algorithm", "")).strip().lower()
    if online_algorithm in {"iac", "maac"}:
        raise ValueError("MARLHF-Iter IAC/MAAC resume is unavailable until critic-route snapshots are persisted.")
    state_dir = settings.get("state_dir")
    if not state_dir:
        state_dir = Path(str(config.trainer.default_local_dir)) / "comlrl_iterative"
    snapshots = Path(str(state_dir)).expanduser().resolve() / "policy_snapshots"
    if not snapshots.is_dir():
        raise FileNotFoundError(f"iterative resume snapshot directory is missing: {snapshots}")
    snapshot_store = PolicySnapshotStore(snapshots)
    snapshot_root = (
        snapshot_store.iteration_path(snapshot_store.available_iterations[-1])
        if snapshot_store.available_iterations
        else snapshot_store.initial_path
    )
    if not snapshot_root.is_dir():
        raise FileNotFoundError("iterative resume has no completed policy snapshot")
    model_ids = tuple(str(value) for value in to_python(config.agent.model_ids))
    worker_groups = config.agent.worker_groups
    with open_dict(config) if isinstance(config, DictConfig) else nullcontext():
        for model_id in model_ids:
            model_path = str(snapshot_root / safe_actor_role_key(model_id) / "huggingface")
            if not Path(model_path).is_dir():
                raise FileNotFoundError(f"iterative resume actor snapshot is missing: {model_path}")
            if isinstance(worker_groups, DictConfig) and model_id in worker_groups:
                worker_groups[model_id].model_path = model_path
            elif isinstance(worker_groups, dict) and model_id in worker_groups:
                worker_groups[model_id]["model_path"] = model_path
            else:
                for group in worker_groups:
                    if str(group.get("id")) == model_id:
                        group["model_path"] = model_path
                        break
                else:
                    raise KeyError(f"iterative resume cannot find worker group {model_id!r}")
        if isinstance(config, DictConfig):
            OmegaConf.update(config, "trainer.resume_mode", "disable", merge=False, force_add=True)
        else:
            config.setdefault("trainer", {})["resume_mode"] = "disable"


def iterative_settings(config: Any) -> dict[str, Any]:
    trajweave = to_python(config.get("trajweave", {}))
    comlrl = trajweave.get("comlrl", {}) if isinstance(trajweave, dict) else {}
    iterative = comlrl.get("iterative", {}) if isinstance(comlrl, dict) else {}
    if not isinstance(iterative, dict) or not bool(iterative.get("enabled", False)):
        return {}
    nested = iterative.get("config")
    if isinstance(nested, dict):
        return {**nested, **{key: value for key, value in iterative.items() if key != "config"}}
    return dict(iterative)


def _comparator_context(
    comparator: dict[str, Any],
    *,
    iteration: int,
    controller: CoMLRLIterativeController,
    trainer: Any,
    model_ids: tuple[str, ...],
) -> dict[str, Any]:
    output = dict(comparator)
    policy = str(output.get("policy", "current"))
    if policy == "history":
        history_k = int(output.get("history_k", 1))
        root = controller.snapshot_store.resolve_history(iteration, history_k=history_k)
        output["model_paths"] = {
            model_id: str(root / safe_actor_role_key(model_id) / "huggingface") for model_id in model_ids
        }
    elif policy == "model":
        model_paths = output.get("model_paths")
        if model_paths is None:
            model_path = output.get("model_path", output.get("model_name"))
            if not model_path:
                raise ValueError("model comparator requires model_path/model_name or model_paths")
            root = Path(str(model_path)).expanduser().resolve()
            discovered = {model_id: root / safe_actor_role_key(model_id) / "huggingface" for model_id in model_ids}
            if all(path.is_dir() for path in discovered.values()):
                output["model_paths"] = {model_id: str(path) for model_id, path in discovered.items()}
            elif root.is_dir():
                output["model_paths"] = dict.fromkeys(model_ids, str(root))
            else:
                raise FileNotFoundError(f"model comparator path is missing: {root}")
    elif policy == "current_copy":
        store = PolicySnapshotStore(controller.workflow_dir / "comparator_snapshots")
        if iteration in store.available_iterations:
            root = store.iteration_path(iteration)
        else:
            root = store.commit_iteration(
                iteration,
                lambda temporary: _export_actor_snapshots(trainer, temporary, iteration),
                {"purpose": "current_copy", "iteration": iteration},
            )
        output["model_paths"] = {
            model_id: str(root / safe_actor_role_key(model_id) / "huggingface") for model_id in model_ids
        }
    return output


def _export_actor_snapshots(trainer: Any, target_path: Path, iteration: int) -> dict[str, Any]:
    model_paths = {}
    synced_paths = dict(getattr(trainer, "_trajweave_rollout_model_paths", {}))
    for group_id, worker_group in trainer.actor_rollout_wgs.items():
        if group_id not in trainer.multi_actor_trainable_group_ids:
            continue
        path = target_path / safe_actor_role_key(group_id)
        hf_path = path / "huggingface"
        synced_value = synced_paths.get(group_id)
        synced_hf_path = Path(str(synced_value)) if synced_value else None
        if synced_hf_path is not None and synced_hf_path.is_dir():
            shutil.copytree(synced_hf_path, hf_path)
        else:
            worker_group.export_hf_rollout_snapshot(str(path), trainer.global_steps, max_ckpt_to_keep=None)
        if not (hf_path / "config.json").is_file():
            raise RuntimeError(f"iterative actor snapshot is incomplete: {hf_path}")
        weight_files = tuple(hf_path.glob("*.safetensors")) + tuple(hf_path.glob("pytorch_model*.bin"))
        if not weight_files:
            raise RuntimeError(f"iterative actor snapshot contains no model weights: {hf_path}")
        model_paths[group_id] = str(hf_path)
    if not model_paths:
        raise RuntimeError("iterative snapshot produced no trainable actor model")
    return {"model_paths": model_paths, "iteration": iteration, "global_step": trainer.global_steps}


def _pairs_to_replay_records(
    pairs: tuple[JointPreferencePair, ...],
    *,
    iteration: int,
    agent_ids: tuple[str, ...],
) -> tuple[PreferenceReplayRecord, ...]:
    records = []
    occurrences: dict[str, int] = {}
    for pair in pairs:
        occurrence = occurrences.get(pair.preference_pair_id, 0)
        occurrences[pair.preference_pair_id] = occurrence + 1
        pair_id = pair.preference_pair_id if occurrence == 0 else f"{pair.preference_pair_id}:collection:{occurrence}"
        records.append(
            PreferenceReplayRecord(
                pair_id=pair_id,
                iteration=iteration,
                prompts=tuple(pair.prompts_by_agent[agent_id] for agent_id in agent_ids),
                chosen=tuple(pair.chosen_by_agent[agent_id] for agent_id in agent_ids),
                rejected=tuple(pair.rejected_by_agent[agent_id] for agent_id in agent_ids),
                processed_chosen_reward=pair.chosen_reward,
                processed_rejected_reward=pair.rejected_reward,
                raw_policy_reward=pair.metadata.get("raw_policy_reward"),
                raw_comparator_reward=pair.metadata.get("raw_comparator_reward"),
                raw_candidate_rewards=tuple(pair.metadata.get("raw_candidate_rewards", ())),
                candidate_mean=pair.candidate_mean,
                policy_provenance=dict(pair.metadata.get("policy_provenance", {"policy": "current"})),
                comparator_provenance=dict(pair.metadata.get("comparator_provenance", {})),
            )
        )
    return tuple(records)


def _iterative_preference_train_batch_size(settings: dict[str, Any], config: Any) -> int:
    # ``train_batch_size`` is the upstream CoMLRL MADPO/MADPOIter field. The
    # explicit preference alias is accepted for TrajWeave callers, while the
    # existing VERL data batch remains the backwards-compatible fallback.
    value = settings.get("train_batch_size")
    if value is None:
        value = settings.get("preference_train_batch_size")
    if value is None:
        value = config.data.train_batch_size
    parsed = int(to_python(value))
    if parsed < 1:
        raise ValueError("iterative preference train_batch_size must be positive")
    return parsed


def _train_madpo_pair_batches(
    trainer: Any,
    pairs: tuple[JointPreferencePair, ...],
    *,
    agent_ids: tuple[str, ...],
    model_ids: tuple[str, ...],
    iteration: int,
    num_train_epochs: int,
    batch_size: int,
) -> tuple[dict[str, Any], ...]:
    metrics = []
    for epoch in range(num_train_epochs):
        epoch_pairs = list(pairs)
        random.shuffle(epoch_pairs)
        for start in range(0, len(epoch_pairs), batch_size):
            metrics.append(
                _train_madpo_pairs(
                    trainer,
                    tuple(epoch_pairs[start : start + batch_size]),
                    agent_ids=agent_ids,
                    model_ids=model_ids,
                    iteration=iteration,
                    epoch=epoch,
                )
            )
    return tuple(metrics)


def _train_madpo_pairs(
    trainer: Any,
    pairs: tuple[JointPreferencePair, ...],
    *,
    agent_ids: tuple[str, ...],
    model_ids: tuple[str, ...],
    iteration: int,
    epoch: int,
) -> dict[str, Any]:
    trainer.global_steps += 1
    batch = _materialize_pairs_to_tq(
        trainer,
        pairs,
        agent_ids=agent_ids,
        model_ids=model_ids,
        iteration=iteration,
        epoch=epoch,
    )
    timing_raw: dict[str, Any] = {}
    trainer.timing_raw = timing_raw
    trainer.on_step_begin()
    metrics: dict[str, Any] = {}
    try:
        batch = trainer._balance_batch(batch, metrics)
        batch = trainer._compute_old_log_prob(batch, metrics)
        batch = trainer._compute_advantage(batch, metrics)
        trainer._update_actor(batch, metrics)
        trainer.on_step_end()
    finally:
        tq.kv_clear(keys=batch.keys, partition_id=batch.partition_id)
    return metrics


def _materialize_pairs_to_tq(
    trainer: Any,
    pairs: tuple[JointPreferencePair, ...],
    *,
    agent_ids: tuple[str, ...],
    model_ids: tuple[str, ...],
    iteration: int,
    epoch: int,
) -> KVBatchMeta:
    rows = []
    tags = []
    keys = []
    response_length = int(trainer.config.actor_rollout_ref.rollout.response_length)
    if response_length < 1:
        raise ValueError("iterative rollout response_length must be positive")
    for pair in pairs:
        for agent_id, model_id in zip(agent_ids, model_ids, strict=True):
            chosen_text = pair.chosen_by_agent[agent_id]
            rejected_text = pair.rejected_by_agent[agent_id]
            chosen_ids = _encode(trainer.tokenizer, chosen_text, special=False)[:response_length]
            rejected_ids = _encode(trainer.tokenizer, rejected_text, special=False)[:response_length]
            for side, response, reward, encoded_response in (
                ("chosen", chosen_text, pair.chosen_reward, chosen_ids),
                ("rejected", rejected_text, pair.rejected_reward, rejected_ids),
            ):
                prompt_ids = _encode(trainer.tokenizer, pair.prompts_by_agent[agent_id], special=True)
                response_ids = encoded_response
                tensors = _pad_pair_tokens(
                    trainer,
                    prompt_ids=prompt_ids,
                    response_ids=response_ids,
                    reward=reward,
                )
                key = f"iter-{iteration}-epoch-{epoch}-{uuid.uuid4().hex}"
                rows.append(
                    {
                        "uid": key,
                        "prompts": tensors["prompts"],
                        "responses": tensors["responses"],
                        "input_ids": tensors["input_ids"],
                        "attention_mask": tensors["attention_mask"],
                        "position_ids": tensors["position_ids"],
                        "response_mask": tensors["response_mask"],
                        "loss_mask": tensors["response_mask"].clone(),
                        "rm_scores": tensors["rm_scores"],
                        "agent_name": agent_id,
                        "agent_id": agent_id,
                        "policy_group": model_id,
                        "worker_group": model_id,
                        "traj_uid": pair.episode_id,
                        "turn_id": 0,
                        "preference_pair_id": pair.preference_pair_id,
                        "preference_side": side,
                        "chosen_reward": pair.chosen_reward,
                        "rejected_reward": pair.rejected_reward,
                        "candidate_mean": pair.candidate_mean,
                        "preference_loss_mask": 1.0,
                        "prompt_text": pair.prompts_by_agent[agent_id],
                        "response_text": response,
                        "completion_id": f"{pair.preference_pair_id}:{agent_id}:{side}",
                        "tree_node_id": pair.tree_node_id,
                        "joint_action_ids": [f"{pair.preference_pair_id}:{side}"],
                        "joint_transition_ids": [f"{pair.preference_pair_id}:{side}:transition"],
                        "joint_return_components": [reward],
                        "joint_reward": reward,
                        "joint_done": True,
                        "joint_truncated": False,
                        "joint_stop_reason": "",
                        "joint_sampling_mode": "aligned",
                    }
                )
                keys.append(key)
                stored_prompt_len = int(tensors["prompts"].numel())
                stored_response_len = int(tensors["responses"].numel())
                tags.append(
                    {
                        "status": "success",
                        "prompt_len": stored_prompt_len,
                        "response_len": stored_response_len,
                        "seq_len": stored_prompt_len + stored_response_len,
                        "global_steps": trainer.global_steps,
                        "min_global_steps": trainer.global_steps,
                        "max_global_steps": trainer.global_steps,
                    }
                )
    tq.kv_batch_put(
        keys=keys,
        partition_id="train",
        fields=list_of_dict_to_tensordict(rows),
        tags=tags,
    )
    return KVBatchMeta(keys=keys, tags=tags, partition_id="train")


def _pad_pair_tokens(
    trainer: Any,
    *,
    prompt_ids: list[int],
    response_ids: list[int],
    reward: float,
) -> dict[str, torch.Tensor]:
    rollout = trainer.config.actor_rollout_ref.rollout
    prompt_length = int(rollout.prompt_length)
    response_length = int(rollout.response_length)
    if prompt_length < 1 or response_length < 1:
        raise ValueError("iterative rollout prompt_length and response_length must be positive")
    prompt_ids = prompt_ids[-prompt_length:]
    response_ids = response_ids[:response_length]
    pad_token_id = trainer.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = trainer.tokenizer.eos_token_id or 0
    prompt_pad = prompt_length - len(prompt_ids)
    response_pad = response_length - len(response_ids)
    prompts = torch.tensor([pad_token_id] * prompt_pad + prompt_ids, dtype=torch.long)
    responses = torch.tensor(response_ids + [pad_token_id] * response_pad, dtype=torch.long)
    response_mask = torch.tensor([1] * len(response_ids) + [0] * response_pad, dtype=torch.long)
    input_ids = torch.cat((prompts, responses))
    attention_mask = torch.tensor(
        [0] * prompt_pad + [1] * len(prompt_ids) + [1] * len(response_ids) + [0] * response_pad,
        dtype=torch.long,
    )
    position_ids = (attention_mask.cumsum(dim=0) - 1).clamp_min(0)
    rm_scores = torch.zeros(response_length, dtype=torch.float32)
    if response_ids:
        rm_scores[len(response_ids) - 1] = float(reward)
    return {
        "prompts": prompts,
        "responses": responses,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "response_mask": response_mask,
        "rm_scores": rm_scores,
    }


def _encode(tokenizer: Any, text: str, *, special: bool) -> list[int]:
    try:
        values = tokenizer.encode(text, add_special_tokens=special)
    except TypeError:
        values = tokenizer.encode(text)
    values = [int(value) for value in values]
    if not values:
        values = [int(tokenizer.eos_token_id or tokenizer.pad_token_id or 0)]
    return values


__all__ = [
    "apply_iterative_resume_model_paths",
    "iterative_settings",
    "run_comlrl_iterative_training",
]
