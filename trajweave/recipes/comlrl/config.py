from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from trajweave.backends.verl.multi_actor import (
    hydra_string_list,
    hydra_worker_group_list,
    normalize_worker_groups,
    replace_override,
    validate_multi_actor_worker_groups,
)
from trajweave.backends.verl.multi_actor.critic_config import normalize_critic_route_specs
from trajweave.backends.verl.runtime_config import normalize_hf_local_dtype

TRAJWEAVE_AGENT_LOOP_MANAGER_FQN = "trajweave.backends.verl.agent_loop.TrajWeaveAgentLoopManager"
COMLRL_REINFORCE_HOOK_FQN = "trajweave.backends.verl.extensions.comlrl.reinforce.CoMLRLReinforceHooks"
_REINFORCE_ALGORITHMS = {"magrpo", "mareinforce", "marloo", "maremax"}
_ACTOR_CRITIC_ALGORITHMS = {"iac", "maac"}
_ITERATIVE_ALIASES = {
    "madpo_iter": "madpo_iter",
    "madpo-iter": "madpo_iter",
    "madpoiter": "madpo_iter",
    "marlhf_iter": "marlhf_iter",
    "marlhf-iter": "marlhf_iter",
    "marlhfiter": "marlhf_iter",
}


def resolve_comlrl_topology(
    config: dict[str, Any],
    *,
    require_worker_assets: bool = False,
) -> dict[str, Any]:
    comlrl = config.get("comlrl", {}) or {}
    agent_ids = tuple(str(value) for value in comlrl.get("agent_ids", ["agent_0", "agent_1"]))
    model_ids = tuple(str(value) for value in comlrl.get("model_ids", [f"policy_{i}" for i in range(len(agent_ids))]))
    if len(agent_ids) < 2 or len(agent_ids) != len(set(agent_ids)):
        raise ValueError("CoMLRL requires at least two unique agent_ids.")
    if len(model_ids) != len(agent_ids) or len(model_ids) != len(set(model_ids)):
        raise ValueError("CoMLRL requires one unique model_id per agent.")
    worker_groups = normalize_worker_groups(
        comlrl.get("worker_groups", {}),
        model_ids=model_ids,
        family="CoMLRL",
    )
    validation = (
        validate_multi_actor_worker_groups(
            comlrl,
            worker_groups=worker_groups,
            model_ids=model_ids,
            enabled=True,
            family="CoMLRL",
        )
        if require_worker_assets
        else {"status": "not_required", "tokenizer_mode": str(comlrl.get("tokenizer_mode", "shared"))}
    )
    return {
        "agent_ids": agent_ids,
        "model_ids": model_ids,
        "worker_groups": worker_groups,
        "validation": validation,
    }


def build_comlrl_launch_overrides(
    config: dict[str, Any],
    *,
    config_path: str | None,
) -> tuple[str, ...]:
    comlrl = config.get("comlrl", {}) or {}
    verl = config.get("verl", {}) or {}
    topology = resolve_comlrl_topology(config, require_worker_assets=True)
    requested_algorithm = str(comlrl.get("algorithm", "magrpo")).strip().lower()
    iterative_algorithm = _ITERATIVE_ALIASES.get(requested_algorithm)
    algorithm = iterative_algorithm or requested_algorithm
    supported = _REINFORCE_ALGORITHMS | _ACTOR_CRITIC_ALGORITHMS | {"madpo", "marlhf", *_ITERATIVE_ALIASES.values()}
    if algorithm not in supported:
        raise ValueError(f"Unsupported CoMLRL algorithm: {requested_algorithm!r}.")
    agent_loop_backend = str(comlrl.get("agent_loop_backend", "hf_local_tq"))
    if agent_loop_backend == "verl_tq":
        raise ValueError("CoMLRL requires synthetic_tq or hf_local_tq.")
    dtype = normalize_hf_local_dtype(comlrl.get("hf_local_dtype", "fp32"))
    marlhf = comlrl.get("marlhf", {}) or {}
    marlhf_rl_algorithm = str(marlhf.get("rl_algorithm", "magrpo")).strip().lower()
    if algorithm in {"marlhf", "marlhf_iter"} and marlhf_rl_algorithm not in (
        _REINFORCE_ALGORITHMS | _ACTOR_CRITIC_ALGORITHMS
    ):
        raise ValueError(f"Unsupported MARLHF RL algorithm: {marlhf_rl_algorithm!r}.")
    if algorithm in {"marlhf", "marlhf_iter"}:
        effective_algorithm = marlhf_rl_algorithm
    elif algorithm == "madpo_iter":
        effective_algorithm = "madpo"
    else:
        effective_algorithm = algorithm
    sequence_kl_coefficient = float(comlrl.get("sequence_kl_coefficient", 0.0))
    reference_kl_enabled = bool(comlrl.get("reference_kl_enabled", False))
    if sequence_kl_coefficient < 0.0:
        raise ValueError("CoMLRL sequence_kl_coefficient must be non-negative.")
    if reference_kl_enabled or sequence_kl_coefficient > 0.0:
        raise ValueError(
            "CoMLRL reference-policy KL is not wired to the current VERL rollout backend; "
            "leave reference_kl_enabled=false and sequence_kl_coefficient=0."
        )
    runtime_algorithm = effective_algorithm if iterative_algorithm is not None else algorithm
    max_turns = int(comlrl.get("max_turns", 1 if algorithm in {"madpo", "marlhf", "madpo_iter", "marlhf_iter"} else 2))
    joint_mode = str(comlrl.get("joint_mode", "aligned")).strip().lower()
    if max_turns < 1:
        raise ValueError("CoMLRL max_turns must be positive.")
    if joint_mode not in {"aligned", "cross"}:
        raise ValueError("CoMLRL joint_mode must be 'aligned' or 'cross'.")
    if algorithm in {"madpo", "marlhf", "madpo_iter", "marlhf_iter"}:
        if max_turns != 1:
            raise ValueError(f"CoMLRL {algorithm.upper()} requires max_turns=1.")
        if joint_mode != "aligned":
            raise ValueError(f"CoMLRL {algorithm.upper()} requires joint_mode='aligned'.")
    source_config = config_path or str(Path.cwd())

    trainer_mode = "trajweave_multi_actor_sync"
    if algorithm in {"madpo", "madpo_iter"}:
        trainer_mode = "trajweave_joint_preference_sync"
    elif effective_algorithm in _ACTOR_CRITIC_ALGORITHMS:
        trainer_mode = "trajweave_multi_actor_critic_sync"
    overrides = replace_override(
        tuple(str(item) for item in verl.get("overrides", [])),
        "trainer.v1.trainer_mode",
        f"trainer.v1.trainer_mode={trainer_mode}",
    )
    overrides = _ensure_per_gpu_micro_batch_default(
        overrides,
        legacy_key="actor_rollout_ref.actor.ppo_micro_batch_size",
        per_gpu_key="actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu",
    )
    overrides = _ensure_per_gpu_micro_batch_default(
        overrides,
        legacy_key="actor_rollout_ref.rollout.log_prob_micro_batch_size",
        per_gpu_key="actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu",
    )
    overrides = replace_override(
        overrides,
        "actor_rollout_ref.rollout.calculate_log_probs",
        "actor_rollout_ref.rollout.calculate_log_probs=false",
    )
    if effective_algorithm in _ACTOR_CRITIC_ALGORITHMS:
        overrides = _ensure_per_gpu_micro_batch_default(
            overrides,
            legacy_key="critic.ppo_micro_batch_size",
            per_gpu_key="critic.ppo_micro_batch_size_per_gpu",
        )
    primary_actor_group = _primary_actor_group(topology)
    overrides = replace_override(
        overrides,
        "actor_rollout_ref.model.path",
        f"actor_rollout_ref.model.path={_hydra_value(primary_actor_group['model_path'])}",
    )
    overrides = replace_override(
        overrides,
        "actor_rollout_ref.model.tokenizer_path",
        f"actor_rollout_ref.model.tokenizer_path={_hydra_value(primary_actor_group['tokenizer_path'])}",
    )
    if algorithm in {"madpo", "madpo_iter"}:
        default_candidates = 80
    elif effective_algorithm in _ACTOR_CRITIC_ALGORITHMS:
        default_candidates = 1
    else:
        default_candidates = 4
    rollout_candidates = int(comlrl.get("num_candidates", default_candidates))
    if effective_algorithm in _ACTOR_CRITIC_ALGORITHMS:
        rollout_candidates = 1
    elif rollout_candidates < 2:
        raise ValueError(f"CoMLRL {algorithm.upper()} requires num_candidates>=2.")
    required = [
        f"+agent.agent_ids={hydra_string_list(topology['agent_ids'])}",
        f"+agent.model_ids={hydra_string_list(topology['model_ids'])}",
        f"+agent.worker_group_ids={hydra_string_list(tuple(topology['worker_groups']))}",
        f"+agent.worker_groups={hydra_worker_group_list(tuple(topology['worker_groups'].items()))}",
        "+trajweave.recipe=comlrl_joint_math",
        f"+trajweave.config={_hydra_value(source_config)}",
        "+trajweave.coordination_protocol=comlrl_full_joint_tree",
        "+trajweave.trajectory_schema=comlrl_joint_v1",
        f"+trajweave.agent_loop_backend={agent_loop_backend}",
        f"+trajweave.hf_local_dtype={dtype}",
        "+trajweave.multi_actor.enabled=true",
        "+trajweave.multi_actor.routing_field=worker_group",
        f"+trajweave.multi_actor.tokenizer_mode={topology['validation']['tokenizer_mode']}",
        "+trajweave.multi_actor.metric_namespace=comlrl",
        f"+trajweave.comlrl.algorithm={runtime_algorithm}",
        f"+trajweave.comlrl.joint_mode={joint_mode}",
        f"+trajweave.comlrl.max_turns={max_turns}",
        f"+trajweave.comlrl.normalize_advantages={_hydra_scalar(comlrl.get('normalize_advantages', True))}",
        f"+trajweave.comlrl.sequence_kl_coefficient={sequence_kl_coefficient}",
        f"actor_rollout_ref.rollout.n={rollout_candidates}",
        f"actor_rollout_ref.rollout.val_kwargs.n={rollout_candidates}",
        f"+actor_rollout_ref.rollout.agent.agent_loop_manager_class={TRAJWEAVE_AGENT_LOOP_MANAGER_FQN}",
    ]
    for field in ("max_joint_actions", "max_tree_nodes"):
        if comlrl.get(field) is not None:
            required.append(f"+trajweave.comlrl.{field}={_hydra_scalar(comlrl[field])}")
    early_stop_threshold = comlrl.get("early_stop_threshold")
    if early_stop_threshold is not None:
        required.append(f"+trajweave.comlrl.early_stop_threshold={float(early_stop_threshold)}")
    if algorithm in _REINFORCE_ALGORITHMS:
        required.extend(
            [
                f"+trajweave.credit_allocator=comlrl_{algorithm}",
                "+trajweave.verl_extensions=[trajweave_comlrl_reinforce]",
                "critic.enable=false",
                "algorithm.adv_estimator=reinforce_plus_plus",
                "algorithm.use_kl_in_reward=false",
                "actor_rollout_ref.actor.policy_loss.loss_mode=gpg",
                "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum",
                "actor_rollout_ref.actor.ppo_epochs=1",
                f"++algorithm.extension_hooks_class={COMLRL_REINFORCE_HOOK_FQN}",
            ]
        )
    elif algorithm in {"madpo", "madpo_iter"}:
        madpo = comlrl.get("madpo", {}) or {}
        beta = float(madpo.get("beta", comlrl.get("beta", 0.1)))
        if beta <= 0.0:
            raise ValueError("CoMLRL MADPO beta must be positive.")
        required.extend(
            [
                "+trajweave.credit_allocator=comlrl_madpo",
                "critic.enable=false",
                "algorithm.use_kl_in_reward=false",
                "actor_rollout_ref.actor.ppo_epochs=1",
                f"+trajweave.comlrl.madpo.beta={beta}",
                (
                    "+trajweave.comlrl.madpo.pair_selection="
                    f"{str(madpo.get('pair_selection', comlrl.get('pair_selection', 'reward_gap')))}"
                ),
                (
                    "+trajweave.comlrl.madpo.pairs_per_sample="
                    f"{int(madpo.get('pairs_per_sample', comlrl.get('preference_pairs_per_sample', 16)))}"
                ),
                (f"+trajweave.comlrl.madpo.random_seed={int(madpo.get('random_seed', comlrl.get('random_seed', 0)))}"),
            ]
        )
    elif algorithm in {"marlhf", "marlhf_iter"}:
        required.extend(
            [
                "+trajweave.credit_allocator=comlrl_marlhf",
                "+trajweave.comlrl.marlhf.enabled=true",
                f"+trajweave.comlrl.marlhf.rl_algorithm={marlhf_rl_algorithm}",
                f"+trajweave.comlrl.marlhf.reward_model_name={_hydra_value(marlhf.get('reward_model_name', ''))}",
                f"+trajweave.comlrl.marlhf.reward_model_device={str(marlhf.get('reward_model_device', 'cpu'))}",
                (
                    "+trajweave.comlrl.marlhf.preference_num_candidates="
                    f"{int(marlhf.get('preference_num_candidates', 80))}"
                ),
                (
                    "+trajweave.comlrl.marlhf.preference_collection_batches="
                    f"{int(marlhf.get('preference_collection_batches', 1))}"
                ),
                (
                    "+trajweave.comlrl.marlhf.reward_freeze_backbone="
                    f"{_hydra_scalar(marlhf.get('reward_freeze_backbone', False))}"
                ),
                f"+trajweave.comlrl.marlhf.reward_learning_rate={float(marlhf.get('reward_learning_rate', 1.0e-5))}",
                f"+trajweave.comlrl.marlhf.reward_torch_dtype={str(marlhf.get('reward_torch_dtype', 'fp32'))}",
                f"+trajweave.comlrl.marlhf.reward_num_train_epochs={int(marlhf.get('reward_num_train_epochs', 1))}",
                f"+trajweave.comlrl.marlhf.reward_train_batch_size={int(marlhf.get('reward_train_batch_size', 1))}",
                "algorithm.use_kl_in_reward=false",
            ]
        )
        for field in ("reward_max_length", "reward_model_checkpoint"):
            if marlhf.get(field) is not None:
                required.append(f"+trajweave.comlrl.marlhf.{field}={_hydra_scalar(marlhf[field])}")
        if effective_algorithm in _ACTOR_CRITIC_ALGORITHMS:
            critic_model = str(marlhf.get("critic_model_name", "")).strip()
            if not critic_model:
                raise ValueError("MARLHF IAC/MAAC requires comlrl.marlhf.critic_model_name.")
            critic_tokenizer = str(marlhf.get("critic_tokenizer_name", critic_model)).strip()
            required.extend(
                [
                    f"+trajweave.comlrl.marlhf.critic_model_name={_hydra_value(critic_model)}",
                    f"+trajweave.comlrl.marlhf.critic_tokenizer_name={_hydra_value(critic_tokenizer)}",
                    f"+trajweave.comlrl.marlhf.critic_type={str(marlhf.get('critic_type', 'v'))}",
                    f"+trajweave.comlrl.marlhf.critic_gpus={int(marlhf.get('critic_gpus', 1))}",
                    f"+trajweave.comlrl.marlhf.critic_max_length={int(marlhf.get('critic_max_length', 2048))}",
                    (
                        "+trajweave.comlrl.marlhf.critic_value_loss_coef="
                        f"{float(marlhf.get('critic_value_loss_coef', 0.6))}"
                    ),
                    (
                        "+trajweave.comlrl.marlhf.critic_learning_rate="
                        f"{float(marlhf.get('critic_learning_rate', 5.0e-6))}"
                    ),
                ]
            )
        else:
            required.extend(
                [
                    "+trajweave.verl_extensions=[trajweave_comlrl_reinforce]",
                    "critic.enable=false",
                    "algorithm.adv_estimator=reinforce_plus_plus",
                    "actor_rollout_ref.actor.policy_loss.loss_mode=gpg",
                    "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum",
                    "actor_rollout_ref.actor.ppo_epochs=1",
                    f"++algorithm.extension_hooks_class={COMLRL_REINFORCE_HOOK_FQN}",
                ]
            )
    else:
        actor_critic = dict(comlrl.get("actor_critic", {}) or {})
        topology_name = "independent" if effective_algorithm == "iac" else "centralized"
        actor_critic.setdefault("topology", topology_name)
        actor_critic.setdefault("critic_type", "v")
        normalize_critic_route_specs(
            actor_critic.get("critic_routes", actor_critic.get("critic_groups", actor_critic.get("critics"))),
            trainable_actor_groups=topology["validation"]["trainable_worker_groups"],
            topology=actor_critic["topology"],
            critic_type=actor_critic["critic_type"],
            max_length=int(actor_critic.get("max_length", 2048)),
        )
        actor_critic.setdefault("value_loss_coef", 0.6)
        critic_learning_rate = float(actor_critic.get("critic_learning_rate", 5.0e-6))
        if critic_learning_rate <= 0.0:
            raise ValueError("CoMLRL actor_critic.critic_learning_rate must be positive.")
        required.extend(
            [
                "critic.enable=true",
                "actor_rollout_ref.actor.policy_loss.loss_mode=gpg",
                "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum",
                "actor_rollout_ref.actor.ppo_epochs=1",
                f"critic.optim.lr={critic_learning_rate}",
                f"+trajweave.comlrl.actor_critic={_hydra_mapping(actor_critic)}",
            ]
        )
    if iterative_algorithm is not None:
        iterative = dict(comlrl.get("iterative", {}) or {})
        iterative.setdefault("enabled", True)
        iterative["algorithm"] = iterative_algorithm
        iterative.setdefault("num_iterations", 6)
        iterative.setdefault("num_train_epochs", 2 if iterative_algorithm == "marlhf_iter" else 1)
        iterative.setdefault("preference_scoring_reward", "task")
        iterative.setdefault("pair_selection", "comparator_reward")
        iterative.setdefault("pairs_per_sample", 4)
        iterative.setdefault("replay", {"mode": "current"})
        iterative.setdefault(
            "comparator",
            {
                "policy": "current",
                "generation_mode": "decentralized",
                "num_candidates": int(comlrl.get("num_candidates", 20)),
                "history_k": 1,
            },
        )
        comparator = iterative.get("comparator", {}) or {}
        if isinstance(comparator, dict):
            if comparator.get("api_key"):
                raise ValueError(
                    "Configure iterative comparator credentials through api_key_env, not YAML/CLI api_key."
                )
            headers = comparator.get("headers", {}) or {}
            if isinstance(headers, dict) and any(
                key.lower() in {"authorization", "proxy-authorization", "x-api-key"} for key in headers
            ):
                raise ValueError("Secret comparator headers must be provided through api_key_env.")
        required.extend(
            [
                "+trajweave.comlrl.iterative.enabled=true",
                f"+trajweave.comlrl.iterative.algorithm={iterative_algorithm}",
                f"+trajweave.comlrl.iterative.config={_hydra_mapping(iterative)}",
            ]
        )
    return overrides + tuple(required)


def _hydra_mapping(value: dict[str, Any]) -> str:
    return _hydra_value(value)


def _ensure_per_gpu_micro_batch_default(
    overrides: Sequence[str],
    *,
    legacy_key: str,
    per_gpu_key: str,
) -> tuple[str, ...]:
    configured = {str(item).split("=", 1)[0].lstrip("+") for item in overrides}
    if configured.intersection({legacy_key, per_gpu_key}):
        return tuple(str(item) for item in overrides)
    return (*tuple(str(item) for item in overrides), f"{per_gpu_key}=1")


def _hydra_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Hydra overrides do not support non-finite CoMLRL values.")
        return str(value)
    if isinstance(value, Mapping):
        fields = []
        for raw_key, item in value.items():
            key = str(raw_key)
            if re.fullmatch(r"[A-Za-z0-9_.-]+", key) is None:
                raise ValueError(f"CoMLRL Hydra mapping key contains unsupported characters: {key!r}.")
            fields.append(f"{key}:{_hydra_value(item)}")
        return "{" + ",".join(fields) + "}"
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return "[" + ",".join(_hydra_value(item) for item in value) + "]"
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _primary_actor_group(topology: dict[str, Any]) -> dict[str, Any]:
    trainable_groups = tuple(topology["validation"].get("trainable_worker_groups", ()))
    if not trainable_groups:
        raise ValueError("CoMLRL requires at least one trainable actor worker group.")
    return topology["worker_groups"][trainable_groups[0]]


def _hydra_scalar(value: Any) -> str:
    return _hydra_value(value)


__all__ = ["build_comlrl_launch_overrides", "resolve_comlrl_topology"]
