from __future__ import annotations

from pathlib import Path
from typing import Any

from trajweave.backends.verl.runtime_config import normalize_hf_local_dtype
from trajweave.backends.verl.tokenizer_compat import assert_compatible_tokenizers

TRAJWEAVE_AGENT_LOOP_MANAGER_FQN = "trajweave.backends.verl.agent_loop.TrajWeaveAgentLoopManager"


def resolve_maporl_credit_settings(config: dict[str, Any]) -> dict[str, Any]:
    maporl_cfg = config.get("maporl", {}) or {}
    credit_cfg = config.get("credit", {}) or {}
    return {
        "rule_horizon": str(credit_cfg.get("rule_horizon", maporl_cfg.get("rule_horizon", "discounted_sum"))),
        "rule_agent_share": str(credit_cfg.get("rule_agent_share", maporl_cfg.get("rule_agent_share", "all"))),
        "rule_discount": float(credit_cfg.get("rule_discount", maporl_cfg.get("rule_discount", 0.3))),
        "alpha": tuple(
            float(value) for value in credit_cfg.get("alpha", maporl_cfg.get("alpha", [0.0, 0.0, 0.0, 0.0]))
        ),
    }


def build_maporl_launch_overrides(
    config: dict[str, Any],
    *,
    config_path: str | None,
) -> tuple[str, ...]:
    maporl_cfg = config.get("maporl", {})
    team_cfg = config.get("team", {})
    protocol_cfg = config.get("protocol", {})
    verl_cfg = config.get("verl", {})
    credit_settings = resolve_maporl_credit_settings(config)

    agent_count = int(maporl_cfg.get("agent_count", team_cfg.get("agent_count", 2)))
    if agent_count < 2:
        raise ValueError("MAPoRL debate requires at least two agents.")
    default_agent_ids = [f"agent_{idx}" for idx in range(agent_count)]
    agent_ids = tuple(str(value) for value in maporl_cfg.get("agent_ids", default_agent_ids))
    if len(agent_ids) != agent_count:
        raise ValueError(
            f"maporl.agent_count={agent_count} does not match {len(agent_ids)} configured maporl.agent_ids."
        )
    if len(set(agent_ids)) != len(agent_ids):
        raise ValueError("maporl.agent_ids must be unique.")
    model_ids = tuple(str(value) for value in maporl_cfg.get("model_ids", ["shared"] * len(agent_ids)))
    if len(agent_ids) != len(model_ids):
        raise ValueError("maporl.model_ids must have the same length as maporl.agent_ids.")
    worker_groups = _normalize_worker_groups(maporl_cfg, model_ids=model_ids)
    trainable_worker_groups = tuple(
        group_id for group_id, group in worker_groups.items() if bool(group.get("trainable", True))
    )
    multi_actor_training = bool(maporl_cfg.get("multi_actor_training", len(trainable_worker_groups) > 1))
    multi_actor_validation = validate_maporl_multi_actor_config(
        maporl_cfg,
        worker_groups=worker_groups,
        model_ids=model_ids,
        multi_actor_training=multi_actor_training,
    )
    max_rounds = int(maporl_cfg.get("max_rounds", team_cfg.get("max_turns", 2)))
    if max_rounds <= 0:
        raise ValueError("MAPoRL max_rounds must be a positive integer.")
    consensus_threshold = int(
        protocol_cfg.get("consensus_threshold", maporl_cfg.get("consensus_threshold", len(agent_ids)))
    )
    if not 1 <= consensus_threshold <= len(agent_ids):
        raise ValueError(
            "MAPoRL consensus_threshold must be between 1 and the number of agents; "
            f"got threshold={consensus_threshold}, agents={len(agent_ids)}."
        )
    early_stop = bool(protocol_cfg.get("early_stop", maporl_cfg.get("early_stop", True)))
    reward_feedback = bool(maporl_cfg.get("reward_feedback", protocol_cfg.get("reward_feedback", False)))
    criteria_percentage = float(
        maporl_cfg.get(
            "criteria_for_consensus_percentage",
            protocol_cfg.get(
                "criteria_for_consensus_percentage", (consensus_threshold - 1e-9) / max(len(agent_ids), 1)
            ),
        )
    )
    criteria_reward = float(
        maporl_cfg.get(
            "criteria_for_consensus_reward_threshold",
            protocol_cfg.get("criteria_for_consensus_reward_threshold", 0.7),
        )
    )
    if not 0.0 <= criteria_percentage <= 1.0:
        raise ValueError("MAPoRL criteria_for_consensus_percentage must be between 0 and 1.")
    if not 0.0 <= criteria_reward <= 1.0:
        raise ValueError("MAPoRL criteria_for_consensus_reward_threshold must be between 0 and 1.")
    policy_separation = bool(maporl_cfg.get("policy_separation", True))
    collaboration_separation = bool(maporl_cfg.get("collaboration_separation", True))
    task_training = bool(maporl_cfg.get("task_training", False))
    rule_horizon = credit_settings["rule_horizon"]
    rule_agent_share = credit_settings["rule_agent_share"]
    rule_discount = credit_settings["rule_discount"]
    alpha = credit_settings["alpha"]
    agent_loop_backend = str(maporl_cfg.get("agent_loop_backend", maporl_cfg.get("rollout_backend", "hf_local_tq")))
    if agent_loop_backend == "verl_tq":
        raise ValueError("MAPoRL full PPO requires agent_loop_backend to be synthetic_tq or hf_local_tq, not verl_tq.")
    hf_local_dtype = normalize_hf_local_dtype(maporl_cfg.get("hf_local_dtype", "fp32"))
    hf_local_model_cache_size = int(maporl_cfg.get("hf_local_model_cache_size", 0))
    if hf_local_model_cache_size < 0:
        raise ValueError("MAPoRL hf_local_model_cache_size must be zero or a positive integer.")
    source_config = config_path or str(Path.cwd())

    base_overrides = tuple(str(item) for item in verl_cfg.get("overrides", []))
    if multi_actor_training:
        base_overrides = _set_override(
            base_overrides,
            "trainer.v1.trainer_mode",
            "trainer.v1.trainer_mode=trajweave_maporl_multi_actor_sync",
        )

    required = [
        "algorithm.adv_estimator=gae",
        "++algorithm.group_by_agent_id=false",
        "++algorithm.extension_hooks_class=trajweave.backends.verl.extensions.common.hooks.MAPoRLFullPPOHooks",
        f"+agent.agent_ids={_hydra_list(agent_ids)}",
        f"+agent.model_ids={_hydra_list(model_ids)}",
        f"+agent.model_sharing={str(len(set(model_ids)) == 1).lower()}",
        f"+agent.worker_group_ids={_hydra_list(tuple(worker_groups))}",
        "+agent.orchestra_type=maporl",
        f"+agent.orchestra.maporl.max_rounds={max_rounds}",
        f"+agent.orchestra.maporl.consensus_threshold={consensus_threshold}",
        f"+agent.orchestra.maporl.early_stop={str(early_stop).lower()}",
        f"+agent.orchestra.maporl.reward_feedback={str(reward_feedback).lower()}",
        f"+agent.orchestra.maporl.criteria_for_consensus_percentage={criteria_percentage}",
        f"+agent.orchestra.maporl.criteria_for_consensus_reward_threshold={criteria_reward}",
        f"+agent.orchestra.maporl.policy_separation={str(policy_separation).lower()}",
        f"+agent.orchestra.maporl.collaboration_separation={str(collaboration_separation).lower()}",
        f"+agent.orchestra.maporl.task_training={str(task_training).lower()}",
        f"+agent.orchestra.maporl.rule_horizon={rule_horizon}",
        f"+agent.orchestra.maporl.rule_agent_share={rule_agent_share}",
        f"+agent.orchestra.maporl.rule_discount={rule_discount}",
        f"+agent.orchestra.maporl.alpha={_hydra_float_list(alpha)}",
        "+trajweave.recipe=maporl_debate_math",
        f"+trajweave.config={source_config}",
        "+trajweave.coordination_protocol=debate_consensus",
        "+trajweave.trajectory_schema=multi_agent_turn_v1",
        "+trajweave.credit_allocator=maporl_ppo_score_rule",
        "+trajweave.verl_extensions=[trajweave_maporl_full_ppo]",
        f"+trajweave.agent_loop_backend={agent_loop_backend}",
        f"+trajweave.hf_local_dtype={hf_local_dtype}",
        f"+trajweave.hf_local_model_cache_size={hf_local_model_cache_size}",
        "+trajweave.turn_padding_multiple=2",
        f"+trajweave.multi_actor.enabled={str(multi_actor_training).lower()}",
        "+trajweave.multi_actor.routing_field=worker_group",
        f"+trajweave.multi_actor.tokenizer_mode={multi_actor_validation['tokenizer_mode']}",
        f"+actor_rollout_ref.rollout.agent.agent_loop_manager_class={TRAJWEAVE_AGENT_LOOP_MANAGER_FQN}",
    ]
    if worker_groups:
        required.append(f"+agent.worker_groups={_hydra_dict_list(tuple(worker_groups.items()))}")
    return base_overrides + tuple(required)


def _hydra_list(values: tuple[str, ...]) -> str:
    return "[" + ",".join(_quote(value) for value in values) + "]"


def _hydra_float_list(values: tuple[float, ...]) -> str:
    return "[" + ",".join(str(value) for value in values) + "]"


def _quote(value: str) -> str:
    escaped = str(value).replace('"', '\\"')
    return f'"{escaped}"'


def _hydra_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int | float):
        return str(value)
    return _quote(str(value))


def _hydra_dict_list(groups: tuple[tuple[str, dict[str, Any]], ...]) -> str:
    items: list[str] = []
    for group_id, group in groups:
        fields = [f"id:{_quote(group_id)}", f"trainable:{str(bool(group.get('trainable', True))).lower()}"]
        for key in ("model_path", "tokenizer_path", "gpus"):
            if key in group:
                fields.append(f"{key}:{_hydra_value(group[key])}")
        items.append("{" + ",".join(fields) + "}")
    return "[" + ",".join(items) + "]"


def _normalize_worker_groups(maporl_cfg: dict[str, Any], *, model_ids: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    raw_groups = maporl_cfg.get("worker_groups", {}) or {}
    if raw_groups and not isinstance(raw_groups, dict):
        raise ValueError("maporl.worker_groups must be a mapping keyed by model id.")

    groups: dict[str, dict[str, Any]] = {}
    for model_id in dict.fromkeys(model_ids):
        raw_group = raw_groups.get(model_id, {}) if isinstance(raw_groups, dict) else {}
        if raw_group is None:
            raw_group = {}
        if not isinstance(raw_group, dict):
            raise ValueError(f"maporl.worker_groups.{model_id} must be a mapping.")
        group = dict(raw_group)
        group.setdefault("trainable", True)
        groups[str(model_id)] = group

    for group_id, raw_group in raw_groups.items():
        if group_id in groups:
            continue
        if not isinstance(raw_group, dict):
            raise ValueError(f"maporl.worker_groups.{group_id} must be a mapping.")
        group = dict(raw_group)
        group.setdefault("trainable", False)
        groups[str(group_id)] = group
    return groups


def _set_override(overrides: tuple[str, ...], key: str, value: str) -> tuple[str, ...]:
    output = []
    replaced = False
    for item in overrides:
        item_key = item.split("=", 1)[0].lstrip("+")
        if item_key == key:
            if not replaced:
                output.append(value)
                replaced = True
            continue
        output.append(item)
    if not replaced:
        output.append(value)
    return tuple(output)


def validate_maporl_multi_actor_config(
    maporl_cfg: dict[str, Any],
    *,
    worker_groups: dict[str, dict[str, Any]],
    model_ids: tuple[str, ...],
    multi_actor_training: bool,
) -> dict[str, Any]:
    """Validate the MAPoRL multi-worker-group contract."""

    tokenizer_mode = str(maporl_cfg.get("tokenizer_mode", "shared"))
    allowed_tokenizer_modes = {"shared", "compatible"}
    if tokenizer_mode not in allowed_tokenizer_modes:
        raise ValueError(
            "MAPoRL multi_actor_training supports tokenizer_mode in "
            f"{sorted(allowed_tokenizer_modes)}; got {tokenizer_mode!r}."
        )
    trainable_worker_groups = tuple(
        group_id for group_id, group in worker_groups.items() if bool(group.get("trainable", True))
    )
    if not multi_actor_training:
        return {
            "status": "disabled",
            "tokenizer_mode": tokenizer_mode,
            "trainable_worker_groups": trainable_worker_groups,
        }

    if len(trainable_worker_groups) < 2:
        raise ValueError("MAPoRL multi_actor_training requires at least two trainable worker groups.")

    missing_groups = [model_id for model_id in dict.fromkeys(model_ids) if model_id not in worker_groups]
    if missing_groups:
        raise ValueError(f"MAPoRL model_ids missing worker_groups entries: {missing_groups}.")

    missing_fields: list[str] = []
    tokenizer_paths: list[str] = []
    for group_id in trainable_worker_groups:
        group = worker_groups[group_id]
        if not group.get("model_path"):
            missing_fields.append(f"{group_id}.model_path")
        if not group.get("tokenizer_path"):
            missing_fields.append(f"{group_id}.tokenizer_path")
        else:
            tokenizer_paths.append(str(group["tokenizer_path"]))
    if missing_fields:
        raise ValueError(
            "MAPoRL multi_actor_training requires explicit model_path and tokenizer_path for every "
            f"trainable worker group; missing: {missing_fields}."
        )

    unique_tokenizer_paths = sorted(set(tokenizer_paths))
    result = {
        "status": "passed",
        "tokenizer_mode": tokenizer_mode,
        "trainable_worker_groups": trainable_worker_groups,
    }
    if tokenizer_mode == "shared":
        if len(unique_tokenizer_paths) != 1:
            raise ValueError(
                "MAPoRL shared-tokenizer multi_actor_training requires identical tokenizer_path across "
                f"trainable worker groups. Got tokenizer paths: {unique_tokenizer_paths}."
            )
        result["shared_tokenizer_path"] = unique_tokenizer_paths[0]
        return result

    fingerprints = assert_compatible_tokenizers(unique_tokenizer_paths)
    if fingerprints:
        result["compatible_tokenizer_digest"] = next(iter(fingerprints.values())).digest
        result["compatible_tokenizer_paths"] = unique_tokenizer_paths
    return result
