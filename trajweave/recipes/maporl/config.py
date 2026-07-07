from __future__ import annotations

from pathlib import Path
from typing import Any

TRAJWEAVE_AGENT_LOOP_MANAGER_FQN = "trajweave.backends.verl.agent_loop.TrajWeaveAgentLoopManager"


def build_maporl_launch_overrides(
    config: dict[str, Any],
    *,
    config_path: str | None,
) -> tuple[str, ...]:
    maporl_cfg = config.get("maporl", {})
    team_cfg = config.get("team", {})
    protocol_cfg = config.get("protocol", {})
    credit_cfg = config.get("credit", {}) or {}
    verl_cfg = config.get("verl", {})

    agent_count = int(maporl_cfg.get("agent_count", team_cfg.get("agent_count", 2)))
    if agent_count < 2:
        raise ValueError("MAPoRL debate requires at least two agents.")
    agent_ids = tuple(maporl_cfg.get("agent_ids", [f"agent_{idx}" for idx in range(agent_count)]))
    model_ids = tuple(maporl_cfg.get("model_ids", ["shared"] * len(agent_ids)))
    if len(agent_ids) != len(model_ids):
        raise ValueError("maporl.model_ids must have the same length as maporl.agent_ids.")
    worker_groups = _normalize_worker_groups(maporl_cfg, model_ids=model_ids)
    trainable_worker_groups = tuple(
        group_id for group_id, group in worker_groups.items() if bool(group.get("trainable", True))
    )
    multi_actor_training = bool(maporl_cfg.get("multi_actor_training", len(trainable_worker_groups) > 1))
    max_rounds = int(maporl_cfg.get("max_rounds", team_cfg.get("max_turns", 2)))
    consensus_threshold = int(
        protocol_cfg.get("consensus_threshold", maporl_cfg.get("consensus_threshold", len(agent_ids)))
    )
    early_stop = bool(protocol_cfg.get("early_stop", maporl_cfg.get("early_stop", True)))
    reward_feedback = bool(maporl_cfg.get("reward_feedback", protocol_cfg.get("reward_feedback", False)))
    criteria_percentage = float(
        maporl_cfg.get(
            "criteria_for_consensus_percentage",
            protocol_cfg.get("criteria_for_consensus_percentage", (consensus_threshold - 1e-9) / max(len(agent_ids), 1)),
        )
    )
    criteria_reward = float(
        maporl_cfg.get(
            "criteria_for_consensus_reward_threshold",
            protocol_cfg.get("criteria_for_consensus_reward_threshold", 0.7),
        )
    )
    policy_separation = bool(maporl_cfg.get("policy_separation", True))
    collaboration_separation = bool(maporl_cfg.get("collaboration_separation", True))
    task_training = bool(maporl_cfg.get("task_training", False))
    rule_horizon = str(maporl_cfg.get("rule_horizon", credit_cfg.get("rule_horizon", "discounted_sum")))
    rule_agent_share = str(maporl_cfg.get("rule_agent_share", credit_cfg.get("rule_agent_share", "all")))
    rule_discount = float(maporl_cfg.get("rule_discount", credit_cfg.get("rule_discount", 0.3)))
    alpha = tuple(float(value) for value in maporl_cfg.get("alpha", credit_cfg.get("alpha", [0.0, 0.0, 0.0, 0.0])))
    agent_loop_backend = str(maporl_cfg.get("agent_loop_backend", maporl_cfg.get("rollout_backend", "hf_local_tq")))
    if agent_loop_backend == "verl_tq":
        raise ValueError("MAPoRL full PPO requires agent_loop_backend to be synthetic_tq or hf_local_tq, not verl_tq.")
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
        f"+trajweave.multi_actor.enabled={str(multi_actor_training).lower()}",
        "+trajweave.multi_actor.routing_field=worker_group",
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
