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
    if len(set(model_ids)) != 1:
        raise ValueError("MAPoRL v1 integration is single-model/shared-policy only; maporl.model_ids must be identical.")

    max_rounds = int(maporl_cfg.get("max_rounds", team_cfg.get("max_turns", 2)))
    consensus_threshold = int(
        protocol_cfg.get("consensus_threshold", maporl_cfg.get("consensus_threshold", len(agent_ids)))
    )
    early_stop = bool(protocol_cfg.get("early_stop", maporl_cfg.get("early_stop", True)))
    correct_turn_bonus = float(maporl_cfg.get("correct_turn_bonus", credit_cfg.get("correct_turn_bonus", 0.25)))
    consensus_bonus = float(maporl_cfg.get("consensus_bonus", credit_cfg.get("consensus_bonus", 0.25)))
    baseline_scope = str(maporl_cfg.get("baseline_scope", credit_cfg.get("baseline_scope", "policy_group")))
    agent_loop_backend = str(maporl_cfg.get("agent_loop_backend", maporl_cfg.get("rollout_backend", "hf_local_tq")))
    if agent_loop_backend == "verl_tq":
        raise ValueError("MAPoRL v1 requires agent_loop_backend to be synthetic_tq or hf_local_tq, not verl_tq.")
    source_config = config_path or str(Path.cwd())

    required = [
        "algorithm.adv_estimator=grpo",
        "++algorithm.group_by_agent_id=true",
        "algorithm.norm_adv_by_std_in_grpo=true",
        f"+agent.agent_ids={_hydra_list(agent_ids)}",
        f"+agent.model_ids={_hydra_list(model_ids)}",
        "+agent.model_sharing=true",
        "+agent.orchestra_type=maporl",
        f"+agent.orchestra.maporl.max_rounds={max_rounds}",
        f"+agent.orchestra.maporl.consensus_threshold={consensus_threshold}",
        f"+agent.orchestra.maporl.early_stop={str(early_stop).lower()}",
        f"+agent.orchestra.maporl.correct_turn_bonus={correct_turn_bonus}",
        f"+agent.orchestra.maporl.consensus_bonus={consensus_bonus}",
        f"+agent.orchestra.maporl.baseline_scope={baseline_scope}",
        "+trajweave.recipe=maporl_debate_math",
        f"+trajweave.config={source_config}",
        "+trajweave.coordination_protocol=debate_consensus",
        "+trajweave.trajectory_schema=multi_agent_turn_v1",
        "+trajweave.credit_allocator=maporl_score_bonus",
        "+trajweave.verl_extensions=[trajweave_maporl_single_model]",
        f"+trajweave.agent_loop_backend={agent_loop_backend}",
        f"+actor_rollout_ref.rollout.agent.agent_loop_manager_class={TRAJWEAVE_AGENT_LOOP_MANAGER_FQN}",
    ]
    return tuple(str(item) for item in verl_cfg.get("overrides", [])) + tuple(required)


def _hydra_list(values: tuple[str, ...]) -> str:
    return "[" + ",".join(_quote(value) for value in values) + "]"


def _quote(value: str) -> str:
    escaped = str(value).replace('"', '\\"')
    return f'"{escaped}"'
