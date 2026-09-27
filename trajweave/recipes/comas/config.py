from __future__ import annotations

from pathlib import Path
from typing import Any

from trajweave.pipeline.config import hydra_path

from trajweave.backends.verl.multi_actor import (
    hydra_string_list,
    hydra_worker_group_list,
    normalize_worker_groups,
    replace_override,
    validate_multi_actor_worker_groups,
)
from trajweave.backends.verl.runtime_config import normalize_hf_local_dtype

TRAJWEAVE_AGENT_LOOP_MANAGER_FQN = "trajweave.backends.verl.agent_loop.TrajWeaveAgentLoopManager"
COMAS_HOOK_FQN = "trajweave.backends.verl.extensions.comas.interaction_reinforce.CoMASInteractionREINFORCEHooks"


def resolve_comas_topology(
    config: dict[str, Any],
    *,
    require_worker_assets: bool = False,
) -> dict[str, Any]:
    comas_cfg = config.get("comas", {}) or {}
    team_cfg = config.get("team", {}) or {}
    agent_count = int(comas_cfg.get("agent_count", team_cfg.get("agent_count", 2)))
    if agent_count < 2:
        raise ValueError("CoMAS requires at least two agents.")
    default_agent_ids = [f"agent_{index}" for index in range(agent_count)]
    agent_ids = tuple(str(value) for value in comas_cfg.get("agent_ids", default_agent_ids))
    if len(agent_ids) != agent_count or len(set(agent_ids)) != agent_count:
        raise ValueError("comas.agent_ids must contain one unique id per configured agent.")

    shared_agents = bool(comas_cfg.get("shared_agents", False))
    default_model_ids = (
        ["shared"] * agent_count if shared_agents else [f"policy_{index}" for index in range(agent_count)]
    )
    model_ids = tuple(str(value) for value in comas_cfg.get("model_ids", default_model_ids))
    if len(model_ids) != agent_count:
        raise ValueError("comas.model_ids must have the same length as comas.agent_ids.")
    if not shared_agents and len(set(model_ids)) != agent_count:
        raise ValueError("CoMAS shared_agents=false requires one independent model_id per agent.")
    if shared_agents and len(set(model_ids)) != 1:
        raise ValueError("CoMAS shared_agents=true requires every agent to use the same model_id.")

    worker_groups = normalize_worker_groups(
        comas_cfg.get("worker_groups", {}),
        model_ids=model_ids,
        family="CoMAS",
    )
    trainable_groups = tuple(
        group_id for group_id, group in worker_groups.items() if bool(group.get("trainable", True))
    )
    multi_actor_training = bool(comas_cfg.get("multi_actor_training", not shared_agents))
    if require_worker_assets:
        validation = validate_multi_actor_worker_groups(
            comas_cfg,
            worker_groups=worker_groups,
            model_ids=model_ids,
            enabled=multi_actor_training,
            family="CoMAS",
        )
    else:
        validation = {
            "status": "not_required",
            "tokenizer_mode": str(comas_cfg.get("tokenizer_mode", "shared")),
            "trainable_worker_groups": trainable_groups,
        }
    if not shared_agents and not multi_actor_training:
        raise ValueError("CoMAS independent agents require multi_actor_training=true.")
    if set(model_ids) != set(trainable_groups):
        raise ValueError("CoMAS trains every configured agent; model_ids and trainable worker groups must match.")

    return {
        "agent_count": agent_count,
        "agent_ids": agent_ids,
        "model_ids": model_ids,
        "shared_agents": shared_agents,
        "worker_groups": worker_groups,
        "trainable_worker_groups": trainable_groups,
        "multi_actor_training": multi_actor_training,
        "multi_actor_validation": validation,
    }


def build_comas_launch_overrides(
    config: dict[str, Any],
    *,
    config_path: str | None,
) -> tuple[str, ...]:
    comas_cfg = config.get("comas", {}) or {}
    verl_cfg = config.get("verl", {}) or {}
    topology = resolve_comas_topology(config, require_worker_assets=True)
    num_rounds = int(comas_cfg.get("num_rounds", 2))
    num_references = int(comas_cfg.get("num_references", 2))
    if num_rounds <= 0:
        raise ValueError("CoMAS num_rounds must be positive.")
    if num_references < 0:
        raise ValueError("CoMAS num_references must be zero or positive.")
    task_name = str(comas_cfg.get("task_name", "math"))
    if task_name != "math":
        raise ValueError("The current TrajWeave CoMAS training recipe supports task_name=math only.")
    assignment_seed = int(comas_cfg.get("assignment_seed", 0))
    agent_loop_backend = str(comas_cfg.get("agent_loop_backend", "hf_local_tq"))
    if agent_loop_backend == "verl_tq":
        raise ValueError("CoMAS requires synthetic_tq or hf_local_tq to run its peer-review workflow.")
    hf_local_dtype = normalize_hf_local_dtype(comas_cfg.get("hf_local_dtype", "fp32"))
    cache_size = int(comas_cfg.get("hf_local_model_cache_size", 0))
    if cache_size < 0:
        raise ValueError("CoMAS hf_local_model_cache_size must be zero or positive.")
    normalize_advantages = bool(comas_cfg.get("normalize_advantages_by_worker_group", True))
    source_config = config_path or str(Path.cwd())

    overrides = tuple(str(item) for item in verl_cfg.get("overrides", []))
    if topology["multi_actor_training"]:
        overrides = replace_override(
            overrides,
            "trainer.v1.trainer_mode",
            "trainer.v1.trainer_mode=trajweave_multi_actor_sync",
        )

    required = (
        "algorithm.adv_estimator=reinforce_plus_plus",
        "algorithm.gamma=1.0",
        "algorithm.lam=1.0",
        "algorithm.use_kl_in_reward=false",
        f"++algorithm.extension_hooks_class={COMAS_HOOK_FQN}",
        f"++algorithm.comas.normalize_advantages_by_worker_group={str(normalize_advantages).lower()}",
        f"+agent.agent_ids={hydra_string_list(topology['agent_ids'])}",
        f"+agent.model_ids={hydra_string_list(topology['model_ids'])}",
        f"+agent.model_sharing={str(topology['shared_agents']).lower()}",
        f"+agent.worker_group_ids={hydra_string_list(tuple(topology['worker_groups']))}",
        f"+agent.worker_groups={hydra_worker_group_list(tuple(topology['worker_groups'].items()))}",
        "+agent.orchestra_type=comas",
        f"+agent.orchestra.comas.num_rounds={num_rounds}",
        f"+agent.orchestra.comas.num_references={num_references}",
        f"+agent.orchestra.comas.task_name={task_name}",
        f"+agent.orchestra.comas.assignment_seed={assignment_seed}",
        "+trajweave.recipe=comas_peer_review_math",
        f"+trajweave.config={hydra_path(source_config)}",
        "+trajweave.coordination_protocol=comas_peer_review",
        "+trajweave.trajectory_schema=comas_interaction_turn_v1",
        "+trajweave.credit_allocator=comas_interaction_reward",
        "+trajweave.verl_extensions=[trajweave_comas_interaction_reinforce]",
        f"+trajweave.agent_loop_backend={agent_loop_backend}",
        f"+trajweave.hf_local_dtype={hf_local_dtype}",
        f"+trajweave.hf_local_model_cache_size={cache_size}",
        "+trajweave.turn_padding_multiple=2",
        f"+trajweave.multi_actor.enabled={str(topology['multi_actor_training']).lower()}",
        "+trajweave.multi_actor.routing_field=worker_group",
        f"+trajweave.multi_actor.tokenizer_mode={topology['multi_actor_validation']['tokenizer_mode']}",
        "+trajweave.multi_actor.metric_namespace=comas",
        f"+actor_rollout_ref.rollout.agent.agent_loop_manager_class={TRAJWEAVE_AGENT_LOOP_MANAGER_FQN}",
    )
    return overrides + required
