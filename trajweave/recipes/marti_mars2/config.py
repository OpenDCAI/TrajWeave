from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TRAJWEAVE_AGENT_LOOP_MANAGER_FQN = "trajweave.backends.verl.agent_loop.TrajWeaveAgentLoopManager"


def build_marti_mars2_launch_overrides(
    config: dict[str, Any],
    *,
    config_path: str | None,
) -> tuple[str, ...]:
    mars2_cfg = config.get("marti_mars2", {})
    rollout_cfg = config.get("rollout", {})
    credit_cfg = config.get("credit", {}) or {}
    verl_cfg = config.get("verl", {})

    max_num_nodes = int(mars2_cfg.get("max_num_nodes", rollout_cfg.get("max_num_nodes", 2)))
    if max_num_nodes < 2:
        raise ValueError("MARTI-MARS2 GRPO requires max_num_nodes >= 2.")
    search_mode = str(mars2_cfg.get("search_mode", "mcts"))
    if search_mode not in {"mcts", "vanilla_grpo"}:
        raise ValueError("marti_mars2.search_mode must be mcts or vanilla_grpo.")
    initial_candidates = int(mars2_cfg.get("initial_candidates", min(2, max_num_nodes)))
    if not 1 <= initial_candidates <= max_num_nodes:
        raise ValueError("marti_mars2.initial_candidates must be between 1 and max_num_nodes.")
    if search_mode == "vanilla_grpo" and initial_candidates != max_num_nodes:
        raise ValueError("Vanilla GRPO requires all rollout nodes to be independent initial candidates.")
    agent_loop_backend = str(mars2_cfg.get("agent_loop_backend", "hf_local_tq"))
    if agent_loop_backend == "verl_tq":
        raise ValueError(
            "MARTI-MARS2 training requires synthetic_tq, hf_local_tq, or vllm_marti_tq, not native verl_tq."
        )

    correction_enabled = bool(mars2_cfg.get("enable_vllm_is_correction", False))
    if search_mode == "vanilla_grpo" and correction_enabled:
        raise ValueError("Vanilla GRPO baseline must disable vLLM importance correction.")
    correction_level = str(mars2_cfg.get("vllm_is_level", "token"))
    if correction_level not in {"token", "sequence"}:
        raise ValueError("marti_mars2.vllm_is_level must be token or sequence.")
    correction_threshold = float(mars2_cfg.get("vllm_is_truncated_threshold", 2.0))
    async_updates = bool(mars2_cfg.get("async_updates", False))
    agent_ids = tuple(str(agent_id) for agent_id in mars2_cfg.get("agent_ids", ("generator",)))
    model_ids = tuple(str(model_id) for model_id in mars2_cfg.get("model_ids", ("shared",) * len(agent_ids)))
    if not agent_ids or len(set(agent_ids)) != len(agent_ids):
        raise ValueError("marti_mars2.agent_ids must be a non-empty list of unique agent names.")
    if len(model_ids) != len(agent_ids):
        raise ValueError("marti_mars2.model_ids must have the same length as marti_mars2.agent_ids.")
    model_sharing = str(len(set(model_ids)) == 1).lower()
    multi_actor_training = bool(mars2_cfg.get("multi_actor_training", len(agent_ids) > 1))
    multi_actor_vllm_enabled = bool((mars2_cfg.get("multi_actor_vllm", {}) or {}).get("enabled", False))
    if multi_actor_training and len(agent_ids) < 2:
        raise ValueError("MARTI multi_actor_training requires at least two agents.")
    if multi_actor_training and agent_loop_backend == "vllm_marti_tq" and not multi_actor_vllm_enabled:
        raise ValueError(
            "MARTI multi_actor_training with vllm_marti_tq requires marti_mars2.multi_actor_vllm.enabled=true."
        )
    if multi_actor_vllm_enabled and agent_loop_backend != "vllm_marti_tq":
        raise ValueError("marti_mars2.multi_actor_vllm.enabled requires agent_loop_backend=vllm_marti_tq.")
    worker_groups = _normalize_worker_groups(mars2_cfg, model_ids=model_ids) if multi_actor_training else {}
    base_overrides = tuple(str(item) for item in verl_cfg.get("overrides", []))
    if multi_actor_training:
        base_overrides = _set_override(
            base_overrides,
            "trainer.v1.trainer_mode",
            "trainer.v1.trainer_mode=trajweave_maporl_multi_actor_sync",
        )
    configured_credit_mode = credit_cfg.get("mode")
    if configured_credit_mode is None:
        recipe_name = str(config.get("recipe", ""))
        credit_mode = "experimental" if "tree_credit_experimental" in recipe_name else "fidelity"
    else:
        credit_mode = str(configured_credit_mode)
    if credit_mode not in {"fidelity", "experimental"}:
        raise ValueError("credit.mode must be fidelity or experimental.")
    source_config = config_path or str(Path.cwd())
    if search_mode == "vanilla_grpo":
        credit_allocator = "vanilla_group_grpo"
        coordination_protocol = "independent_group_rollouts"
    else:
        credit_allocator = (
            "marti_mars2_tree_path_grpo" if credit_mode == "experimental" else "marti_mars2_fidelity_group_grpo"
        )
        coordination_protocol = "mcts_selection_expansion_refinement_termination"

    required = [
        "algorithm.adv_estimator=grpo",
        "++algorithm.group_by_agent_id=false",
        f"actor_rollout_ref.rollout.n={max_num_nodes}",
        f"actor_rollout_ref.rollout.val_kwargs.n={max_num_nodes}",
        "+agent.orchestra_type=marti_mars2",
        f"+agent.orchestra.marti_mars2.max_num_nodes={max_num_nodes}",
        f"+agent.orchestra.marti_mars2.initial_candidates={initial_candidates}",
        f"+agent.orchestra.marti_mars2.search_mode={search_mode}",
        f"+agent.orchestra.marti_mars2.parent_sibling_gamma={float(credit_cfg.get('parent_sibling_gamma', 0.3))}",
        f"+agent.orchestra.marti_mars2.sibling_mix={float(credit_cfg.get('sibling_mix', 0.5))}",
        f"+agent.orchestra.marti_mars2.path_discount={float(credit_cfg.get('path_discount', 0.3))}",
        f"+agent.orchestra.marti_mars2.credit_mode={credit_mode}",
        f"+agent.orchestra.marti_mars2.agent_ids={_hydra_list(agent_ids)}",
        f"+agent.orchestra.marti_mars2.model_ids={_hydra_list(model_ids)}",
        f"+agent.agent_ids={_hydra_list(agent_ids)}",
        f"+agent.model_ids={_hydra_list(model_ids)}",
        f"+agent.model_sharing={model_sharing}",
        "+trajweave.recipe=marti_mars2_single_mcts",
        f"+trajweave.config={source_config}",
        f"+trajweave.coordination_protocol={coordination_protocol}",
        "+trajweave.trajectory_schema=tree_trajectory_v1",
        f"+trajweave.credit_allocator={credit_allocator}",
        "+trajweave.verl_extensions=[trajweave_marti_mars2_tree_grpo]",
        f"+trajweave.agent_loop_backend={agent_loop_backend}",
        f"+trajweave.max_policy_lag={int(mars2_cfg.get('max_policy_lag', 1))}",
        f"+trajweave.buffer_min_batch_size={int(mars2_cfg.get('buffer_min_batch_size', 1))}",
        f"+trajweave.async_buffer.enabled={str(async_updates).lower()}",
        f"+trajweave.correction_hook.enabled={str(correction_enabled).lower()}",
        f"+actor_rollout_ref.rollout.agent.agent_loop_manager_class={TRAJWEAVE_AGENT_LOOP_MANAGER_FQN}",
    ]
    fixture_candidates = mars2_cfg.get("fixture_candidates")
    if fixture_candidates is not None:
        required.append(
            f"+agent.orchestra.marti_mars2.fixture_candidates={_hydra_string_list(tuple(fixture_candidates))}"
        )
    reward_range = mars2_cfg.get("dynamic_filter_reward_range")
    if reward_range is not None:
        if not isinstance(reward_range, (list, tuple)) or len(reward_range) != 2 or float(reward_range[0]) >= float(reward_range[1]):
            raise ValueError("marti_mars2.dynamic_filter_reward_range must be [lower, upper]")
        required.append(f"+trajweave.dynamic_filter_reward_range={list(map(float, reward_range))}")
    if multi_actor_training:
        required.extend(
            [
                f"+agent.worker_group_ids={_hydra_list(tuple(worker_groups))}",
                "+trajweave.multi_actor.enabled=true",
                "+trajweave.multi_actor.routing_field=worker_group",
                f"+trajweave.multi_actor.vllm.enabled={str(multi_actor_vllm_enabled).lower()}",
                f"+agent.worker_groups={_hydra_dict_list(tuple(worker_groups.items()))}",
            ]
        )
    if correction_enabled:
        required.extend(
            [
                f"algorithm.rollout_correction.rollout_is={correction_level}",
                f"algorithm.rollout_correction.rollout_is_threshold={correction_threshold}",
                "algorithm.rollout_correction.bypass_mode=false",
                "actor_rollout_ref.rollout.calculate_log_probs=true",
            ]
        )
    return base_overrides + tuple(required)


def _hydra_list(values: tuple[str, ...]) -> str:
    return "[" + ",".join(f'\"{value}\"' for value in values) + "]"


def _hydra_string_list(values: tuple[str, ...]) -> str:
    return json.dumps(list(values), ensure_ascii=True, separators=(",", ":"))


def _hydra_dict_list(groups: tuple[tuple[str, dict[str, Any]], ...]) -> str:
    items = []
    for group_id, group in groups:
        fields = [f'id:\"{group_id}\"', f"trainable:{str(bool(group.get('trainable', True))).lower()}"]
        for key in ("model_path", "tokenizer_path", "gpus"):
            if key in group:
                value = group[key]
                rendered = str(value).lower() if isinstance(value, bool) else str(value)
                if not isinstance(value, (int, float, bool)):
                    rendered = f'\"{rendered}\"'
                fields.append(f"{key}:{rendered}")
        items.append("{" + ",".join(fields) + "}")
    return "[" + ",".join(items) + "]"


def _normalize_worker_groups(
    mars2_cfg: dict[str, Any], *, model_ids: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    raw_groups = mars2_cfg.get("worker_groups", {}) or {}
    if not isinstance(raw_groups, dict):
        raise ValueError("marti_mars2.worker_groups must be a mapping keyed by model id.")
    groups: dict[str, dict[str, Any]] = {}
    for model_id in dict.fromkeys(model_ids):
        group = raw_groups.get(model_id, {}) or {}
        if not isinstance(group, dict):
            raise ValueError(f"marti_mars2.worker_groups.{model_id} must be a mapping.")
        groups[str(model_id)] = {**group, "trainable": bool(group.get("trainable", True))}
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
