from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

RuntimeExtension = Callable[[Any], None]


def apply_verl_runtime_extensions(config: Any) -> tuple[str, ...]:
    """Apply TrajWeave runtime patches around VERL without editing VERL files."""

    names = _extension_names(config)
    applied: list[str] = []
    for name in names:
        extension = _registry()[name]
        extension(config)
        applied.append(name)
    return tuple(applied)


def _extension_names(config: Any) -> tuple[str, ...]:
    trajweave = _get(config, "trajweave", default={}) or {}
    explicit = _get(trajweave, "verl_extensions", default=None)
    if explicit:
        return _normalize_names(explicit)

    credit_allocator = _get(trajweave, "credit_allocator", default=None)
    recipe = _get(trajweave, "recipe", default=None)
    if credit_allocator == "matpo_parent_broadcast_grpo" or recipe == "matpo_browse":
        return ("trajweave_matpo_parent_broadcast",)
    if credit_allocator == "marft_ctde" or recipe == "marft_math_workflow":
        return ("trajweave_marft_ctde",)
    if credit_allocator == "mrlx_mgrpo" or recipe == "mrlx_research_qa":
        return ("trajweave_mrlx_mgrpo",)
    if credit_allocator == "agentflow_planner_only_grpo" or recipe == "agentflow_planner_tool":
        return ("trajweave_agentflow_planner_grpo",)
    if credit_allocator == "gigpo_hierarchical_grpo" or recipe == "gigpo_solver_verifier_math":
        return ("trajweave_gigpo_hierarchical_grpo",)
    if credit_allocator == "atgrpo_agent_turn_wise_grpo" or recipe == "atgrpo_solver_verifier_math":
        return ("trajweave_atgrpo_agent_turn_wise_grpo",)
    if credit_allocator == "c3_contextual_counterfactual" or recipe == "c3_reasoner_actor_math":
        return ("trajweave_c3_contextual_counterfactual",)
    if credit_allocator in {
        "comlrl_reinforce",
        "comlrl_magrpo",
        "comlrl_mareinforce",
        "comlrl_maremax",
        "comlrl_marloo",
    }:
        return ("trajweave_comlrl_reinforce",)
    if credit_allocator == "comas_interaction_reward" or recipe == "comas_peer_review_math":
        return ("trajweave_comas_interaction_reinforce",)
    if credit_allocator in {"maporl_ppo_score_rule", "maporl_full_ppo"} or recipe == "maporl_debate_math":
        return ("trajweave_maporl_full_ppo",)
    if credit_allocator == "maporl_score_bonus":
        return ("trajweave_maporl_single_model",)
    if credit_allocator == "drmas_agent_wise_grpo" or recipe in {"doctor_mas_math", "doctor_mas_search"}:
        return ("drmas_agent_wise_grpo",)
    return ()


def _normalize_names(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        names = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, Iterable):
        names = [str(part).strip() for part in value if str(part).strip()]
    else:
        raise TypeError(f"trajweave.verl_extensions must be a string or sequence, got {type(value)!r}.")

    unknown = sorted(set(names) - set(_registry()))
    if unknown:
        raise ValueError(f"Unknown TrajWeave VERL runtime extension(s): {unknown}.")
    return tuple(dict.fromkeys(names))


def _registry() -> dict[str, RuntimeExtension]:
    from trajweave.backends.verl.extensions.agentflow import apply_agentflow_planner_grpo_patch
    from trajweave.backends.verl.extensions.atgrpo import apply_atgrpo_agent_turn_wise_grpo_patch
    from trajweave.backends.verl.extensions.c3 import apply_c3_contextual_counterfactual_patch
    from trajweave.backends.verl.extensions.comas import apply_comas_interaction_reinforce_patch
    from trajweave.backends.verl.extensions.comlrl import apply_comlrl_reinforce_patch
    from trajweave.backends.verl.extensions.common.nested_compat import apply_tq_nested_compat_patch
    from trajweave.backends.verl.extensions.drmas import apply_drmas_agent_wise_grpo_patch
    from trajweave.backends.verl.extensions.gigpo import apply_gigpo_hierarchical_grpo_patch
    from trajweave.backends.verl.extensions.maporl import apply_maporl_full_ppo_patch, apply_maporl_single_model_patch
    from trajweave.backends.verl.extensions.marft import apply_marft_ctde_patch
    from trajweave.backends.verl.extensions.matpo import apply_matpo_parent_broadcast_patch
    from trajweave.backends.verl.extensions.mrlx import apply_mrlx_mgrpo_patch

    return {
        "drmas_agent_wise_grpo": apply_drmas_agent_wise_grpo_patch,
        "trajweave_agentflow_planner_grpo": apply_agentflow_planner_grpo_patch,
        "trajweave_atgrpo_agent_turn_wise_grpo": apply_atgrpo_agent_turn_wise_grpo_patch,
        "trajweave_comas_interaction_reinforce": apply_comas_interaction_reinforce_patch,
        "trajweave_c3_contextual_counterfactual": apply_c3_contextual_counterfactual_patch,
        "trajweave_comlrl_reinforce": apply_comlrl_reinforce_patch,
        "trajweave_gigpo_hierarchical_grpo": apply_gigpo_hierarchical_grpo_patch,
        "trajweave_matpo_parent_broadcast": apply_matpo_parent_broadcast_patch,
        "trajweave_mrlx_mgrpo": apply_mrlx_mgrpo_patch,
        "trajweave_maporl_full_ppo": apply_maporl_full_ppo_patch,
        "trajweave_maporl_single_model": apply_maporl_single_model_patch,
        "trajweave_marft_ctde": apply_marft_ctde_patch,
        "tq_nested_compat": apply_tq_nested_compat_patch,
    }


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    try:
        return obj.get(key, default)
    except (AttributeError, TypeError):
        return getattr(obj, key, default)
