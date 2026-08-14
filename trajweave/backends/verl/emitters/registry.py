from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EmitterRoute:
    recipe: str
    synthetic_method: str
    hf_local_method: str

    def method_name(self, *, use_hf_local: bool) -> str:
        return self.hf_local_method if use_hf_local else self.synthetic_method


EMITTER_ROUTES: dict[str, EmitterRoute] = {
    "marft_math_workflow": EmitterRoute(
        recipe="marft_math_workflow",
        synthetic_method="_build_marft_math_workflow_outputs",
        hf_local_method="_build_hf_marft_math_workflow_outputs",
    ),
    "c3_reasoner_actor_math": EmitterRoute(
        recipe="c3_reasoner_actor_math",
        synthetic_method="_build_c3_reasoner_actor_outputs",
        hf_local_method="_build_hf_c3_reasoner_actor_outputs",
    ),
    "doctor_mas_math": EmitterRoute(
        recipe="doctor_mas_math",
        synthetic_method="_build_solver_verifier_outputs",
        hf_local_method="_build_hf_solver_verifier_outputs",
    ),
    "doctor_mas_search": EmitterRoute(
        recipe="doctor_mas_search",
        synthetic_method="_build_search_answer_outputs",
        hf_local_method="_build_hf_search_answer_outputs",
    ),
    "maporl_debate_math": EmitterRoute(
        recipe="maporl_debate_math",
        synthetic_method="_build_maporl_debate_math_outputs",
        hf_local_method="_build_hf_maporl_debate_math_outputs",
    ),
    "agentflow_planner_tool": EmitterRoute(
        recipe="agentflow_planner_tool",
        synthetic_method="_build_agentflow_planner_tool_outputs",
        hf_local_method="_build_hf_agentflow_planner_tool_outputs",
    ),
    "matpo_browse": EmitterRoute(
        recipe="matpo_browse",
        synthetic_method="_build_matpo_browse_outputs",
        hf_local_method="_build_hf_matpo_browse_outputs",
    ),
    "mrlx_research_qa": EmitterRoute(
        recipe="mrlx_research_qa",
        synthetic_method="_build_mrlx_research_outputs",
        hf_local_method="_build_hf_mrlx_research_outputs",
    ),
    "gigpo_solver_verifier_math": EmitterRoute(
        recipe="gigpo_solver_verifier_math",
        synthetic_method="_build_gigpo_solver_verifier_outputs",
        hf_local_method="_build_hf_gigpo_solver_verifier_outputs",
    ),
    "atgrpo_solver_verifier_math": EmitterRoute(
        recipe="atgrpo_solver_verifier_math",
        synthetic_method="_build_atgrpo_solver_verifier_outputs",
        hf_local_method="_build_hf_atgrpo_solver_verifier_outputs",
    ),
    "comas_peer_review_math": EmitterRoute(
        recipe="comas_peer_review_math",
        synthetic_method="_build_comas_peer_review_outputs",
        hf_local_method="_build_hf_comas_peer_review_outputs",
    ),
}


def supported_emitter_recipes() -> set[str]:
    return set(EMITTER_ROUTES)


def build_recipe_outputs(
    worker: Any,
    *,
    recipe: str | None,
    use_hf_local: bool,
    prompt: dict[str, Any],
    session_id: int,
    validate: bool = False,
) -> list[Any]:
    route = EMITTER_ROUTES.get(recipe or "doctor_mas_math")
    if route is None:
        known = ", ".join(sorted(EMITTER_ROUTES))
        raise ValueError(f"Unsupported TrajWeave emitter recipe: {recipe!r}. Known recipes: {known}.")
    if use_hf_local:
        from trajweave.backends.verl.workflow_runtime import build_hf_workflow_outputs

        return build_hf_workflow_outputs(
            worker,
            recipe=route.recipe,
            prompt=prompt,
            session_id=session_id,
            validate=validate,
        )
    method = getattr(worker, route.synthetic_method)
    if route.recipe in {"c3_reasoner_actor_math", "marft_math_workflow"}:
        return method(prompt, session_id=session_id, validate=validate)
    return method(prompt, session_id=session_id)
