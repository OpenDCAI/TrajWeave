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
    "marti_mars2_single_mcts": EmitterRoute(
        recipe="marti_mars2_single_mcts",
        synthetic_method="_build_marti_mars2_outputs",
        hf_local_method="_build_hf_marti_mars2_outputs",
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
) -> list[Any]:
    route = EMITTER_ROUTES.get(recipe or "doctor_mas_math")
    if route is None:
        known = ", ".join(sorted(EMITTER_ROUTES))
        raise ValueError(f"Unsupported TrajWeave emitter recipe: {recipe!r}. Known recipes: {known}.")
    method = getattr(worker, route.method_name(use_hf_local=use_hf_local))
    return method(prompt, session_id=session_id)
