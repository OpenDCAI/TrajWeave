from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trajweave.backends.verl.runtime_config import config_get
from trajweave.backends.verl.schema import to_python


class MARFTEmitterMixin:
    """MARFT static-DAG rollout settings and AgentLoop entry points."""

    def _build_marft_math_workflow_outputs(
        self,
        prompt: dict[str, Any],
        *,
        session_id: int = 0,
        validate: bool = False,
    ) -> list[Any]:
        del validate
        from trajweave.backends.verl.workflow_runtime import build_rule_marft_workflow_outputs

        return build_rule_marft_workflow_outputs(self, prompt=prompt, session_id=session_id)

    def _build_hf_marft_math_workflow_outputs(
        self,
        prompt: dict[str, Any],
        *,
        session_id: int = 0,
        validate: bool = False,
    ) -> list[Any]:
        from trajweave.backends.verl.workflow_runtime import build_hf_workflow_outputs

        return build_hf_workflow_outputs(
            self,
            recipe="marft_math_workflow",
            prompt=prompt,
            session_id=session_id,
            validate=validate,
        )

    def _marft_orchestra_config(self) -> Any:
        agent = config_get(self.config, "agent", default={}) or {}
        orchestra = config_get(agent, "orchestra", default={}) or {}
        return config_get(orchestra, "marft", default={}) or {}

    def _marft_role_names(self) -> tuple[str, ...]:
        raw = to_python(config_get(self._marft_orchestra_config(), "role_names", default=None))
        if not raw:
            agent = config_get(self.config, "agent", default={}) or {}
            raw = to_python(config_get(agent, "agent_ids", default=["planner", "solver"]))
        roles = tuple(str(value).strip() for value in raw)
        if len(roles) < 2 or len(roles) != len(set(roles)) or any(not role for role in roles):
            raise ValueError("MARFT runtime requires at least two unique non-empty role names.")
        return roles

    def _marft_model_ids(self, *, role_names: tuple[str, ...]) -> tuple[str, ...]:
        agent = config_get(self.config, "agent", default={}) or {}
        raw = to_python(config_get(agent, "model_ids", default=["shared"] * len(role_names)))
        model_ids = tuple(str(value).strip() for value in raw)
        if len(model_ids) != len(role_names) or any(not model_id for model_id in model_ids):
            raise ValueError("MARFT agent.model_ids must align with role names and be non-empty.")
        return model_ids

    def _marft_role_configs(self) -> dict[str, dict[str, Any]]:
        raw = to_python(config_get(self._marft_orchestra_config(), "role_configs", default={})) or {}
        if not isinstance(raw, Mapping):
            raise ValueError("MARFT runtime role_configs must be a mapping.")
        return {str(role): dict(value or {}) for role, value in raw.items()}

    def _marft_graph_config(self) -> dict[str, Any]:
        raw = to_python(config_get(self._marft_orchestra_config(), "graph_config", default={})) or {}
        if not isinstance(raw, Mapping):
            raise ValueError("MARFT runtime graph_config must be a mapping.")
        return dict(raw)

    def _marft_credit_settings(self) -> dict[str, Any]:
        config = self._marft_orchestra_config()
        raw_per_agent = to_python(config_get(config, "per_agent_reward_fns", default={})) or {}
        if not isinstance(raw_per_agent, Mapping):
            raise ValueError("MARFT runtime per_agent_reward_fns must be a mapping.")
        return {
            "strategy": str(config_get(config, "credit_strategy", default="equal")),
            "discount": float(config_get(config, "credit_discount", default=1.0)),
            "gamma": float(config_get(config, "return_gamma", default=1.0)),
            "step_reward_fn": str(config_get(config, "step_reward_fn", default="") or "").strip(),
            "per_agent_reward_fns": {str(role): str(path).strip() for role, path in raw_per_agent.items()},
        }


__all__ = ["MARFTEmitterMixin"]
