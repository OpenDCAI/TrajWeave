from __future__ import annotations

from typing import Any

from trajweave.backends.verl.runtime_config import config_get
from trajweave.backends.verl.schema import to_python


class C3EmitterMixin:
    """C3 nested prefix-tree rollout settings and entry points."""

    def _build_c3_reasoner_actor_outputs(
        self,
        prompt: dict[str, Any],
        *,
        session_id: int = 0,
        validate: bool = False,
    ) -> list[Any]:
        from trajweave.backends.verl.workflow_runtime import build_synthetic_c3_workflow_outputs

        return build_synthetic_c3_workflow_outputs(
            self,
            prompt=prompt,
            session_id=session_id,
            validate=validate,
        )

    def _build_hf_c3_reasoner_actor_outputs(
        self,
        prompt: dict[str, Any],
        *,
        session_id: int = 0,
        validate: bool = False,
    ) -> list[Any]:
        from trajweave.backends.verl.workflow_runtime import build_hf_workflow_outputs

        return build_hf_workflow_outputs(
            self,
            recipe="c3_reasoner_actor_math",
            prompt=prompt,
            session_id=session_id,
            validate=validate,
        )

    def _c3_config(self) -> Any:
        trajweave = config_get(self.config, "trajweave", default={}) or {}
        return config_get(trajweave, "c3", default={}) or {}

    def _c3_fanout(self) -> tuple[int, int]:
        raw = to_python(config_get(self._c3_config(), "fanout", default=[2, 2]))
        if not isinstance(raw, list | tuple) or len(raw) != 2:
            raise ValueError("trajweave.c3.fanout must contain exactly two values.")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 2 for value in raw):
            raise ValueError("trajweave.c3.fanout values must be integers of at least 2.")
        fanout = tuple(raw)
        return fanout

    def _c3_credit_variant(self) -> str:
        value = str(config_get(self._c3_config(), "credit_variant", default="value_assisted")).strip().lower()
        if value not in {"reward_only", "value_only", "value_assisted"}:
            raise ValueError("trajweave.c3.credit_variant is invalid.")
        return value

    def _c3_model_ids(self) -> tuple[str, str]:
        agent = config_get(self.config, "agent", default={}) or {}
        raw = to_python(config_get(agent, "model_ids", default=["reasoner", "actor"]))
        if not isinstance(raw, list | tuple) or len(raw) != 2:
            raise ValueError("agent.model_ids must contain the Reasoner and Actor worker groups.")
        model_ids = tuple(str(value).strip() for value in raw)
        if len(set(model_ids)) != 2 or any(not value for value in model_ids):
            raise ValueError("C3 requires two distinct non-empty agent.model_ids.")
        return model_ids


__all__ = ["C3EmitterMixin"]
