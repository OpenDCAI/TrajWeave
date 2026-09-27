from __future__ import annotations

from typing import Any

from trajweave.backends.verl.runtime_config import config_get
from trajweave.backends.verl.schema import to_python
from verl.experimental.agent_loop.agent_loop import AgentLoopOutput


class CoMASEmitterMixin:
    def _build_comas_peer_review_outputs(
        self,
        prompt: dict[str, Any],
        *,
        session_id: int = 0,
    ) -> list[AgentLoopOutput]:
        from trajweave.backends.verl.workflow_runtime import build_rule_comas_workflow_outputs

        return build_rule_comas_workflow_outputs(self, prompt=prompt, session_id=session_id)

    def _comas_agent_ids(self) -> list[str]:
        agent_cfg = config_get(self.config, "agent", default={}) or {}
        values = to_python(config_get(agent_cfg, "agent_ids", default=None))
        return [str(value) for value in values] if values else ["agent_0", "agent_1"]

    def _comas_model_ids(self, *, default_agent_ids: list[str]) -> list[str]:
        agent_cfg = config_get(self.config, "agent", default={}) or {}
        values = to_python(config_get(agent_cfg, "model_ids", default=None))
        if not values:
            return [f"policy_{index}" for index in range(len(default_agent_ids))]
        model_ids = [str(value) for value in values]
        if len(model_ids) != len(default_agent_ids):
            raise ValueError("agent.model_ids must have the same length as agent.agent_ids for CoMAS.")
        return model_ids

    def _comas_num_rounds(self) -> int:
        return int(config_get(self._comas_orchestra_config(), "num_rounds", default=2))

    def _comas_num_references(self) -> int:
        return int(config_get(self._comas_orchestra_config(), "num_references", default=2))

    def _comas_task_name(self) -> str:
        return str(config_get(self._comas_orchestra_config(), "task_name", default="math"))

    def _comas_assignment_seed(self) -> int:
        return int(config_get(self._comas_orchestra_config(), "assignment_seed", default=0))

    def _comas_orchestra_config(self) -> Any:
        agent_cfg = config_get(self.config, "agent", default={}) or {}
        orchestra_cfg = config_get(agent_cfg, "orchestra", default={}) or {}
        return config_get(orchestra_cfg, "comas", default={}) or {}
