from __future__ import annotations

from typing import Any

from trajweave.backends.verl.runtime_config import config_get
from trajweave.backends.verl.schema import to_python
from verl.experimental.agent_loop.agent_loop import AgentLoopOutput


class MARSHALSelfPlayEmitterMixin:
    def _build_marshal_tictactoe_outputs(
        self,
        prompt: dict[str, Any],
        *,
        session_id: int = 0,
    ) -> list[AgentLoopOutput]:
        from trajweave.backends.verl.workflow_runtime import build_rule_marshal_selfplay_outputs

        return build_rule_marshal_selfplay_outputs(self, prompt=prompt, session_id=session_id)

    def _build_hf_marshal_tictactoe_outputs(
        self,
        prompt: dict[str, Any],
        *,
        session_id: int = 0,
    ) -> list[AgentLoopOutput]:
        from trajweave.backends.verl.workflow_runtime import build_hf_workflow_outputs

        return build_hf_workflow_outputs(
            self,
            recipe="marshal_tictactoe_selfplay",
            prompt=prompt,
            session_id=session_id,
        )

    def _marshal_orchestra_config(self) -> Any:
        agent = config_get(self.config, "agent", default={}) or {}
        orchestra = config_get(agent, "orchestra", default={}) or {}
        return config_get(orchestra, "marshal", default={}) or {}

    def _marshal_player_ids(self) -> tuple[str, str]:
        configured = to_python(config_get(self._marshal_orchestra_config(), "player_ids", default=[])) or []
        values = tuple(str(value) for value in configured)
        if not values:
            agent = config_get(self.config, "agent", default={}) or {}
            values = tuple(str(value) for value in (to_python(config_get(agent, "agent_ids", default=[])) or []))
        if len(values) != 2 or len(set(values)) != 2:
            raise ValueError("MARSHAL AgentLoop requires exactly two distinct player IDs.")
        return values

    def _marshal_shared_model_id(self) -> str:
        return str(config_get(self._marshal_orchestra_config(), "shared_model_id", default="shared_policy"))

    def _marshal_max_actions(self) -> int:
        return int(config_get(self._marshal_orchestra_config(), "max_actions", default=9))

    def _marshal_format_reward(self) -> float:
        return float(config_get(self._marshal_orchestra_config(), "format_reward", default=0.05))

    def _marshal_allow_bare_actions(self) -> bool:
        return bool(config_get(self._marshal_orchestra_config(), "allow_bare_actions", default=False))
