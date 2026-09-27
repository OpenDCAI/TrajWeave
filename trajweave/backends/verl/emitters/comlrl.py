from __future__ import annotations

from typing import Any

from trajweave.backends.verl.runtime_config import config_get
from trajweave.backends.verl.schema import to_python
from verl.experimental.agent_loop.agent_loop import AgentLoopOutput

_REINFORCE_ALGORITHMS = {"magrpo", "mareinforce", "marloo", "maremax"}
_ACTOR_CRITIC_ALGORITHMS = {"iac", "maac"}
_PREFERENCE_ALGORITHMS = {"madpo"}
_COMLRL_ALGORITHMS = _REINFORCE_ALGORITHMS | _ACTOR_CRITIC_ALGORITHMS | _PREFERENCE_ALGORITHMS


class CoMLRLEmitterMixin:
    """CoMLRL joint-math rollout settings and synthetic/HF entry points."""

    def _build_comlrl_joint_math_outputs(
        self,
        prompt: dict[str, Any],
        *,
        session_id: int = 0,
        validate: bool = False,
    ) -> list[AgentLoopOutput]:
        from trajweave.backends.verl.workflow_runtime import build_synthetic_comlrl_workflow_outputs

        return build_synthetic_comlrl_workflow_outputs(
            self,
            prompt=prompt,
            session_id=session_id,
            validate=validate,
        )

    def _build_hf_comlrl_joint_math_outputs(
        self,
        prompt: dict[str, Any],
        *,
        session_id: int = 0,
        validate: bool = False,
    ) -> list[AgentLoopOutput]:
        from trajweave.backends.verl.workflow_runtime import build_hf_workflow_outputs

        return build_hf_workflow_outputs(
            self,
            recipe="comlrl_joint_math",
            prompt=prompt,
            session_id=session_id,
            validate=validate,
        )

    def _comlrl_config(self) -> Any:
        trajweave = config_get(self.config, "trajweave", default={}) or {}
        return config_get(trajweave, "comlrl", default={}) or {}

    def _comlrl_iterative_context(self) -> dict[str, Any]:
        value = getattr(self, "_trajweave_comlrl_iterative_context", None)
        return dict(value) if isinstance(value, dict) else {}

    def _comlrl_algorithm(self) -> str:
        value = str(config_get(self._comlrl_config(), "algorithm", default="magrpo")).strip().lower()
        if value not in _COMLRL_ALGORITHMS:
            raise ValueError(f"trajweave.comlrl.algorithm must be one of {sorted(_COMLRL_ALGORITHMS)}; got {value!r}.")
        return value

    def _comlrl_agent_ids(self) -> list[str]:
        agent_cfg = config_get(self.config, "agent", default={}) or {}
        raw = to_python(config_get(agent_cfg, "agent_ids", default=None))
        values = [str(value).strip() for value in raw] if raw else ["agent_0", "agent_1"]
        if any(not value for value in values) or len(values) != len(set(values)):
            raise ValueError("agent.agent_ids must contain unique non-empty IDs for CoMLRL.")
        return values

    def _comlrl_model_ids(self, *, default_agent_ids: list[str]) -> list[str]:
        agent_cfg = config_get(self.config, "agent", default={}) or {}
        raw = to_python(config_get(agent_cfg, "model_ids", default=None))
        values = (
            [str(value).strip() for value in raw] if raw else [f"policy_{i}" for i in range(len(default_agent_ids))]
        )
        if len(values) != len(default_agent_ids):
            raise ValueError("agent.model_ids must have the same length as agent.agent_ids for CoMLRL.")
        if any(not value for value in values) or len(values) != len(set(values)):
            raise ValueError("agent.model_ids must contain one unique non-empty worker group per CoMLRL agent.")
        return values

    def _comlrl_num_candidates(self, prompt: dict[str, Any]) -> int:
        value = int(to_python(prompt.get("__comlrl_num_candidates__", 1)))
        if value < 1:
            raise ValueError("CoMLRL num_candidates must be positive.")
        if self._comlrl_iterative_context().get("phase") == "preference":
            if value < 2:
                raise ValueError("CoMLRL iterative preference generation requires num_candidates>=2.")
            return value
        algorithm = self._comlrl_algorithm()
        if algorithm in _ACTOR_CRITIC_ALGORITHMS and value != 1:
            raise ValueError(f"CoMLRL {algorithm.upper()} requires num_candidates=1; got {value}.")
        if algorithm in _REINFORCE_ALGORITHMS and value < 2:
            raise ValueError(f"CoMLRL {algorithm.upper()} requires num_candidates>=2 for a non-zero baseline.")
        if algorithm in _PREFERENCE_ALGORITHMS and value < 2:
            raise ValueError(f"CoMLRL {algorithm.upper()} requires num_candidates>=2 for preference pairing.")
        return value

    def _comlrl_joint_mode(self) -> str:
        value = str(config_get(self._comlrl_config(), "joint_mode", default="aligned")).strip().lower()
        aliases = {"align": "aligned", "aligned": "aligned", "cross": "cross", "crossed": "cross"}
        if value not in aliases:
            raise ValueError(f"Unsupported CoMLRL joint_mode: {value!r}.")
        mode = aliases[value]
        algorithm = self._comlrl_algorithm()
        if algorithm in (_ACTOR_CRITIC_ALGORITHMS | _PREFERENCE_ALGORITHMS) and mode != "aligned":
            raise ValueError(f"CoMLRL {algorithm.upper()} requires joint_mode=aligned; got {mode!r}.")
        return mode

    def _comlrl_max_turns(self) -> int:
        config = self._comlrl_config()
        value = int(config_get(config, "max_turns", config_get(config, "num_turns", 2)))
        if value < 1:
            raise ValueError("trajweave.comlrl.max_turns must be positive.")
        algorithm = self._comlrl_algorithm()
        if algorithm in _PREFERENCE_ALGORITHMS and value != 1:
            raise ValueError(f"CoMLRL {algorithm.upper()} requires max_turns=1; got {value}.")
        return value

    def _comlrl_preference_pair_settings(self) -> tuple[str, int, int]:
        config = self._comlrl_config()
        preference = config_get(config, "madpo", default={}) or {}
        selection = (
            str(config_get(preference, "pair_selection", config_get(config, "pair_selection", "reward_gap")))
            .strip()
            .lower()
        )
        if selection not in {"reward_gap", "random", "all"}:
            raise ValueError("CoMLRL MADPO pair_selection must be one of: reward_gap, random, all.")
        raw_limit = config_get(
            preference,
            "pairs_per_sample",
            config_get(
                preference,
                "preference_pairs_per_sample",
                config_get(config, "preference_pairs_per_sample", 16),
            ),
        )
        if raw_limit is None:
            if selection != "all":
                raise ValueError("CoMLRL MADPO pairs_per_sample may be null only for pair_selection='all'.")
            limit = 1
        else:
            limit = int(to_python(raw_limit))
            if limit < 1:
                raise ValueError("CoMLRL MADPO pairs_per_sample must be positive.")
        seed = int(to_python(config_get(preference, "random_seed", config_get(config, "random_seed", 0))))
        return selection, limit, seed

    def _comlrl_marlhf_reward_scorer(self, team: Any) -> Any:
        marlhf = config_get(self._comlrl_config(), "marlhf", default={}) or {}
        iterative_marlhf = self._comlrl_iterative_context().get("marlhf")
        if isinstance(iterative_marlhf, dict):
            marlhf = {**dict(to_python(marlhf)), **iterative_marlhf}
        if not bool(config_get(marlhf, "reward_model_active", False)):
            return None
        checkpoint_path = str(config_get(marlhf, "reward_model_checkpoint", "")).strip()
        model_name = str(config_get(marlhf, "reward_model_name", "")).strip()
        if not checkpoint_path or not model_name:
            raise ValueError("Active MARLHF online reward requires reward_model_checkpoint and reward_model_name.")
        cache_key = (
            checkpoint_path,
            model_name,
            bool(config_get(marlhf, "reward_freeze_backbone", False)),
            str(config_get(marlhf, "reward_model_device", "cpu")),
            str(config_get(marlhf, "reward_torch_dtype", "fp32")),
            int(config_get(marlhf, "reward_max_length", 0) or 0),
            str(config_get(marlhf, "reward_model_version", "")),
        )
        if getattr(self, "_trajweave_marlhf_scorer_key", None) == cache_key:
            return self._trajweave_marlhf_scorer

        from trajweave.backends.verl.workers.scalar_head import RewardModelWorker
        from trajweave.credit.comlrl.marlhf import JointRewardModelScorer

        worker = RewardModelWorker.from_pretrained(
            model_name,
            team,
            checkpoint_path=checkpoint_path,
            freeze_backbone=cache_key[2],
            max_length=cache_key[5] or None,
            device=cache_key[3],
            torch_dtype=cache_key[4],
            frozen_for_evaluation=True,
        )
        scorer = JointRewardModelScorer(worker)
        self._trajweave_marlhf_scorer_key = cache_key
        self._trajweave_marlhf_scorer = scorer
        return scorer

    def _comlrl_optional_positive_int(self, field: str) -> int | None:
        value = to_python(config_get(self._comlrl_config(), field, default=None))
        if value is None:
            return None
        parsed = int(value)
        if parsed < 1:
            raise ValueError(f"trajweave.comlrl.{field} must be positive when configured.")
        return parsed

    def _comlrl_early_stop_threshold(self) -> float | None:
        config = self._comlrl_config()
        value = to_python(
            config_get(
                config,
                "early_stop_threshold",
                config_get(config, "early_termination_threshold", None),
            )
        )
        return None if value is None else float(value)

    def _comlrl_normalize_advantages(self) -> bool:
        config = self._comlrl_config()
        return bool(
            config_get(
                config,
                "normalize_advantages",
                config_get(config, "advantage_normalization", True),
            )
        )

    def _comlrl_sequence_kl_coefficient(self) -> float:
        config = self._comlrl_config()
        value = config_get(
            config,
            "sequence_kl_coefficient",
            config_get(config, "reference_kl_coef", 0.0)
            if bool(config_get(config, "reference_kl_enabled", False))
            else 0.0,
        )
        coefficient = float(value)
        if coefficient < 0.0:
            raise ValueError("trajweave.comlrl.sequence_kl_coefficient must be non-negative.")
        return coefficient


__all__ = [
    "CoMLRLEmitterMixin",
    "_ACTOR_CRITIC_ALGORITHMS",
    "_COMLRL_ALGORITHMS",
    "_PREFERENCE_ALGORITHMS",
    "_REINFORCE_ALGORITHMS",
]
