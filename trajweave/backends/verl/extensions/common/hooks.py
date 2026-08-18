from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

_DEFAULT_ADVANTAGE_TQ_FIELDS = (
    "uid",
    "response_mask",
    "rm_scores",
    "rollout_log_probs",
    "old_log_probs",
    "ref_log_prob",
    "values",
)


@dataclass(frozen=True)
class PPOExtensionHooks:
    name: str = "default"

    def batch_schema_fields(self, stage: str, config: Any = None) -> tuple[str, ...]:
        return ()

    def tq_select_fields(
        self,
        stage: str,
        default_fields: tuple[str, ...],
        config: Any = None,
    ) -> tuple[str, ...]:
        return default_fields

    def prepare_dataproto(self, data: Any, *, stage: str, config: Any = None) -> Any:
        return data

    def compute_advantage(
        self,
        data: Any,
        *,
        batch_keys: list[str],
        adv_estimator: Any,
        gamma: float,
        lam: float,
        num_repeat: int,
        norm_adv_by_std_in_grpo: bool,
        config: Any = None,
        fallback: Callable[..., Any],
    ) -> Any:
        return fallback(
            data,
            batch_keys=batch_keys,
            adv_estimator=adv_estimator,
            gamma=gamma,
            lam=lam,
            num_repeat=num_repeat,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config,
        )

    def on_rollout_output(self, data: Any) -> Any:
        return data

    def process_rewards(self, data: Any) -> Any:
        return data

    def build_advantage_groups(self, data: Any) -> Any:
        return data.non_tensor_batch["uid"]

    def compute_extra_metrics(self, data: Any, metrics: dict[str, Any], stage: str) -> dict[str, Any]:
        return {}

    def output_fields(
        self,
        stage: str,
        default_fields: tuple[str, ...],
        data: Any,
        config: Any = None,
    ) -> tuple[str, ...]:
        return default_fields

    def update_metrics(self, data: Any, metrics: dict[str, Any], *, stage: str, config: Any = None) -> None:
        metrics.update(self.compute_extra_metrics(data, metrics, stage))


@dataclass(frozen=True)
class AgentWiseGRPOHooks(PPOExtensionHooks):
    name: str = "agent_wise_grpo"

    def batch_schema_fields(self, stage: str, config: Any = None) -> tuple[str, ...]:
        return ("agent_id", "traj_uid", "turn_id")

    def tq_select_fields(
        self,
        stage: str,
        default_fields: tuple[str, ...] | None = None,
        config: Any = None,
    ) -> tuple[str, ...]:
        if default_fields is None:
            default_fields = _DEFAULT_ADVANTAGE_TQ_FIELDS if stage == "advantage" else ()
        fields = list(super().tq_select_fields(stage, default_fields=default_fields, config=config))
        if stage == "advantage" and _config_get(config, "group_by_agent_id", False):
            fields.extend(["agent_id", "traj_uid"])
        return tuple(dict.fromkeys(fields))

    def build_advantage_groups(self, data: Any) -> Any:
        import numpy as np

        if "agent_id" not in data.non_tensor_batch:
            raise KeyError("agent-wise GRPO requires non_tensor_batch['agent_id'].")
        return np.array(
            [
                f"{uid}_{agent_id}"
                for uid, agent_id in zip(data.non_tensor_batch["uid"], data.non_tensor_batch["agent_id"], strict=True)
            ],
            dtype=object,
        )

    def compute_grpo_outcome_advantage(
        self,
        *,
        token_level_rewards: Any,
        response_mask: Any,
        index: Any,
        traj_index: Any | None = None,
        epsilon: float = 1e-6,
        norm_adv_by_std_in_grpo: bool = True,
        group_by_agent_id: bool = False,
        pettingllms_singleton_semantics: bool = False,
    ) -> tuple[Any, Any]:
        import numpy as np
        import torch

        scores = token_level_rewards.sum(dim=-1)
        id2score: dict[Any, list[torch.Tensor]] = defaultdict(list)
        id2mean: dict[Any, torch.Tensor] = {}
        id2std: dict[Any, torch.Tensor] = {}
        traj_accumulator: dict[tuple[Any, Any], list[torch.Tensor]] = defaultdict(list)
        traj2avg: dict[tuple[Any, Any], torch.Tensor] = {}

        with torch.no_grad():
            batch_size = scores.shape[0]
            if traj_index is None:
                traj_index = np.array([str(i) for i in range(batch_size)], dtype=object)

            for row in range(batch_size):
                traj_accumulator[(index[row], traj_index[row])].append(scores[row])

            for key, reward_list in traj_accumulator.items():
                group_id, traj_id = key
                if group_by_agent_id:
                    id2score[group_id].extend(reward_list)
                else:
                    avg_score = torch.stack(reward_list).mean()
                    traj2avg[(group_id, traj_id)] = avg_score
                    id2score[group_id].append(avg_score)

            if not group_by_agent_id:
                for row in range(batch_size):
                    scores[row] = traj2avg[(index[row], traj_index[row])]

            max_group_size = max((len(group_scores) for group_scores in id2score.values()), default=0)
            for group_id, group_scores in id2score.items():
                if len(group_scores) == 1:
                    if pettingllms_singleton_semantics and max_group_size > 1:
                        id2mean[group_id] = group_scores[0]
                        id2std[group_id] = scores.new_tensor(0.0)
                    else:
                        id2mean[group_id] = scores.new_tensor(0.0)
                        id2std[group_id] = scores.new_tensor(1.0)
                elif len(group_scores) > 1:
                    scores_tensor = torch.stack(group_scores)
                    id2mean[group_id] = torch.mean(scores_tensor)
                    id2std[group_id] = torch.std(scores_tensor)
                else:
                    raise ValueError(f"no score in prompt index: {group_id}")

            for row in range(batch_size):
                if norm_adv_by_std_in_grpo:
                    scores[row] = (scores[row] - id2mean[index[row]]) / (id2std[index[row]] + epsilon)
                else:
                    scores[row] = scores[row] - id2mean[index[row]]
            scores = scores.unsqueeze(-1) * response_mask

        return scores, scores

    def compute_advantage(
        self,
        data: Any,
        *,
        batch_keys: list[str] | None = None,
        adv_estimator: Any,
        gamma: float = 1.0,
        lam: float = 1.0,
        num_repeat: int = 1,
        norm_adv_by_std_in_grpo: bool = True,
        config: Any = None,
        fallback: Any = None,
    ) -> Any:
        from verl.trainer.ppo import core_algos
        from verl.trainer.ppo.ray_trainer import compute_response_mask

        if adv_estimator != core_algos.AdvantageEstimator.GRPO:
            if fallback is None:
                raise ValueError(f"{self.name} cannot handle advantage estimator: {adv_estimator}")
            return fallback(
                data,
                batch_keys=batch_keys,
                adv_estimator=adv_estimator,
                gamma=gamma,
                lam=lam,
                num_repeat=num_repeat,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                config=config,
            )

        if "response_mask" not in data.batch.keys():
            data.batch["response_mask"] = compute_response_mask(data)

        group_by_agent_id = bool(_config_get(config, "group_by_agent_id", False))
        group_index = self.build_advantage_groups(data) if group_by_agent_id else data.non_tensor_batch["uid"]
        advantages, returns = self.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=group_index,
            traj_index=data.non_tensor_batch.get("traj_uid"),
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            group_by_agent_id=group_by_agent_id,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        return data


@dataclass(frozen=True)
class ATGRPOHooks(AgentWiseGRPOHooks):
    """Agent- and Turn-wise GRPO (AT-GRPO) advantage grouping.

    Extends :class:`AgentWiseGRPOHooks` by adding the turn dimension to the
    advantage grouping key, so trajectories are normalized within
    ``(rollout_group, turn_id, agent_id)`` buckets instead of only
    ``(rollout_group, agent_id)``. This mirrors PettingLLMs' AT-GRPO, which
    computes baselines jointly across the agent-role and turn dimensions.
    """

    name: str = "atgrpo_agent_turn_wise_grpo"

    def batch_schema_fields(self, stage: str, config: Any = None) -> tuple[str, ...]:
        if stage != "advantage":
            return ()
        return (
            "agent_id",
            "traj_uid",
            "turn_id",
            "root_id",
            "node_id",
            "parent_node_id",
            "observation_group_id",
        )

    def tq_select_fields(
        self,
        stage: str,
        default_fields: tuple[str, ...] | None = None,
        config: Any = None,
    ) -> tuple[str, ...]:
        if default_fields is None:
            default_fields = _DEFAULT_ADVANTAGE_TQ_FIELDS if stage == "advantage" else ()
        fields = list(super().tq_select_fields(stage, default_fields=default_fields, config=config))
        if stage == "advantage":
            fields.extend(self.batch_schema_fields(stage, config=config))
        return tuple(dict.fromkeys(fields))

    def compute_grpo_outcome_advantage(
        self,
        *,
        token_level_rewards: Any,
        response_mask: Any,
        index: Any,
        traj_index: Any | None = None,
        epsilon: float = 1e-6,
        norm_adv_by_std_in_grpo: bool = True,
        group_by_agent_id: bool = False,
    ) -> tuple[Any, Any]:
        return super().compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            traj_index=traj_index,
            epsilon=epsilon,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            group_by_agent_id=group_by_agent_id,
            pettingllms_singleton_semantics=True,
        )

    def build_advantage_groups(self, data: Any) -> Any:
        import numpy as np

        observation_groups = data.non_tensor_batch.get("observation_group_id")
        if observation_groups is not None:
            empty_rows = [row for row, value in enumerate(observation_groups) if not str(value)]
            if empty_rows:
                raise KeyError(f"AT-GRPO tree rows require observation_group_id; missing rows: {empty_rows}.")
            return np.array([str(value) for value in observation_groups], dtype=object)

        missing = [field for field in ("agent_id", "turn_id") if field not in data.non_tensor_batch]
        if missing:
            raise KeyError(f"AT-GRPO requires non_tensor_batch fields: {missing}.")
        return np.array(
            [
                f"legacy:{uid}_{turn_id}_{agent_id}"
                for uid, turn_id, agent_id in zip(
                    data.non_tensor_batch["uid"],
                    data.non_tensor_batch["turn_id"],
                    data.non_tensor_batch["agent_id"],
                    strict=True,
                )
            ],
            dtype=object,
        )


@dataclass(frozen=True)
class GiGPOHooks(PPOExtensionHooks):
    name: str = "gigpo_hierarchical_grpo"

    def batch_schema_fields(self, stage: str, config: Any = None) -> tuple[str, ...]:
        if stage != "advantage":
            return ()
        return ("agent_id", "traj_uid", "turn_id", "anchor_obs", "next_obs", "step_reward", "active_mask")

    def tq_select_fields(
        self,
        stage: str,
        default_fields: tuple[str, ...] | None = None,
        config: Any = None,
    ) -> tuple[str, ...]:
        if default_fields is None:
            default_fields = _DEFAULT_ADVANTAGE_TQ_FIELDS if stage == "advantage" else ()
        fields = list(super().tq_select_fields(stage, default_fields=default_fields, config=config))
        if stage == "advantage":
            fields.extend(self.batch_schema_fields(stage, config=config))
        return tuple(dict.fromkeys(fields))

    def compute_advantage(
        self,
        data: Any,
        *,
        batch_keys: list[str],
        adv_estimator: Any,
        gamma: float,
        lam: float,
        num_repeat: int,
        norm_adv_by_std_in_grpo: bool,
        config: Any = None,
        fallback: Callable[..., Any],
    ) -> Any:
        del batch_keys, lam, num_repeat, norm_adv_by_std_in_grpo, fallback

        import torch

        from trajweave.credit.gigpo import compute_gigpo_scalar_advantages
        from verl.trainer.ppo import core_algos
        from verl.trainer.ppo.ray_trainer import compute_response_mask

        if adv_estimator != core_algos.AdvantageEstimator.GRPO:
            raise ValueError(f"GiGPO requires the critic-free GRPO trainer path, got {adv_estimator!r}.")
        if "response_mask" not in data.batch:
            data.batch["response_mask"] = compute_response_mask(data)
        missing = [field for field in self.batch_schema_fields("advantage") if field not in data.non_tensor_batch]
        if missing:
            raise KeyError(f"GiGPO advantage requires non_tensor_batch fields: {missing}.")

        response_mask = data.batch["response_mask"]
        row_count = response_mask.shape[0]
        non_tensors = data.non_tensor_batch
        raw_active_values = _batch_values(non_tensors["active_mask"], row_count=row_count, field="active_mask")
        active_values = [bool(float(value)) for value in raw_active_values]
        valid_rows = response_mask.bool().any(dim=-1).detach().cpu().tolist()
        active_mask = [bool(active and valid) for active, valid in zip(active_values, valid_rows, strict=True)]
        raw_step_rewards = _batch_values(non_tensors["step_reward"], row_count=row_count, field="step_reward")
        step_rewards = torch.tensor(
            [float(value) for value in raw_step_rewards],
            dtype=torch.float32,
            device=response_mask.device,
        )
        gigpo_config = _config_get(config, "gigpo", {}) or {}
        result = compute_gigpo_scalar_advantages(
            episode_rewards=data.batch["token_level_rewards"].sum(dim=-1),
            step_rewards=step_rewards,
            rollout_groups=_batch_values(non_tensors["uid"], row_count=row_count, field="uid"),
            trajectory_ids=_batch_values(non_tensors["traj_uid"], row_count=row_count, field="traj_uid"),
            turn_ids=[
                int(value) for value in _batch_values(non_tensors["turn_id"], row_count=row_count, field="turn_id")
            ],
            anchor_observations=_batch_values(non_tensors["anchor_obs"], row_count=row_count, field="anchor_obs"),
            gamma=float(gamma),
            step_advantage_weight=float(_config_get(gigpo_config, "step_advantage_weight", 1.0)),
            mode=str(_config_get(gigpo_config, "mode", "mean_std_norm")),
            enable_similarity=bool(_config_get(gigpo_config, "enable_similarity", False)),
            similarity_threshold=float(_config_get(gigpo_config, "similarity_threshold", 0.95)),
            active_mask=active_mask,
        )
        advantages = result.advantages.unsqueeze(-1) * response_mask
        data.batch["advantages"] = advantages
        data.batch["returns"] = advantages
        data.batch["gigpo_episode_advantage"] = result.episode_advantages
        data.batch["gigpo_step_advantage"] = result.step_advantages
        data.batch["gigpo_step_return"] = result.step_returns
        data.non_tensor_batch["gigpo_step_group_uid"] = result.step_group_ids
        data.meta_info["gigpo_step_group_sizes"] = result.step_group_sizes
        data.meta_info["gigpo_active_mask"] = active_mask
        return data

    def compute_extra_metrics(self, data: Any, metrics: dict[str, Any], stage: str) -> dict[str, Any]:
        del metrics
        if stage != "advantage" or "gigpo_step_return" not in data.batch:
            return {}
        import torch

        active_mask = torch.tensor(
            data.meta_info.get("gigpo_active_mask", []),
            dtype=torch.bool,
            device=data.batch["gigpo_step_return"].device,
        )
        if active_mask.numel() == 0 or not bool(active_mask.any()):
            return {}
        group_sizes = data.meta_info.get("gigpo_step_group_sizes", {})
        values = {
            "episode_advantage": data.batch["gigpo_episode_advantage"][active_mask],
            "step_advantage": data.batch["gigpo_step_advantage"][active_mask],
            "step_return": data.batch["gigpo_step_return"][active_mask],
        }
        combined = data.batch["advantages"][active_mask]
        response_mask = data.batch["response_mask"][active_mask].bool()
        first_tokens = response_mask.float().argmax(dim=-1)
        combined_scalars = combined[torch.arange(combined.shape[0], device=combined.device), first_tokens]
        output = {
            "trajweave/gigpo/step_group_count": float(len(group_sizes)),
            "trajweave/gigpo/step_group_size_mean": (
                float(sum(group_sizes.values()) / len(group_sizes)) if group_sizes else 0.0
            ),
            "trajweave/gigpo/advantage_mean": float(combined_scalars.mean()),
            "trajweave/gigpo/advantage_std": float(combined_scalars.std(unbiased=False)),
            "trajweave/gigpo/nonzero_advantage_ratio": float((combined_scalars.abs() > 1e-8).float().mean()),
        }
        for name, tensor in values.items():
            output[f"trajweave/gigpo/{name}_mean"] = float(tensor.mean())
            output[f"trajweave/gigpo/{name}_std"] = float(tensor.std(unbiased=False))
        return output


@dataclass(frozen=True)
class MAPoRLFullPPOHooks(PPOExtensionHooks):
    name: str = "maporl_full_ppo"

    def batch_schema_fields(self, stage: str, config: Any = None) -> tuple[str, ...]:
        return (
            "agent_id",
            "policy_group",
            "worker_group",
            "worker_group_model_path",
            "traj_uid",
            "turn_id",
            "round_id",
            "agent_index",
            "raw_score",
            "correctness",
            "consensus_reached",
            "finished_round",
        )

    def tq_select_fields(
        self,
        stage: str,
        default_fields: tuple[str, ...],
        config: Any = None,
    ) -> tuple[str, ...]:
        fields = list(super().tq_select_fields(stage, default_fields=default_fields, config=config))
        if stage == "advantage":
            fields.extend(self.batch_schema_fields(stage, config=config))
        return tuple(dict.fromkeys(fields))

    def prepare_dataproto(self, data: Any, *, stage: str, config: Any = None) -> Any:
        if stage == "advantage":
            return _apply_maporl_turn_credit(data, config=config)
        return data

    def compute_advantage(
        self,
        data: Any,
        *,
        batch_keys: list[str],
        adv_estimator: Any,
        gamma: float,
        lam: float,
        num_repeat: int,
        norm_adv_by_std_in_grpo: bool,
        config: Any = None,
        fallback: Callable[..., Any],
    ) -> Any:
        data = _apply_maporl_turn_credit(data, config=config)
        return fallback(
            data,
            batch_keys=batch_keys,
            adv_estimator=adv_estimator,
            gamma=gamma,
            lam=lam,
            num_repeat=num_repeat,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config,
        )

    def output_fields(
        self,
        stage: str,
        default_fields: tuple[str, ...],
        data: Any,
        config: Any = None,
    ) -> tuple[str, ...]:
        fields = list(default_fields)
        if stage == "advantage":
            for field in ("token_level_scores", "token_level_rewards"):
                if field in data.batch:
                    fields.append(field)
        return tuple(dict.fromkeys(fields))

    def compute_extra_metrics(self, data: Any, metrics: dict[str, Any], stage: str) -> dict[str, Any]:
        if stage != "advantage":
            return {}
        output: dict[str, Any] = {}
        non_tensors = getattr(data, "non_tensor_batch", {})
        response_mask = getattr(data, "batch", {}).get("response_mask")
        valid_rows = None
        if response_mask is not None and response_mask.ndim == 2:
            valid_rows = response_mask.bool().any(dim=-1).detach().cpu().tolist()
        for field in (
            "raw_score",
            "correctness",
            "consensus_reached",
            "maporl_score",
            "maporl_bonus",
            "maporl_reward",
        ):
            values = non_tensors.get(field)
            if values is None:
                continue
            try:
                numeric_values = [
                    float(value) for row, value in enumerate(values) if valid_rows is None or valid_rows[row]
                ]
            except (TypeError, ValueError):
                continue
            if numeric_values:
                output[f"trajweave/maporl/{field}/mean"] = sum(numeric_values) / len(numeric_values)
        return output


@dataclass(frozen=True)
class AgentFlowPlannerGRPOHooks(PPOExtensionHooks):
    name: str = "agentflow_planner_grpo"

    def batch_schema_fields(self, stage: str, config: Any = None) -> tuple[str, ...]:
        return (
            "agent_id",
            "traj_uid",
            "turn_id",
            "agentflow_stage",
            "tool_name",
            "sub_goal",
            "tool_result",
            "verifier_decision",
            "step_id",
        )

    def tq_select_fields(
        self,
        stage: str,
        default_fields: tuple[str, ...] | None = None,
        config: Any = None,
    ) -> tuple[str, ...]:
        if default_fields is None:
            default_fields = _DEFAULT_ADVANTAGE_TQ_FIELDS if stage == "advantage" else ()
        fields = list(super().tq_select_fields(stage, default_fields=default_fields, config=config))
        if stage == "advantage":
            fields.extend(self.batch_schema_fields(stage, config=config))
        return tuple(dict.fromkeys(fields))

    def compute_advantage(
        self,
        data: Any,
        *,
        batch_keys: list[str] | None = None,
        adv_estimator: Any,
        gamma: float = 1.0,
        lam: float = 1.0,
        num_repeat: int = 1,
        norm_adv_by_std_in_grpo: bool = True,
        config: Any = None,
        fallback: Any = None,
    ) -> Any:
        if fallback is None:
            raise ValueError("AgentFlowPlannerGRPOHooks requires a fallback advantage implementation.")
        return fallback(
            data,
            batch_keys=batch_keys,
            adv_estimator=adv_estimator,
            gamma=gamma,
            lam=lam,
            num_repeat=num_repeat,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config,
        )

    def compute_extra_metrics(self, data: Any, metrics: dict[str, Any], stage: str) -> dict[str, Any]:
        if stage != "advantage":
            return {}
        non_tensors = getattr(data, "non_tensor_batch", {})
        decisions = non_tensors.get("verifier_decision")
        stages = non_tensors.get("agentflow_stage")
        output: dict[str, Any] = {}
        if decisions is not None:
            values = [str(value) for value in decisions]
            if values:
                output["trajweave/agentflow/verifier_stop_rate"] = values.count("STOP") / len(values)
        if stages is not None:
            planner_turns = [str(value) for value in stages].count("planner_next_step")
            output["trajweave/agentflow/planner_turns"] = planner_turns
        return output


_MAPORL_REQUIRED_TURN_FIELDS = (
    "traj_uid",
    "round_id",
    "agent_index",
    "raw_score",
    "correctness",
    "finished_round",
)
_MAPORL_CREDIT_APPLIED = "trajweave_maporl_credit_applied"


def _apply_maporl_turn_credit(data: Any, *, config: Any = None) -> Any:
    meta_info = getattr(data, "meta_info", None)
    if meta_info is not None and meta_info.get(_MAPORL_CREDIT_APPLIED, False):
        return data

    import numpy as np
    import torch

    from trajweave.credit.maporl import MAPoRLPPOScoreRuleCreditAssigner

    non_tensors = getattr(data, "non_tensor_batch", {})
    missing = [field for field in _MAPORL_REQUIRED_TURN_FIELDS if field not in non_tensors]
    if missing:
        raise KeyError(f"MAPoRL advantage credit requires non_tensor_batch fields: {missing}.")
    if "response_mask" not in data.batch or "token_level_scores" not in data.batch:
        raise KeyError("MAPoRL advantage credit requires response_mask and token_level_scores tensors.")

    response_mask = data.batch["response_mask"]
    previous_scores = data.batch["token_level_scores"]
    if response_mask.ndim != 2 or previous_scores.shape != response_mask.shape:
        raise ValueError("MAPoRL response_mask and token_level_scores must be aligned 2D tensors.")
    row_count = response_mask.shape[0]
    turn_fields = {
        field: _batch_values(non_tensors[field], row_count=row_count, field=field)
        for field in _MAPORL_REQUIRED_TURN_FIELDS
    }
    valid_rows = response_mask.bool().any(dim=-1).detach().cpu().tolist()
    real_rows = [row for row, is_valid in enumerate(valid_rows) if is_valid]
    if not real_rows:
        raise ValueError("MAPoRL credit calculation requires at least one non-padding response row.")

    grouped_rows: dict[object, list[int]] = defaultdict(list)
    for row in real_rows:
        grouped_rows[turn_fields["traj_uid"][row]].append(row)

    assigner = MAPoRLPPOScoreRuleCreditAssigner.from_config(config)
    row_credits: list[Any | None] = [None] * row_count
    for traj_uid, rows in grouped_rows.items():
        finished_rounds = {int(turn_fields["finished_round"][row]) for row in rows}
        if len(finished_rounds) != 1:
            raise ValueError(f"MAPoRL trajectory {traj_uid!r} has inconsistent finished_round values.")
        credits = assigner.score_turns(
            round_ids=[int(turn_fields["round_id"][row]) for row in rows],
            agent_indices=[int(turn_fields["agent_index"][row]) for row in rows],
            raw_scores=[float(turn_fields["raw_score"][row]) for row in rows],
            correctnesses=[float(turn_fields["correctness"][row]) for row in rows],
            finished_round=finished_rounds.pop(),
        )
        for row, credit in zip(rows, credits, strict=True):
            row_credits[row] = credit

    if any(row_credits[row] is None for row in real_rows):
        raise RuntimeError("MAPoRL credit calculation did not produce a reward for every real response row.")

    token_level_scores = torch.zeros_like(previous_scores)
    maporl_scores = np.zeros(row_count, dtype=np.float32)
    maporl_bonuses = np.zeros(row_count, dtype=np.float32)
    maporl_rewards = np.zeros(row_count, dtype=np.float32)
    for row in real_rows:
        credit = row_credits[row]
        if credit is None:
            raise RuntimeError(f"MAPoRL row {row} is missing its calculated credit.")
        valid_tokens = torch.nonzero(response_mask[row].bool(), as_tuple=False).flatten()
        token_level_scores[row, valid_tokens[-1]] = credit.reward
        maporl_scores[row] = credit.score
        maporl_bonuses[row] = credit.bonus
        maporl_rewards[row] = credit.reward

    previous_rewards = data.batch.get("token_level_rewards")
    reward_adjustment = 0.0 if previous_rewards is None else previous_rewards - previous_scores
    data.batch["token_level_scores"] = token_level_scores
    data.batch["token_level_rewards"] = token_level_scores + reward_adjustment
    if any(not is_valid for is_valid in valid_rows):
        padding_mask = torch.tensor(
            [not is_valid for is_valid in valid_rows],
            dtype=torch.bool,
            device=data.batch["token_level_rewards"].device,
        )
        data.batch["token_level_rewards"][padding_mask] = 0
    data.non_tensor_batch["maporl_score"] = maporl_scores
    data.non_tensor_batch["maporl_bonus"] = maporl_bonuses
    data.non_tensor_batch["maporl_reward"] = maporl_rewards
    if meta_info is None:
        data.meta_info = {}
    data.meta_info[_MAPORL_CREDIT_APPLIED] = True
    return data


def _batch_values(values: Any, *, row_count: int, field: str) -> list[Any]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, list | tuple):
        values = [values]
    output = list(values)
    if len(output) != row_count:
        raise ValueError(f"MAPoRL field {field!r} has {len(output)} rows, expected {row_count}.")
    return output


def extension_hooks_for_config(config: Any) -> PPOExtensionHooks:
    trajweave = _config_get(config, "trajweave", {}) or {}
    credit_allocator = _config_get(trajweave, "credit_allocator", None)
    recipe = _config_get(trajweave, "recipe", None)
    extensions = _config_get(trajweave, "verl_extensions", None)
    extension_names = _normalize_extensions(extensions)
    if credit_allocator == "marft_ctde" or recipe == "marft_math_workflow" or "trajweave_marft_ctde" in extension_names:
        from trajweave.backends.verl.extensions.marft import MARFTPPOHooks

        return MARFTPPOHooks()
    if (
        credit_allocator == "c3_contextual_counterfactual"
        or recipe == "c3_reasoner_actor_math"
        or "trajweave_c3_contextual_counterfactual" in extension_names
    ):
        from trajweave.backends.verl.extensions.c3 import C3ContextualCounterfactualHooks

        return C3ContextualCounterfactualHooks()
    if (
        credit_allocator
        in {
            "comlrl_reinforce",
            "comlrl_magrpo",
            "comlrl_mareinforce",
            "comlrl_maremax",
            "comlrl_marloo",
        }
        or "trajweave_comlrl_reinforce" in extension_names
    ):
        from trajweave.backends.verl.extensions.comlrl import CoMLRLReinforceHooks

        return CoMLRLReinforceHooks()
    if (
        credit_allocator == "comas_interaction_reward"
        or recipe == "comas_peer_review_math"
        or "trajweave_comas_interaction_reinforce" in extension_names
    ):
        from trajweave.backends.verl.extensions.comas import CoMASInteractionREINFORCEHooks

        return CoMASInteractionREINFORCEHooks()
    if (
        credit_allocator == "marshal_turn_level_reinforce"
        or recipe == "marshal_tictactoe_selfplay"
        or "trajweave_marshal_turn_advantage" in extension_names
    ):
        from trajweave.backends.verl.extensions.marshal import MARSHALHooks

        return MARSHALHooks()
    if (
        credit_allocator == "wideseek_r1_multi_agent_grpo"
        or recipe == "wideseek_r1_broad_search"
        or "trajweave_wideseek_r1_grpo" in extension_names
    ):
        from trajweave.backends.verl.extensions.wideseek_r1 import WideSeekR1GRPOHooks

        return WideSeekR1GRPOHooks()
    if (
        credit_allocator == "matpo_parent_broadcast_grpo"
        or recipe == "matpo_browse"
        or "trajweave_matpo_parent_broadcast" in extension_names
    ):
        from trajweave.backends.verl.extensions.matpo import MATPOParentBroadcastHooks

        return MATPOParentBroadcastHooks()
    if (
        credit_allocator == "gigpo_hierarchical_grpo"
        or recipe == "gigpo_solver_verifier_math"
        or "trajweave_gigpo_hierarchical_grpo" in extension_names
    ):
        return GiGPOHooks()
    if (
        credit_allocator == "agentflow_planner_only_grpo"
        or recipe == "agentflow_planner_tool"
        or "trajweave_agentflow_planner_grpo" in extension_names
    ):
        return AgentFlowPlannerGRPOHooks()
    if (
        credit_allocator in {"maporl_ppo_score_rule", "maporl_full_ppo"}
        or recipe in {"maporl_debate_math", "maporl.debate_math.full_verl_tiny"}
        or "trajweave_maporl_full_ppo" in extension_names
    ):
        return MAPoRLFullPPOHooks()
    if (
        credit_allocator == "atgrpo_agent_turn_wise_grpo"
        or recipe == "atgrpo_solver_verifier_math"
        or "trajweave_atgrpo_agent_turn_wise_grpo" in extension_names
    ):
        return ATGRPOHooks()
    if (
        credit_allocator in {"drmas_agent_wise_grpo", "maporl_score_bonus"}
        or recipe in {"doctor_mas_math", "doctor_mas_search", "maporl_debate_math"}
        or "drmas_agent_wise_grpo" in extension_names
        or "trajweave_maporl_single_model" in extension_names
    ):
        return AgentWiseGRPOHooks()
    return PPOExtensionHooks()


def _normalize_extensions(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {part.strip() for part in value.split(",") if part.strip()}
    try:
        return {str(part).strip() for part in value if str(part).strip()}
    except TypeError:
        return set()


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    try:
        return config.get(key, default)
    except (AttributeError, TypeError):
        return getattr(config, key, default)
