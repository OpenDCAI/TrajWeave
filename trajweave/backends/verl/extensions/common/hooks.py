from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PPOExtensionHooks:
    name: str = "default"

    def batch_schema_fields(self, stage: str) -> tuple[str, ...]:
        return ()

    def tq_select_fields(
        self,
        stage: str,
        default_fields: tuple[str, ...] | None = None,
        config: Any = None,
    ) -> tuple[str, ...]:
        if default_fields is not None:
            return default_fields
        if stage != "advantage":
            return ()
        return ("uid", "response_mask", "rm_scores", "rollout_log_probs", "old_log_probs", "ref_log_prob", "values")

    def prepare_dataproto(self, data: Any, *, stage: str, config: Any = None) -> Any:
        return data

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

    def batch_schema_fields(self, stage: str) -> tuple[str, ...]:
        return ("agent_id", "traj_uid", "turn_id")

    def tq_select_fields(
        self,
        stage: str,
        default_fields: tuple[str, ...] | None = None,
        config: Any = None,
    ) -> tuple[str, ...]:
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

            for group_id, group_scores in id2score.items():
                if len(group_scores) == 1:
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
        adv_estimator: Any,
        gamma: float = 1.0,
        lam: float = 1.0,
        num_repeat: int = 1,
        norm_adv_by_std_in_grpo: bool = True,
        config: Any = None,
        fallback: Any = None,
        batch_keys: list[str] | None = None,
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
class MAPoRLFullPPOHooks(PPOExtensionHooks):
    name: str = "maporl_full_ppo"

    def batch_schema_fields(self, stage: str) -> tuple[str, ...]:
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
        default_fields: tuple[str, ...] | None = None,
        config: Any = None,
    ) -> tuple[str, ...]:
        fields = list(super().tq_select_fields(stage, default_fields=default_fields, config=config))
        if stage == "advantage":
            fields.extend(self.batch_schema_fields(stage))
        return tuple(dict.fromkeys(fields))

    def compute_advantage(
        self,
        data: Any,
        *,
        adv_estimator: Any,
        gamma: float = 1.0,
        lam: float = 1.0,
        num_repeat: int = 1,
        norm_adv_by_std_in_grpo: bool = True,
        config: Any = None,
        fallback: Any = None,
        batch_keys: list[str] | None = None,
    ) -> Any:
        if fallback is None:
            raise ValueError("MAPoRLFullPPOHooks requires a fallback advantage implementation.")
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
        output: dict[str, Any] = {}
        non_tensors = getattr(data, "non_tensor_batch", {})
        for field in ("raw_score", "correctness", "consensus_reached"):
            values = non_tensors.get(field)
            if values is None:
                continue
            try:
                numeric_values = [float(value) for value in values]
            except (TypeError, ValueError):
                continue
            if numeric_values:
                output[f"trajweave/maporl/{field}/mean"] = sum(numeric_values) / len(numeric_values)
        return output


@dataclass(frozen=True)
class AgentFlowPlannerGRPOHooks(PPOExtensionHooks):
    name: str = "agentflow_planner_grpo"

    def batch_schema_fields(self, stage: str) -> tuple[str, ...]:
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
        fields = list(super().tq_select_fields(stage, default_fields=default_fields, config=config))
        if stage == "advantage":
            fields.extend(self.batch_schema_fields(stage))
        return tuple(dict.fromkeys(fields))

    def compute_advantage(
        self,
        data: Any,
        *,
        adv_estimator: Any,
        gamma: float = 1.0,
        lam: float = 1.0,
        num_repeat: int = 1,
        norm_adv_by_std_in_grpo: bool = True,
        config: Any = None,
        fallback: Any = None,
        batch_keys: list[str] | None = None,
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


@dataclass(frozen=True)
class MARTIMARS2TreeGRPOHooks(PPOExtensionHooks):
    name: str = "marti_mars2_tree_grpo"
    credit_mode: str = "fidelity"
    parent_sibling_gamma: float = 0.3
    sibling_mix: float = 0.5
    path_discount: float = 0.3

    def batch_schema_fields(self, stage: str) -> tuple[str, ...]:
        return (
            "tree_id",
            "prompt_id",
            "node_id",
            "parent_idx",
            "path",
            "raw_score",
            "search_stage",
            "traj_uid",
            "turn_id",
            "verifier_name",
            "verifier_success",
            "verifier_terminal",
            "verifier_feedback",
            "failure_type",
            "verification_mode",
            "verifier_metadata",
        )

    def tq_select_fields(
        self,
        stage: str,
        default_fields: tuple[str, ...] | None = None,
        config: Any = None,
    ) -> tuple[str, ...]:
        fields = list(super().tq_select_fields(stage, default_fields=default_fields, config=config))
        if stage == "advantage":
            fields.extend(self.batch_schema_fields(stage))
        return tuple(dict.fromkeys(fields))

    def process_rewards(self, data: Any) -> Any:
        import torch

        from trajweave.core import SearchNode
        from trajweave.credit import discounted_path_returns, parent_sibling_shaped_rewards

        non_tensors = data.non_tensor_batch
        required = {"tree_id", "node_id", "parent_idx", "path"}
        missing = sorted(required - set(non_tensors))
        if missing:
            raise KeyError(f"MARTI-MARS2 tree credit requires fields: {missing}.")

        tree_rows: dict[str, list[int]] = {}
        for row, tree_id in enumerate(non_tensors["tree_id"]):
            tree_rows.setdefault(str(tree_id), []).append(row)

        rewards = data.batch["token_level_rewards"].sum(dim=-1)
        response_mask = data.batch["response_mask"]

        if self.credit_mode == "fidelity":
            # The MARTI fidelity path uses raw verifier node rewards and lets
            # the GRPO advantage stage normalize by explicit tree identity.
            # Parent/sibling/path shaping is an opt-in experimental variant.
            raw = rewards.detach().cpu().numpy()
            data.non_tensor_batch["parent_sibling_reward"] = raw.copy()
            data.non_tensor_batch["path_return"] = raw.copy()
            return data
        if self.credit_mode != "experimental":
            raise ValueError(f"Unsupported MARTI-MARS2 credit_mode: {self.credit_mode!r}.")

        shaped_rewards = rewards.clone()
        path_returns = rewards.clone()
        for tree_id, rows in tree_rows.items():
            nodes = []
            for row in rows:
                node_id = int(non_tensors["node_id"][row])
                parent_idx = int(non_tensors["parent_idx"][row])
                path = _normalize_tree_path(non_tensors["path"][row], fallback=node_id)
                nodes.append(
                    SearchNode(
                        tree_id=tree_id,
                        prompt_id=str(non_tensors.get("prompt_id", non_tensors["tree_id"])[row]),
                        node_id=node_id,
                        parent_idx=None if parent_idx < 0 else parent_idx,
                        agent_name="generator",
                        role="generator",
                        policy_group="shared",
                        prompt="",
                        action_text="",
                        path=path,
                        reward=float(rewards[row].item()),
                    )
                )
            shaped = parent_sibling_shaped_rewards(
                nodes,
                gamma=self.parent_sibling_gamma,
                sibling_mix=self.sibling_mix,
            )
            returns = discounted_path_returns(nodes, shaped, discount=self.path_discount)
            for row, shaped_reward, path_return in zip(rows, shaped, returns, strict=True):
                shaped_rewards[row] = shaped_reward
                path_returns[row] = path_return

        token_level_rewards = torch.zeros_like(data.batch["token_level_rewards"])
        for row in range(len(data)):
            valid = torch.nonzero(response_mask[row], as_tuple=False).flatten()
            reward_index = int(valid[-1].item()) if valid.numel() else 0
            token_level_rewards[row, reward_index] = path_returns[row]
        data.batch["token_level_rewards"] = token_level_rewards
        data.non_tensor_batch["parent_sibling_reward"] = shaped_rewards.detach().cpu().numpy()
        data.non_tensor_batch["path_return"] = path_returns.detach().cpu().numpy()
        return data

    def compute_advantage(
        self,
        data: Any,
        *,
        adv_estimator: Any,
        gamma: float = 1.0,
        lam: float = 1.0,
        num_repeat: int = 1,
        norm_adv_by_std_in_grpo: bool = True,
        config: Any = None,
        fallback: Any = None,
        batch_keys: list[str] | None = None,
    ) -> Any:
        if fallback is None:
            raise ValueError("MARTIMARS2TreeGRPOHooks requires a fallback advantage implementation.")
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
        output = {}
        for field in ("raw_score", "parent_sibling_reward", "path_return"):
            values = non_tensors.get(field)
            if values is not None and len(values):
                output[f"trajweave/marti_mars2/{field}/mean"] = sum(float(value) for value in values) / len(values)
        return output


def extension_hooks_for_config(config: Any) -> PPOExtensionHooks:
    trajweave = _config_get(config, "trajweave", {}) or {}
    credit_allocator = _config_get(trajweave, "credit_allocator", None)
    recipe = _config_get(trajweave, "recipe", None)
    extensions = _config_get(trajweave, "verl_extensions", None)
    extension_names = _normalize_extensions(extensions)
    if (
        credit_allocator in {"marti_mars2_fidelity_group_grpo", "marti_mars2_tree_path_grpo"}
        or recipe == "marti_mars2_single_mcts"
        or "trajweave_marti_mars2_tree_grpo" in extension_names
    ):
        agent = _config_get(config, "agent", {}) or {}
        orchestra = _config_get(agent, "orchestra", {}) or {}
        marti = _config_get(orchestra, "marti_mars2", {}) or {}
        return MARTIMARS2TreeGRPOHooks(
            credit_mode=str(_config_get(marti, "credit_mode", "fidelity")),
            parent_sibling_gamma=float(_config_get(marti, "parent_sibling_gamma", 0.3)),
            sibling_mix=float(_config_get(marti, "sibling_mix", 0.5)),
            path_discount=float(_config_get(marti, "path_discount", 0.3)),
        )
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


def _normalize_tree_path(value: Any, *, fallback: int) -> tuple[int, ...]:
    if value is None:
        return (fallback,)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        parts = [part for part in value.replace("/", ",").split(",") if part.strip()]
        return tuple(int(part) for part in parts) or (fallback,)
    try:
        output = tuple(int(part) for part in value)
    except TypeError:
        output = (int(value),)
    return output or (fallback,)


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    try:
        return config.get(key, default)
    except (AttributeError, TypeError):
        return getattr(config, key, default)
