from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from trajweave.backends.verl.extensions.common.hooked_grpo import apply_hooked_grpo_patch
from trajweave.backends.verl.extensions.common.hooks import PPOExtensionHooks, _normalize_tree_path


def apply_marti_mars2_tree_grpo_patch(config: Any = None) -> None:
    """Install the shared hook bridge for MARTI-MARS² tree GRPO."""
    apply_hooked_grpo_patch(config)


@dataclass(frozen=True)
class MARTIMARS2TreeGRPOHooks(PPOExtensionHooks):
    name: str = "marti_mars2_tree_grpo"
    credit_mode: str = "fidelity"
    parent_sibling_gamma: float = 0.3
    sibling_mix: float = 0.5
    path_discount: float = 0.3

    def batch_schema_fields(self, stage: str) -> tuple[str, ...]:
        return (
            "tree_id", "prompt_id", "node_id", "parent_idx", "path", "raw_score",
            "search_stage", "traj_uid", "turn_id", "verifier_name", "verifier_success",
            "verifier_terminal", "verifier_feedback", "failure_type", "verification_mode",
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
        missing = sorted({"tree_id", "node_id", "parent_idx", "path"} - set(non_tensors))
        if missing:
            raise KeyError(f"MARTI-MARS2 tree credit requires fields: {missing}.")
        tree_rows: dict[str, list[int]] = {}
        for row, tree_id in enumerate(non_tensors["tree_id"]):
            tree_rows.setdefault(str(tree_id), []).append(row)
        rewards = data.batch["token_level_rewards"].sum(dim=-1)
        response_mask = data.batch["response_mask"]
        if self.credit_mode == "fidelity":
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
                        path=_normalize_tree_path(non_tensors["path"][row], fallback=node_id),
                        reward=float(rewards[row].item()),
                    )
                )
            shaped = parent_sibling_shaped_rewards(nodes, gamma=self.parent_sibling_gamma, sibling_mix=self.sibling_mix)
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
