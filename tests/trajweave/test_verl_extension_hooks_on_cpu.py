import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from trajweave.backends.verl.extensions.common.hooks import (
    AgentFlowPlannerGRPOHooks,
    AgentWiseGRPOHooks,
    MAPoRLFullPPOHooks,
    PPOExtensionHooks,
    extension_hooks_for_config,
)
from trajweave.backends.verl.extensions.marti_mars2 import MARTIMARS2TreeGRPOHooks
from verl.protocol import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator


def _batch() -> DataProto:
    tensors = {
        "token_level_rewards": torch.tensor([[1.0], [0.0], [0.0], [10.0]], dtype=torch.float32),
        "response_mask": torch.ones(4, 1, dtype=torch.long),
    }
    non_tensors = {
        "uid": np.array(["prompt"] * 4, dtype=object),
        "agent_id": np.array(["Solver Agent", "Verifier Agent", "Solver Agent", "Verifier Agent"], dtype=object),
        "traj_uid": np.array(["traj-0", "traj-0", "traj-1", "traj-1"], dtype=object),
    }
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)


def test_agent_wise_hook_declares_stable_batch_contract():
    hooks = AgentWiseGRPOHooks()

    assert hooks.batch_schema_fields("advantage") == ("agent_id", "traj_uid", "turn_id")
    assert hooks.tq_select_fields("advantage", config={"group_by_agent_id": True})[-2:] == ("agent_id", "traj_uid")
    assert "agent_id" not in hooks.tq_select_fields("advantage", config={"group_by_agent_id": False})


def test_agent_wise_hook_builds_agent_advantage_groups():
    groups = AgentWiseGRPOHooks().build_advantage_groups(_batch())

    assert groups.tolist() == [
        "prompt_Solver Agent",
        "prompt_Verifier Agent",
        "prompt_Solver Agent",
        "prompt_Verifier Agent",
    ]


def test_agent_wise_hook_computes_drmas_grpo_advantage():
    data = _batch()
    result = AgentWiseGRPOHooks().compute_advantage(
        data,
        adv_estimator=AdvantageEstimator.GRPO,
        config=OmegaConf.create({"group_by_agent_id": True}),
    )

    assert result.batch["advantages"][0].item() > 0
    assert result.batch["advantages"][1].item() < 0
    assert result.batch["advantages"][2].item() < 0
    assert result.batch["advantages"][3].item() > 0


def test_extension_hooks_for_config_selects_agent_wise_grpo():
    assert isinstance(
        extension_hooks_for_config({"trajweave": {"credit_allocator": "drmas_agent_wise_grpo"}}),
        AgentWiseGRPOHooks,
    )
    assert isinstance(
        extension_hooks_for_config({"trajweave": {"credit_allocator": "maporl_score_bonus"}}),
        AgentWiseGRPOHooks,
    )
    assert isinstance(extension_hooks_for_config({}), PPOExtensionHooks)


def test_extension_hooks_for_config_selects_maporl_full_ppo():
    hooks = extension_hooks_for_config({"trajweave": {"credit_allocator": "maporl_ppo_score_rule"}})

    assert isinstance(hooks, MAPoRLFullPPOHooks)
    fields = hooks.tq_select_fields("advantage", default_fields=("uid", "rm_scores"), config={})
    assert "round_id" in fields
    assert "raw_score" in fields
    assert "policy_group" in fields
    assert "worker_group" in fields


def test_extension_hooks_for_config_selects_agentflow_planner_grpo():
    hooks = extension_hooks_for_config({"trajweave": {"credit_allocator": "agentflow_planner_only_grpo"}})

    assert isinstance(hooks, AgentFlowPlannerGRPOHooks)
    fields = hooks.tq_select_fields("advantage", default_fields=("uid", "rm_scores"), config={})
    assert "agentflow_stage" in fields
    assert "tool_name" in fields
    assert "verifier_decision" in fields
    assert "step_id" in fields


def test_agentflow_hook_delegates_advantage_to_fallback():
    calls = []

    def fallback(data, **kwargs):
        calls.append(kwargs)
        return {"ok": True}

    result = AgentFlowPlannerGRPOHooks().compute_advantage(
        "batch",
        adv_estimator=AdvantageEstimator.GRPO,
        fallback=fallback,
        batch_keys=["responses"],
    )

    assert result == {"ok": True}
    assert calls[0]["batch_keys"] == ["responses"]


def test_marti_mars2_fidelity_hook_keeps_raw_verifier_rewards():
    data = DataProto.from_dict(
        tensors={
            "token_level_rewards": torch.tensor([[1.0], [0.0]], dtype=torch.float32),
            "response_mask": torch.ones(2, 1, dtype=torch.long),
        },
        non_tensors={
            "uid": np.array(["prompt", "prompt"], dtype=object),
            "tree_id": np.array(["tree-0", "tree-0"], dtype=object),
            "prompt_id": np.array(["prompt-0", "prompt-0"], dtype=object),
            "node_id": np.array([0, 1], dtype=object),
            "parent_idx": np.array([-1, -1], dtype=object),
            "path": np.array([(0,), (1,)], dtype=object),
        },
    )

    hooks = MARTIMARS2TreeGRPOHooks()
    result = hooks.process_rewards(data)

    assert result.batch["token_level_rewards"].flatten().tolist() == pytest.approx([1.0, 0.0])
    assert result.non_tensor_batch["parent_sibling_reward"].tolist() == pytest.approx([1.0, 0.0])
    assert result.non_tensor_batch["path_return"].tolist() == pytest.approx([1.0, 0.0])
    assert isinstance(
        extension_hooks_for_config({"trajweave": {"credit_allocator": "marti_mars2_tree_path_grpo"}}),
        MARTIMARS2TreeGRPOHooks,
    )


def test_marti_mars2_experimental_hook_applies_tree_reward_shaping():
    data = DataProto.from_dict(
        tensors={
            "token_level_rewards": torch.tensor([[1.0], [0.0]], dtype=torch.float32),
            "response_mask": torch.ones(2, 1, dtype=torch.long),
        },
        non_tensors={
            "uid": np.array(["prompt", "prompt"], dtype=object),
            "tree_id": np.array(["tree-0", "tree-0"], dtype=object),
            "prompt_id": np.array(["prompt-0", "prompt-0"], dtype=object),
            "node_id": np.array([0, 1], dtype=object),
            "parent_idx": np.array([-1, -1], dtype=object),
            "path": np.array([(0,), (1,)], dtype=object),
        },
    )

    hooks = MARTIMARS2TreeGRPOHooks(credit_mode="experimental")
    result = hooks.process_rewards(data)

    assert result.batch["token_level_rewards"].flatten().tolist() == pytest.approx([1.3, -0.3])
    assert result.non_tensor_batch["parent_sibling_reward"].tolist() == pytest.approx([1.3, -0.3])


def test_marti_mars2_hook_applies_discounted_parent_path_credit():
    data = DataProto.from_dict(
        tensors={
            "token_level_rewards": torch.tensor([[1.0], [0.0], [0.0]], dtype=torch.float32),
            "response_mask": torch.ones(3, 1, dtype=torch.long),
        },
        non_tensors={
            "uid": np.array(["prompt"] * 3, dtype=object),
            "tree_id": np.array(["tree-0"] * 3, dtype=object),
            "prompt_id": np.array(["prompt-0"] * 3, dtype=object),
            "node_id": np.array([0, 1, 2], dtype=object),
            "parent_idx": np.array([-1, -1, 0], dtype=object),
            "path": np.array([(0,), (1,), (0, 2)], dtype=object),
        },
    )

    result = MARTIMARS2TreeGRPOHooks(credit_mode="experimental").process_rewards(data)
    metrics = {}
    MARTIMARS2TreeGRPOHooks(credit_mode="experimental").update_metrics(result, metrics, stage="advantage")

    assert result.non_tensor_batch["parent_sibling_reward"].tolist() == pytest.approx([1.3, -0.3, -0.3])
    assert result.non_tensor_batch["path_return"].tolist() == pytest.approx([1.3, -0.3, 0.09])
    assert metrics["trajweave/marti_mars2/parent_sibling_reward/mean"] == pytest.approx(0.2333333333)
    assert metrics["trajweave/marti_mars2/path_return/mean"] == pytest.approx(0.3633333333)
