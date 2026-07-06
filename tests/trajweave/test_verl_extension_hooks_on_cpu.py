import numpy as np
import torch
from omegaconf import OmegaConf

from trajweave.backends.verl.extensions.hooks import AgentWiseGRPOHooks, PPOExtensionHooks, extension_hooks_for_config
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
