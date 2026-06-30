import numpy as np
import torch
from omegaconf import OmegaConf

from verl.protocol import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator, compute_grpo_outcome_advantage
from verl.trainer.ppo.ray_trainer import compute_advantage
from verl.trainer.ppo.v1.utils import compute_advantage_for_multi_trajectories


def _drmas_batch() -> DataProto:
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


def test_grpo_supports_drmas_agent_wise_grouping():
    data = _drmas_batch()
    global_adv, _ = compute_grpo_outcome_advantage(
        token_level_rewards=data.batch["token_level_rewards"].clone(),
        response_mask=data.batch["response_mask"],
        index=data.non_tensor_batch["uid"],
        traj_index=data.non_tensor_batch["traj_uid"],
        group_by_agent_id=False,
    )
    group_index = np.array(
        [
            f"{uid}_{agent_id}"
            for uid, agent_id in zip(data.non_tensor_batch["uid"], data.non_tensor_batch["agent_id"], strict=True)
        ],
        dtype=object,
    )
    agent_adv, _ = compute_grpo_outcome_advantage(
        token_level_rewards=data.batch["token_level_rewards"].clone(),
        response_mask=data.batch["response_mask"],
        index=group_index,
        traj_index=data.non_tensor_batch["traj_uid"],
        group_by_agent_id=True,
    )

    assert torch.allclose(global_adv.squeeze(), torch.tensor([-0.7071, -0.7071, 0.7071, 0.7071]), atol=1e-4)
    assert torch.allclose(agent_adv[[0, 2]].mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(agent_adv[[1, 3]].mean(), torch.tensor(0.0), atol=1e-6)
    assert agent_adv[2].item() < 0
    assert global_adv[2].item() > 0


def test_compute_advantage_reads_group_by_agent_id_from_config():
    data = _drmas_batch()
    result = compute_advantage(
        data,
        adv_estimator=AdvantageEstimator.GRPO,
        config=OmegaConf.create({"group_by_agent_id": True}),
    )

    assert result.batch["advantages"][2].item() < 0
    assert result.batch["advantages"][3].item() > 0


def test_v1_multi_trajectory_advantage_uses_all_agent_rows_for_drmas():
    data = _drmas_batch()
    result = compute_advantage_for_multi_trajectories(
        data=data,
        batch_keys=["prompt_0_0", "prompt_0_1", "prompt_1_0", "prompt_1_1"],
        adv_estimator=AdvantageEstimator.GRPO,
        config=OmegaConf.create({"group_by_agent_id": True}),
    )

    assert result.batch["advantages"][0].item() > 0
    assert result.batch["advantages"][1].item() < 0
    assert result.batch["advantages"][2].item() < 0
    assert result.batch["advantages"][3].item() > 0
