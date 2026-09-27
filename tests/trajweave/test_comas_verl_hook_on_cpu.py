from __future__ import annotations

import numpy as np
import pytest
import torch

from trajweave.backends.verl.extensions.comas import CoMASInteractionREINFORCEHooks
from verl.protocol import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator


def _comas_batch() -> DataProto:
    response_mask = torch.tensor([[1, 1, 0]] * 6, dtype=torch.long)
    token_level_rewards = torch.zeros(6, 3, dtype=torch.float32)
    intrinsic_rewards = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    token_level_rewards[:, 1] = torch.tensor(intrinsic_rewards)
    non_tensors = {
        "agent_id": np.array(["agent_0", "agent_1", "agent_0", "agent_1", "agent_0", "agent_1"], object),
        "policy_group": np.array(["policy_0", "policy_1", "policy_0", "policy_1", "policy_0", "policy_1"], object),
        "worker_group": np.array(["policy_0", "policy_1", "policy_0", "policy_1", "policy_0", "policy_1"], object),
        "worker_group_model_path": np.array(["/models/qwen"] * 6, object),
        "traj_uid": np.array(["trajectory"] * 6, object),
        "turn_id": np.arange(6, dtype=object),
        "round_id": np.zeros(6, dtype=object),
        "discussion_id": np.array([0, 0, 0, 1, 1, 1], object),
        "interaction_id": np.array(["interaction-0"] * 3 + ["interaction-1"] * 3, object),
        "comas_stage": np.array(["solver", "evaluator", "scorer", "solver", "evaluator", "scorer"], object),
        "solver_agent_id": np.array(["agent_0"] * 3 + ["agent_1"] * 3, object),
        "evaluator_agent_id": np.array(["agent_1"] * 3 + ["agent_0"] * 3, object),
        "scorer_agent_id": np.array(["agent_0"] * 3 + ["agent_1"] * 3, object),
        "generated_score": np.array([3, 3, 3, -1, -1, -1], object),
        "score_valid": np.array([True, True, True, False, False, False], object),
        "normalized_score": np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0], object),
        "intrinsic_reward": np.array(intrinsic_rewards, object),
    }
    return DataProto.from_dict(
        tensors={"response_mask": response_mask, "token_level_rewards": token_level_rewards},
        non_tensors=non_tensors,
    )


def _compute(data: DataProto) -> DataProto:
    return CoMASInteractionREINFORCEHooks().compute_advantage(
        data,
        batch_keys=[f"row-{index}" for index in range(6)],
        adv_estimator=AdvantageEstimator.REINFORCE_PLUS_PLUS,
        gamma=1.0,
        lam=1.0,
        num_repeat=1,
        norm_adv_by_std_in_grpo=True,
        config={"comas": {"normalize_advantages_by_worker_group": True}},
        fallback=None,
    )


def test_comas_hook_builds_reinforce_returns_and_normalizes_each_worker_group():
    result = _compute(_comas_batch())
    mask = result.batch["response_mask"].bool()

    torch.testing.assert_close(
        result.batch["returns"][:, :2],
        torch.tensor([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [-1.0, -1.0]]),
    )
    groups = result.non_tensor_batch["worker_group"].tolist()
    for group_id in {"policy_0", "policy_1"}:
        rows = torch.tensor([value == group_id for value in groups], dtype=torch.bool)
        values = result.batch["advantages"][rows][mask[rows]]
        assert abs(float(values.mean())) < 1e-6
        assert float(values.std(unbiased=False)) == pytest.approx(1.0, abs=1e-6)
    assert torch.isfinite(result.batch["advantages"]).all()


def test_comas_hook_reports_reward_invariants_and_role_metrics():
    hooks = CoMASInteractionREINFORCEHooks()
    result = _compute(_comas_batch())
    metrics = hooks.compute_extra_metrics(result, {}, "advantage")

    assert metrics["trajweave/comas/score_valid_rate"] == 0.5
    assert metrics["trajweave/comas/reward/solver_mean"] == 0.5
    assert metrics["trajweave/comas/reward/evaluator_mean"] == 0.0
    assert metrics["trajweave/comas/reward/scorer_mean"] == -0.5
    assert metrics["trajweave/comas/interaction_count"] == 2.0
    assert metrics["trajweave/comas/reward_invariant_violations"] == 0.0
    assert metrics["trajweave/comas/advantage_finite"] == 1.0


def test_comas_hook_rejects_ground_truth_reward_contamination():
    data = _comas_batch()
    data.batch["token_level_rewards"][0, 1] = 2.0

    with pytest.raises(ValueError, match="contaminated"):
        _compute(data)


def test_comas_hook_declares_complete_transfer_queue_schema():
    hooks = CoMASInteractionREINFORCEHooks()
    fields = hooks.tq_select_fields("advantage", ("uid", "response_mask", "rm_scores"), config={})

    assert "worker_group" in fields
    assert "interaction_id" in fields
    assert "comas_stage" in fields
    assert "intrinsic_reward" in fields
