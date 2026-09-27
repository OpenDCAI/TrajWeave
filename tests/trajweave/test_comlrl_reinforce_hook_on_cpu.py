from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from trajweave.backends.verl.extensions.comlrl import CoMLRLReinforceHooks


def _data() -> SimpleNamespace:
    return SimpleNamespace(
        batch={
            "response_mask": torch.tensor([[1, 1, 0], [1, 0, 0], [0, 0, 0]]),
            "token_level_rewards": torch.tensor([[0.0, 1.5, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        },
        non_tensor_batch={
            "joint_advantage": [0.25, -0.5, 0.0],
            "effective_projected_joint_return": [1.5, 2.0, 0.0],
            "joint_action_ids": [["a0", "a1"], ["a2"], []],
            "joint_transition_ids": [["t0", "t1"], ["t2"], []],
            "joint_return_components": [[0.5, 1.0], [2.0], []],
        },
        meta_info={},
    )


def _compute(data: SimpleNamespace, marker: str = "grpo", config=None) -> SimpleNamespace:
    return CoMLRLReinforceHooks().compute_advantage(
        data,
        batch_keys=[],
        adv_estimator=marker,
        gamma=1.0,
        lam=1.0,
        num_repeat=1,
        norm_adv_by_std_in_grpo=True,
        config=config or {},
        fallback=lambda *_args, **_kwargs: pytest.fail("native VERL advantage path must not run"),
    )


@pytest.mark.parametrize("marker", ["grpo", "reinforce_plus_plus", "rloo", "remax"])
def test_hook_uses_all_verl_estimators_only_as_dispatch_markers(marker: str):
    output = _compute(_data(), marker)

    assert torch.equal(
        output.batch["advantages"],
        torch.tensor([[0.25, 0.25, 0.0], [-0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )
    assert torch.equal(
        output.batch["returns"],
        torch.tensor([[1.5, 1.5, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )
    assert output.meta_info["comlrl_adv_estimator_dispatch_marker"] == marker
    assert output.meta_info["comlrl_real_row_count"] == 2


def test_hook_excludes_padding_and_requires_finite_real_scalars():
    data = _data()
    data.non_tensor_batch["joint_advantage"][2] = float("nan")
    data.batch["token_level_rewards"][0, 2] = float("nan")
    data.batch["token_level_rewards"][2] = float("nan")
    output = _compute(data)
    assert output.batch["advantages"][2].sum().item() == 0.0

    data = _data()
    data.non_tensor_batch["effective_projected_joint_return"][0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        _compute(data)


def test_hook_validates_real_action_transition_return_component_lengths():
    for field, value in (
        ("joint_transition_ids", [["t0"], ["t2"], []]),
        ("joint_return_components", [[1.5], [2.0], []]),
    ):
        data = _data()
        data.non_tensor_batch[field] = value
        with pytest.raises(ValueError, match="equal action, transition, and return-component lengths"):
            _compute(data)


def test_hook_rejects_reward_contamination_but_allows_explicit_external_path():
    contaminated = _data()
    contaminated.batch["token_level_rewards"][0, 1] = 9.0
    with pytest.raises(ValueError, match="must sum to effective_projected_joint_return"):
        _compute(contaminated)

    allowed = _data()
    allowed.batch["token_level_rewards"][0, 1] = 9.0
    output = _compute(allowed, config={"comlrl": {"token_reward_validation": "external"}})
    assert output.meta_info["comlrl_token_reward_validation"] == "external"
    assert output.batch["returns"][0, 0].item() == pytest.approx(1.5)


def test_hook_requires_explicit_kl_reward_path_and_rejects_unrelated_estimators():
    with pytest.raises(ValueError, match="use_kl_in_reward=true"):
        _compute(_data(), config={"comlrl": {"token_reward_validation": "kl"}})

    output = _compute(
        deepcopy(_data()),
        config={"use_kl_in_reward": True, "comlrl": {"token_reward_validation": "kl"}},
    )
    assert output.meta_info["comlrl_token_reward_validation"] == "kl"

    with pytest.raises(ValueError, match="dispatch marker"):
        _compute(_data(), marker="gae")
