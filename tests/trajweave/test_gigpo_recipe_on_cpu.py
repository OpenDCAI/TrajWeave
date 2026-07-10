from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from trajweave.backends.verl.extensions.common.hooks import GiGPOHooks
from trajweave.credit.common import StepGroupBuilder
from trajweave.credit.gigpo import compute_gigpo_scalar_advantages
from trajweave.recipes.gigpo.config import build_gigpo_launch_overrides, resolve_gigpo_settings
from trajweave.recipes.registry import resolve_recipe
from verl import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator


def test_gigpo_registry_and_qwen_config_are_exposed():
    recipe = resolve_recipe("gigpo.math")
    assert recipe.name == "gigpo.solver_verifier_math"
    assert recipe.runtime_recipe == "gigpo_solver_verifier_math"

    config_path = Path("configs/gigpo/solver_verifier_math_qwen05b_2gpu.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings = resolve_gigpo_settings(config)
    overrides = build_gigpo_launch_overrides(config, config_path=str(config_path))
    assert settings["gamma"] == 0.95
    assert "algorithm.adv_estimator=grpo" in overrides
    assert "++algorithm.gigpo.step_advantage_weight=1.0" in overrides
    assert any("GiGPOHooks" in item for item in overrides)
    assert any("trajweave_gigpo_hierarchical_grpo" in item for item in overrides)


def test_step_group_builder_groups_same_anchor_only_within_the_same_task():
    result = StepGroupBuilder().build(
        anchor_observations=["start", "start", "start", "other"],
        rollout_groups=["task-a", "task-a", "task-b", "task-a"],
    )
    assert result.group_ids[0] == result.group_ids[1]
    assert result.group_ids[0] != result.group_ids[2]
    assert result.group_ids[0] != result.group_ids[3]
    assert sorted(result.group_sizes.values()) == [1, 1, 2]


def test_gigpo_combines_episode_and_discounted_step_advantage():
    result = compute_gigpo_scalar_advantages(
        episode_rewards=torch.tensor([1.0, 1.0, 0.0, 0.0]),
        step_rewards=torch.tensor([0.0, 1.0, 0.0, 0.0]),
        rollout_groups=["task", "task", "task", "task"],
        trajectory_ids=["traj-a", "traj-a", "traj-b", "traj-b"],
        turn_ids=[0, 1, 0, 1],
        anchor_observations=["start", "feedback-a", "start", "feedback-b"],
        gamma=0.5,
        step_advantage_weight=1.0,
        mode="mean_norm",
    )
    torch.testing.assert_close(result.step_returns, torch.tensor([0.5, 1.0, 0.0, 0.0]))
    torch.testing.assert_close(result.episode_advantages, torch.tensor([0.5, 0.5, -0.5, -0.5]))
    torch.testing.assert_close(result.step_advantages, torch.tensor([0.25, 0.0, -0.25, 0.0]))
    torch.testing.assert_close(result.advantages, torch.tensor([0.75, 0.5, -0.75, -0.5]))


def test_gigpo_verl_hook_ignores_padding_and_emits_metrics():
    response_mask = torch.tensor([[1, 1], [1, 1], [1, 1], [1, 1], [0, 0]], dtype=torch.int64)
    rm_scores = torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    data = DataProto.from_dict(
        tensors={"response_mask": response_mask, "token_level_rewards": rm_scores},
        non_tensors={
            "uid": np.array(["task", "task", "task", "task", "padding"], dtype=object),
            "agent_id": np.array(["solver"] * 4 + ["__padding__"], dtype=object),
            "traj_uid": np.array(["traj-a", "traj-a", "traj-b", "traj-b", "padding"], dtype=object),
            "turn_id": np.array([0, 1, 0, 1, 0], dtype=object),
            "anchor_obs": np.array(["start", "feedback-a", "start", "feedback-b", "padding"], dtype=object),
            "next_obs": np.array(["feedback-a", "done", "feedback-b", "done", "padding"], dtype=object),
            "step_reward": np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=object),
            "active_mask": np.array([1.0, 1.0, 1.0, 1.0, 0.0], dtype=object),
        },
    )
    hooks = GiGPOHooks()
    result = hooks.compute_advantage(
        data,
        batch_keys=["task_0_0", "task_0_1", "task_1_0", "task_1_1", "task_1_-1"],
        adv_estimator=AdvantageEstimator.GRPO,
        gamma=0.5,
        lam=1.0,
        num_repeat=2,
        norm_adv_by_std_in_grpo=False,
        config={"gigpo": {"mode": "mean_norm", "step_advantage_weight": 1.0}},
        fallback=lambda *_args, **_kwargs: None,
    )
    torch.testing.assert_close(result.batch["advantages"][:4, 0], torch.tensor([0.75, 0.5, -0.75, -0.5]))
    torch.testing.assert_close(result.batch["advantages"][4], torch.zeros(2))
    metrics = hooks.compute_extra_metrics(result, {}, "advantage")
    assert metrics["trajweave/gigpo/step_group_count"] == 3.0
    assert metrics["trajweave/gigpo/nonzero_advantage_ratio"] == 1.0
