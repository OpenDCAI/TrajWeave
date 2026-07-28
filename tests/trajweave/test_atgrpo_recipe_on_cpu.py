from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml

from trajweave.backends.verl.extensions.common.hooks import ATGRPOHooks
from trajweave.credit.atgrpo import ATGRPOCreditAssigner
from trajweave.recipes.atgrpo.config import build_atgrpo_launch_overrides, resolve_atgrpo_settings
from trajweave.recipes.registry import resolve_recipe
from verl import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator


def test_atgrpo_registry_and_qwen_config_are_exposed():
    recipe = resolve_recipe("atgrpo.math")
    assert recipe.name == "atgrpo.solver_verifier_math"
    assert recipe.runtime_recipe == "atgrpo_solver_verifier_math"

    config_path = Path("configs/atgrpo/solver_verifier_math_qwen05b_2gpu.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings = resolve_atgrpo_settings(config)
    overrides = build_atgrpo_launch_overrides(config, config_path=str(config_path))
    assert settings["max_turns"] == 3
    assert settings["normalize_by_std"] is True
    assert "algorithm.adv_estimator=grpo" in overrides
    assert "algorithm.norm_adv_by_std_in_grpo=true" in overrides
    assert "++algorithm.group_by_agent_id=true" in overrides
    assert any("ATGRPOHooks" in item for item in overrides)
    assert any("trajweave_atgrpo_agent_turn_wise_grpo" in item for item in overrides)


def test_atgrpo_normalize_by_std_false_propagates_to_verl_override():
    config = {"atgrpo": {"normalize_by_std": False, "max_turns": 2}, "verl": {"overrides": []}}
    overrides = build_atgrpo_launch_overrides(config, config_path="dummy.yaml")
    assert "algorithm.norm_adv_by_std_in_grpo=false" in overrides
    assert "algorithm.norm_adv_by_std_in_grpo=true" not in overrides


def test_atgrpo_credit_assigner_normalizes_within_rollout_turn_and_agent():
    from trajweave.core.specs import AgentSpec, PolicyGroupSpec, TeamSpec
    from trajweave.core.trajectory import AgentTurn, MultiAgentTrajectory

    team = TeamSpec(
        name="atgrpo_solver_verifier_math",
        agents=(
            AgentSpec(name="solver", role="solver", policy_group="shared", trainable=True),
            AgentSpec(name="verifier", role="verifier", policy_group="shared", trainable=True),
        ),
        policy_groups=(PolicyGroupSpec(name="shared", backend="local", trainable=True),),
        orchestra="solver_verifier",
        reward="math_exact_match",
        credit="atgrpo_agent_turn_wise_grpo",
        max_turns=2,
    )

    def _trajectory(episode_id: str, rollout_group: str, reward: float) -> MultiAgentTrajectory:
        trajectory = MultiAgentTrajectory(
            episode_id=episode_id, task_id=rollout_group, rollout_group=rollout_group, team_name=team.name
        )
        trajectory.add_turn(
            AgentTurn(
                episode_id=episode_id,
                task_id=rollout_group,
                turn_id=0,
                agent_name="solver",
                role="solver",
                policy_group="shared",
                observation="q",
                prompt="p",
                action_text="a",
            )
        )
        trajectory.add_turn(
            AgentTurn(
                episode_id=episode_id,
                task_id=rollout_group,
                turn_id=1,
                agent_name="verifier",
                role="verifier",
                policy_group="shared",
                observation="q",
                prompt="p",
                action_text="a",
            )
        )
        trajectory.global_reward = reward
        trajectory.success = reward > 0
        return trajectory

    trajectories = [
        _trajectory("ep-0", "task", 1.0),
        _trajectory("ep-1", "task", 0.0),
    ]
    samples = ATGRPOCreditAssigner().assign(trajectories, team)
    assert len(samples) == 4

    turn0_solver = [s for s in samples if s.turn_id == 0 and s.agent_name == "solver"]
    turn1_verifier = [s for s in samples if s.turn_id == 1 and s.agent_name == "verifier"]
    assert len(turn0_solver) == 2
    assert len(turn1_verifier) == 2
    # Within each (turn, agent) group, advantages are zero-mean.
    assert abs(sum(s.advantage for s in turn0_solver)) < 1e-6
    assert abs(sum(s.advantage for s in turn1_verifier)) < 1e-6
    high_reward_sample = next(s for s in turn0_solver if s.reward == 1.0)
    low_reward_sample = next(s for s in turn0_solver if s.reward == 0.0)
    assert high_reward_sample.advantage > low_reward_sample.advantage


def test_atgrpo_credit_assigner_singleton_group_matches_verl_hooks_convention():
    """A (turn, agent) group with only one sample (e.g. an extra retry turn that only
    one trajectory reaches) must not be silently zeroed out. It should behave like
    VERL's native GRPO singleton convention (mean=0, std=1 -> advantage == reward),
    matching ATGRPOHooks.compute_grpo_outcome_advantage on the VERL training path.
    """
    from trajweave.core.specs import AgentSpec, PolicyGroupSpec, TeamSpec
    from trajweave.core.trajectory import AgentTurn, MultiAgentTrajectory

    team = TeamSpec(
        name="atgrpo_solver_verifier_math",
        agents=(
            AgentSpec(name="solver", role="solver", policy_group="shared", trainable=True),
            AgentSpec(name="verifier", role="verifier", policy_group="shared", trainable=True),
        ),
        policy_groups=(PolicyGroupSpec(name="shared", backend="local", trainable=True),),
        orchestra="solver_verifier",
        reward="math_exact_match",
        credit="atgrpo_agent_turn_wise_grpo",
        max_turns=3,
    )

    def _trajectory(episode_id: str, reward: float, n_turns: int) -> MultiAgentTrajectory:
        trajectory = MultiAgentTrajectory(
            episode_id=episode_id, task_id="task", rollout_group="task", team_name=team.name
        )
        names = ["solver", "verifier", "solver"]
        for turn_id in range(n_turns):
            trajectory.add_turn(
                AgentTurn(
                    episode_id=episode_id,
                    task_id="task",
                    turn_id=turn_id,
                    agent_name=names[turn_id],
                    role=names[turn_id],
                    policy_group="shared",
                    observation="q",
                    prompt="p",
                    action_text="a",
                )
            )
        trajectory.global_reward = reward
        trajectory.success = reward > 0
        return trajectory

    # ep-2 is the only trajectory with a 3rd turn (turn_id=2, solver) -> singleton group.
    trajectories = [
        _trajectory("ep-0", 0.0, 2),
        _trajectory("ep-1", 1.0, 2),
        _trajectory("ep-2", 1.0, 3),
    ]
    samples = ATGRPOCreditAssigner(normalize_by_std=True).assign(trajectories, team)
    singleton = next(s for s in samples if s.turn_id == 2 and s.agent_name == "solver")
    assert singleton.reward == 1.0
    # advantage == reward / (std + epsilon) == reward - mean, with mean=0, std=1 (matches
    # ATGRPOHooks/core_algos.py's singleton convention rather than self-centering to zero).
    assert singleton.advantage == pytest.approx(singleton.reward, abs=1e-4)
    assert singleton.metadata["reward_mean"] == 0.0
    assert singleton.metadata["reward_std"] == 1.0


def _mixed_reward_team() -> Any:
    from trajweave.core.specs import AgentSpec, PolicyGroupSpec, TeamSpec

    return TeamSpec(
        name="atgrpo_solver_verifier_math",
        agents=(
            AgentSpec(name="solver", role="solver", policy_group="shared", trainable=True),
            AgentSpec(name="verifier", role="verifier", policy_group="shared", trainable=True),
        ),
        policy_groups=(PolicyGroupSpec(name="shared", backend="local", trainable=True),),
        orchestra="solver_verifier",
        reward="math_exact_match",
        credit="atgrpo_agent_turn_wise_grpo",
        max_turns=2,
    )


def _mixed_reward_trajectory(episode_id: str, *, global_reward: float, model_approved: bool) -> Any:
    from trajweave.core.trajectory import AgentTurn, MultiAgentTrajectory

    team = _mixed_reward_team()
    trajectory = MultiAgentTrajectory(episode_id=episode_id, task_id="task", rollout_group="task", team_name=team.name)
    trajectory.add_turn(
        AgentTurn(
            episode_id=episode_id,
            task_id="task",
            turn_id=0,
            agent_name="solver",
            role="solver",
            policy_group="shared",
            observation="q",
            prompt="p",
            action_text="a",
        )
    )
    trajectory.add_turn(
        AgentTurn(
            episode_id=episode_id,
            task_id="task",
            turn_id=1,
            agent_name="verifier",
            role="verifier",
            policy_group="shared",
            observation="q",
            prompt="p",
            action_text="a",
            metadata={"model_approved": model_approved, "approved": model_approved},
        )
    )
    trajectory.global_reward = global_reward
    trajectory.success = global_reward > 0
    return trajectory


def test_atgrpo_mixed_reward_gives_verifier_agent_specific_signal():
    """With mixed_reward enabled, a verifier that judged incorrectly must receive a
    different (lower) reward than one that judged correctly, even when both episodes
    share the same global_reward. Solver reward stays tied to the global outcome only,
    since it has no independent local signal in this environment.
    """
    team = _mixed_reward_team()
    correct_judgment = _mixed_reward_trajectory("ep-correct", global_reward=1.0, model_approved=True)
    wrong_judgment = _mixed_reward_trajectory("ep-wrong", global_reward=1.0, model_approved=False)

    samples = ATGRPOCreditAssigner(mixed_reward_enabled=True, alpha=1.0, verifier_local_reward_scale=1.0).assign(
        [correct_judgment, wrong_judgment], team
    )

    solver_rewards = {s.episode_id: s.reward for s in samples if s.agent_name == "solver"}
    verifier_rewards = {s.episode_id: s.reward for s in samples if s.agent_name == "verifier"}

    # Solver reward is untouched by mixed reward -- both episodes share the same outcome.
    assert solver_rewards["ep-correct"] == solver_rewards["ep-wrong"] == 1.0

    # Verifier reward now diverges: alpha*global + local, local=+1 if judged correctly else -1.
    assert verifier_rewards["ep-correct"] == pytest.approx(1.0 * 1.0 + 1.0)
    assert verifier_rewards["ep-wrong"] == pytest.approx(1.0 * 1.0 - 1.0)
    assert verifier_rewards["ep-correct"] != verifier_rewards["ep-wrong"]
    # Before this fix, solver and verifier rewards within the same episode were forced
    # identical (both == global_reward). Now the verifier's reward differs from the
    # solver's when its judgment was correct/incorrect relative to the global outcome.
    assert verifier_rewards["ep-wrong"] != solver_rewards["ep-wrong"]


def test_atgrpo_mixed_reward_disabled_preserves_legacy_behavior():
    """Default (mixed_reward_enabled=False) must be byte-identical to the pre-fix
    behavior: solver and verifier both get the plain global_reward."""
    team = _mixed_reward_team()
    trajectory = _mixed_reward_trajectory("ep-0", global_reward=1.0, model_approved=False)

    samples = ATGRPOCreditAssigner().assign([trajectory], team)
    rewards = {s.agent_name: s.reward for s in samples}
    assert rewards["solver"] == 1.0
    assert rewards["verifier"] == 1.0


def test_atgrpo_verl_hook_groups_by_rollout_turn_and_agent():
    response_mask = torch.ones(4, 1, dtype=torch.int64)
    token_level_rewards = torch.tensor([[1.0], [0.0], [0.0], [10.0]], dtype=torch.float32)
    data = DataProto.from_dict(
        tensors={"token_level_rewards": token_level_rewards, "response_mask": response_mask},
        non_tensors={
            "uid": np.array(["task"] * 4, dtype=object),
            "agent_id": np.array(["solver", "verifier", "solver", "verifier"], dtype=object),
            "traj_uid": np.array(["ep-0", "ep-0", "ep-1", "ep-1"], dtype=object),
            "turn_id": np.array([0, 1, 0, 1], dtype=object),
        },
    )
    hooks = ATGRPOHooks()
    result = hooks.compute_advantage(
        data,
        batch_keys=None,
        adv_estimator=AdvantageEstimator.GRPO,
        gamma=1.0,
        lam=1.0,
        num_repeat=1,
        norm_adv_by_std_in_grpo=False,
        config={"group_by_agent_id": True},
        fallback=None,
    )
    advantages = result.batch["advantages"].squeeze(-1)
    # Groups are ("task", turn_id, agent_id): rows 0/2 share (turn 0, solver),
    # rows 1/3 share (turn 1, verifier). Higher reward within each group yields
    # a positive advantage relative to its group mean.
    assert advantages[0].item() > 0  # solver, reward=1.0, group mean=0.5
    assert advantages[2].item() < 0  # solver, reward=0.0, group mean=0.5
    assert advantages[1].item() < 0  # verifier, reward=0.0, group mean=5.0
    assert advantages[3].item() > 0  # verifier, reward=10.0, group mean=5.0

