from __future__ import annotations

from copy import deepcopy
from math import isfinite

import numpy as np
import pytest
import torch

from trajweave.backends.verl.schema import resolve_comlrl_extra_fields
from trajweave.core.joint_trajectory import JointAction, JointCompletion, JointTransition, JointTreeNode
from trajweave.core.specs import AgentSpec, PolicyGroupSpec, TeamSpec
from trajweave.core.trajectory import MultiAgentTrajectory
from trajweave.credit.comlrl import (
    CoMLRLReinforceCreditAssigner,
    apply_sequence_kl,
    compute_group_advantages,
    compute_joint_tree_returns,
    project_joint_returns_to_completions,
)


def _terminal_action(
    action_id: str,
    node_id: str,
    reward: float,
    *,
    completion_ids: dict[str, str] | None = None,
    transition_id: str | None = None,
    done: bool = True,
    truncated: bool = False,
    stop_reason: str | None = None,
) -> JointAction:
    completion_ids = completion_ids or {"alice": f"{action_id}:alice"}
    return JointAction(
        joint_action_id=action_id,
        tree_node_id=node_id,
        completion_ids=completion_ids,
        candidate_indices={name: 0 for name in completion_ids},
        joint_transition_id=transition_id or f"{action_id}:transition",
        shared_reward=reward,
        done=done,
        truncated=truncated,
        stop_reason=stop_reason,
    )


def test_two_level_returns_are_computed_per_action_branch_without_mutation():
    nodes = [
        JointTreeNode("root", "root", 0),
        JointTreeNode("left", "root", 1, parent_joint_action_id="root-left"),
        JointTreeNode("right", "root", 1, parent_joint_action_id="root-right"),
    ]
    actions = [
        JointAction(
            "root-left",
            "root",
            {"alice": "root-left:alice"},
            {"alice": 0},
            "root-left:transition",
            shared_reward=1.0,
            child_tree_node_id="left",
        ),
        JointAction(
            "root-right",
            "root",
            {"alice": "root-right:alice"},
            {"alice": 0},
            "root-right:transition",
            shared_reward=10.0,
            child_tree_node_id="right",
        ),
        _terminal_action("left-0", "left", 2.0),
        _terminal_action("left-1", "left", 4.0),
        _terminal_action("right-0", "right", -2.0),
        _terminal_action("right-1", "right", 6.0),
    ]

    returns = compute_joint_tree_returns(nodes, actions)

    assert returns == {
        "root-left": 4.0,
        "root-right": 12.0,
        "left-0": 2.0,
        "left-1": 4.0,
        "right-0": -2.0,
        "right-1": 6.0,
    }
    assert all(action.shared_return is None for action in actions)


def test_truncated_and_unexpanded_leaf_actions_use_immediate_reward_only():
    nodes = [
        JointTreeNode("root", "root", 0),
        JointTreeNode(
            "safe-leaf",
            "root",
            1,
            parent_joint_action_id="unexpanded",
            truncated=True,
            stop_reason="max_joint_actions",
        ),
    ]
    actions = [
        _terminal_action(
            "truncated",
            "root",
            7.0,
            done=False,
            truncated=True,
            stop_reason="max_turns",
        ),
        JointAction(
            "unexpanded",
            "root",
            {"alice": "unexpanded:alice"},
            {"alice": 0},
            "unexpanded:transition",
            shared_reward=3.0,
            child_tree_node_id="safe-leaf",
        ),
    ]

    assert compute_joint_tree_returns(nodes, actions) == {"truncated": 7.0, "unexpanded": 3.0}


def _projection_completions() -> list[JointCompletion]:
    return [
        JointCompletion("a0", "root", "alice", 0, "a0"),
        JointCompletion("a1", "root", "alice", 1, "a1"),
        JointCompletion("b0", "root", "bob", 0, "b0"),
        JointCompletion("b1", "root", "bob", 1, "b1"),
    ]


def _projection_action(action_id: str, a: str, b: str, reward: float) -> JointAction:
    return JointAction(
        action_id,
        "root",
        {"alice": a, "bob": b},
        {"alice": int(a[-1]), "bob": int(b[-1])},
        f"{action_id}:transition",
        shared_reward=reward,
        done=True,
    )


def test_aligned_projection_keeps_one_action_transition_component_per_completion():
    completions = _projection_completions()
    actions = [
        _projection_action("j0", "a0", "b0", 1.0),
        _projection_action("j1", "a1", "b1", 3.0),
    ]

    projected = project_joint_returns_to_completions(
        completions,
        actions,
        {"j0": 1.5, "j1": 4.0},
        "aligned",
    )

    assert list(projected) == ["a0", "a1", "b0", "b1"]
    assert projected["a0"].joint_action_ids == ["j0"]
    assert projected["a0"].joint_transition_ids == ["j0:transition"]
    assert projected["a0"].joint_return_components == [1.5]
    assert projected["a0"].projected_joint_return == 1.5
    assert projected["b1"].projected_joint_return == 4.0


def test_cross_projection_sums_every_containing_action_with_aligned_ids():
    completions = _projection_completions()
    actions = [
        _projection_action("j00", "a0", "b0", 0.0),
        _projection_action("j01", "a0", "b1", 0.0),
        _projection_action("j10", "a1", "b0", 0.0),
        _projection_action("j11", "a1", "b1", 0.0),
    ]
    returns = {"j00": 1.0, "j01": 2.0, "j10": 4.0, "j11": 8.0}

    projected = project_joint_returns_to_completions(completions, actions, returns, "cross")

    assert projected["a0"].joint_action_ids == ["j00", "j01"]
    assert projected["a0"].joint_transition_ids == ["j00:transition", "j01:transition"]
    assert projected["a0"].joint_return_components == [1.0, 2.0]
    assert projected["a0"].projected_joint_return == 3.0
    assert projected["a1"].projected_joint_return == 12.0
    assert projected["b0"].projected_joint_return == 5.0
    assert projected["b1"].projected_joint_return == 10.0


def test_projection_rejects_unpaired_completions_and_partial_cross_grids():
    completions = _projection_completions()

    with pytest.raises(ValueError, match="does not belong to any joint action"):
        project_joint_returns_to_completions(
            completions,
            [_projection_action("j0", "a0", "b0", 1.0)],
            {"j0": 1.0},
            "aligned",
        )

    partial_cross = [
        _projection_action("j00", "a0", "b0", 1.0),
        _projection_action("j01", "a0", "b1", 1.0),
        _projection_action("j10", "a1", "b0", 1.0),
    ]
    with pytest.raises(ValueError, match="complete Cartesian grid"):
        project_joint_returns_to_completions(
            completions,
            partial_cross,
            {action.joint_action_id: 1.0 for action in partial_cross},
            "cross",
        )

    duplicate_cross = [
        _projection_action("duplicate-00-a", "a0", "b0", 1.0),
        _projection_action("duplicate-00-b", "a0", "b0", 1.0),
        _projection_action("duplicate-11-a", "a1", "b1", 1.0),
        _projection_action("duplicate-11-b", "a1", "b1", 1.0),
    ]
    with pytest.raises(ValueError, match="complete Cartesian grid"):
        project_joint_returns_to_completions(
            completions,
            duplicate_cross,
            {action.joint_action_id: 1.0 for action in duplicate_cross},
            "cross",
        )


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("raw", [1.0, 2.0, 4.0]),
        ("mean", [-4.0 / 3.0, -1.0 / 3.0, 5.0 / 3.0]),
        ("rloo", [-2.0, -0.5, 2.5]),
        ("max", [-3.0, -2.0, 0.0]),
    ],
)
def test_advantage_modes_without_normalization(mode: str, expected: list[float]):
    assert compute_group_advantages([1.0, 2.0, 4.0], mode, False) == pytest.approx(expected)


@pytest.mark.parametrize("mode", ["raw", "mean", "rloo", "max"])
def test_advantage_modes_use_population_std_when_normalized(mode: str):
    advantages = compute_group_advantages([1.0, 2.0, 4.0], mode, True)

    assert sum(advantages) == pytest.approx(0.0)
    assert sum(value * value for value in advantages) / len(advantages) == pytest.approx(1.0)


def test_default_normalization_makes_all_four_v141_modes_equivalent():
    values = [1.0, 2.0, 4.0]
    outputs = [compute_group_advantages(values, mode, True) for mode in ("raw", "mean", "rloo", "max")]

    for output in outputs[1:]:
        assert output == pytest.approx(outputs[0])


def test_verl_hook_routes_projected_returns_into_real_actor_advantages():
    from trajweave.backends.verl.extensions.comlrl import CoMLRLReinforceHooks
    from trajweave.backends.verl.extensions.common.hooks import extension_hooks_for_config
    from trajweave.backends.verl.extensions.registry import _extension_names
    from verl.protocol import DataProto

    def object_rows(values):
        array = np.empty(len(values), dtype=object)
        array[:] = values
        return array

    response_mask = torch.tensor([[1, 1], [1, 0], [1, 1], [1, 0], [0, 0]])
    proto = DataProto.from_dict(
        tensors={
            "response_mask": response_mask,
            "token_level_rewards": torch.tensor([[0.0, 1.0], [3.0, 0.0], [0.0, 10.0], [14.0, 0.0], [0.0, 0.0]]),
        },
        non_tensors={
            "traj_uid": object_rows(["episode"] * 4 + ["padding"]),
            "agent_id": object_rows(["alice", "alice", "bob", "bob", "padding"]),
            "worker_group": object_rows(["actor-a", "actor-a", "actor-b", "actor-b", "actor-a"]),
            "tree_node_id": object_rows(["node"] * 4 + ["__padding__:node"]),
            "projected_joint_return": object_rows([1.0, 3.0, 10.0, 14.0, 0.0]),
            "effective_projected_joint_return": object_rows([1.0, 3.0, 10.0, 14.0, 0.0]),
            "joint_advantage": object_rows([-1.0, 1.0, -2.0, 2.0, 0.0]),
            "joint_action_ids": object_rows([["a0"], ["a1"], ["a0"], ["a1"], []]),
            "joint_transition_ids": object_rows([["t0"], ["t1"], ["t0"], ["t1"], []]),
            "joint_return_components": object_rows([[1.0], [3.0], [10.0], [14.0], []]),
            "joint_sampling_mode": object_rows(["aligned"] * 4 + ["__padding__"]),
        },
    )

    result = CoMLRLReinforceHooks().compute_advantage(
        proto,
        batch_keys=[f"row-{index}" for index in range(5)],
        adv_estimator="reinforce_plus_plus",
        gamma=1.0,
        lam=1.0,
        num_repeat=1,
        norm_adv_by_std_in_grpo=False,
        config={"advantage_mode": "mean", "normalize_advantages": False},
    )

    assert torch.equal(
        result.batch["advantages"],
        torch.tensor([[-1.0, -1.0], [1.0, 0.0], [-2.0, -2.0], [2.0, 0.0], [0.0, 0.0]]),
    )
    assert torch.equal(
        result.batch["returns"],
        torch.tensor([[1.0, 1.0], [3.0, 0.0], [10.0, 10.0], [14.0, 0.0], [0.0, 0.0]]),
    )
    config = {"trajweave": {"credit_allocator": "comlrl_maremax"}}
    assert isinstance(extension_hooks_for_config(config), CoMLRLReinforceHooks)
    assert _extension_names(config) == ("trajweave_comlrl_reinforce",)


def test_kl_is_applied_after_projection_and_before_group_advantage():
    effective = [apply_sequence_kl(3.0, 2.0, 0.5), apply_sequence_kl(1.0, 0.0, 0.5)]

    assert effective == [2.0, 1.0]
    assert compute_group_advantages(effective, "mean", False) == [0.5, -0.5]


@pytest.mark.parametrize("mode", ["raw", "mean", "rloo", "max"])
def test_zero_variance_and_singleton_advantages_are_finite(mode: str):
    zero_variance = compute_group_advantages([2.0, 2.0], mode, True)
    singleton = compute_group_advantages([3.0], mode, True)

    assert zero_variance == [0.0, 0.0]
    assert all(isfinite(value) for value in singleton)
    assert singleton == ([3.0] if mode == "raw" else [0.0])


def test_returns_reject_bad_discount_non_finite_reward_and_bad_reference():
    root = JointTreeNode("root", "root", 0)
    terminal = _terminal_action("terminal", "root", 1.0)

    with pytest.raises(ValueError, match="exactly 1.0"):
        compute_joint_tree_returns([root], [terminal], discount=0.9)
    terminal.shared_reward = float("nan")
    with pytest.raises(ValueError, match="finite"):
        compute_joint_tree_returns([root], [terminal])
    terminal.shared_reward = 1.0
    terminal.tree_node_id = "missing"
    with pytest.raises(ValueError, match="unknown source"):
        compute_joint_tree_returns([root], [terminal])


def test_returns_reject_cycles():
    nodes = [
        JointTreeNode("a", "a", 0, parent_joint_action_id="b-to-a"),
        JointTreeNode("b", "a", 1, parent_joint_action_id="a-to-b"),
    ]
    actions = [
        JointAction(
            "a-to-b",
            "a",
            {"alice": "a"},
            {"alice": 0},
            "ta",
            shared_reward=1.0,
            child_tree_node_id="b",
        ),
        JointAction(
            "b-to-a",
            "b",
            {"alice": "b"},
            {"alice": 0},
            "tb",
            shared_reward=1.0,
            child_tree_node_id="a",
        ),
    ]

    with pytest.raises(ValueError, match="cycle"):
        compute_joint_tree_returns(nodes, actions)


@pytest.mark.parametrize(
    ("projected_return", "sequence_kl", "coefficient"),
    [(0.0, -1.0, 1.0), (0.0, 1.0, -1.0), (0.0, float("nan"), 1.0)],
)
def test_sequence_kl_rejects_negative_or_non_finite_inputs(
    projected_return: float,
    sequence_kl: float,
    coefficient: float,
):
    with pytest.raises(ValueError):
        apply_sequence_kl(projected_return, sequence_kl, coefficient)


def _team() -> TeamSpec:
    return TeamSpec(
        name="joint-team",
        agents=(
            AgentSpec("alice", "solver", "alice-policy"),
            AgentSpec("bob", "reviewer", "bob-policy"),
        ),
        policy_groups=(PolicyGroupSpec("alice-policy"), PolicyGroupSpec("bob-policy")),
        orchestra="comlrl_full_tree",
        reward="joint",
        credit="comlrl_reinforce",
    )


def _credit_trajectory() -> MultiAgentTrajectory:
    node = JointTreeNode("root", "root", 0)
    completions = [
        JointCompletion(
            "a0",
            "root",
            "alice",
            0,
            "alice zero",
            [10],
            [-0.1],
            {"prompt": "prompt a", "observation": "obs a", "sequence_kl": 2.0},
        ),
        JointCompletion(
            "a1",
            "root",
            "alice",
            1,
            "alice one",
            [11],
            [-0.2],
            {"prompt": "prompt a", "observation": "obs a"},
        ),
        JointCompletion(
            "b0",
            "root",
            "bob",
            0,
            "bob zero",
            [20],
            [-0.3],
            {"prompt": "prompt b", "observation": "obs b"},
        ),
        JointCompletion(
            "b1",
            "root",
            "bob",
            1,
            "bob one",
            [21],
            [-0.4],
            {"prompt": "prompt b", "observation": "obs b"},
        ),
    ]
    actions = [
        _projection_action("j0", "a0", "b0", 1.0),
        _projection_action("j1", "a1", "b1", 3.0),
    ]
    transitions = [
        JointTransition("j0:transition", "j0", "root", 0, done=True),
        JointTransition("j1:transition", "j1", "root", 0, done=True),
    ]
    return MultiAgentTrajectory(
        episode_id="episode-0",
        task_id="task-0",
        rollout_group="group-0",
        team_name="joint-team",
        metadata={"joint_mode": "aligned"},
        joint_nodes=[node],
        joint_completions=completions,
        joint_actions=actions,
        joint_transitions=transitions,
    )


def test_credit_assigner_emits_one_sample_per_completion_and_all_bridge_fields():
    samples = CoMLRLReinforceCreditAssigner(
        advantage_mode="mean",
        normalize=False,
        sequence_kl_coefficient=0.5,
    ).assign([_credit_trajectory()], _team())

    assert [sample.completion_id for sample in samples] == ["a0", "a1", "b0", "b1"]
    by_id = {sample.completion_id: sample for sample in samples}
    assert by_id["a0"].role == "solver"
    assert by_id["a0"].policy_group == "alice-policy"
    assert by_id["a0"].prompt == "prompt a"
    assert by_id["a0"].response == "alice zero"
    assert by_id["a0"].reward == 0.0
    assert by_id["a0"].advantage == pytest.approx(-1.5)
    assert by_id["a1"].advantage == pytest.approx(1.5)
    assert by_id["b0"].advantage == pytest.approx(-1.0)
    assert by_id["b1"].advantage == pytest.approx(1.0)

    sample = by_id["a0"]
    assert sample.tree_node_id == "root"
    assert sample.joint_action_ids == ["j0"]
    assert sample.joint_transition_ids == ["j0:transition"]
    assert sample.metadata["joint_return_components"] == [1.0]
    assert sample.metadata["projected_joint_return"] == 1.0
    assert sample.metadata["joint_reward"] == 1.0
    assert sample.metadata["joint_done"] is True
    assert sample.metadata["joint_truncated"] is False
    assert sample.metadata["joint_stop_reason"] == ""
    assert sample.metadata["joint_sampling_mode"] == "aligned"
    assert sample.metadata["observation"] == "obs a"

    extra_fields = {
        **sample.metadata,
        "completion_id": sample.completion_id,
        "tree_node_id": sample.tree_node_id,
        "joint_action_ids": sample.joint_action_ids,
        "joint_transition_ids": sample.joint_transition_ids,
    }
    resolved = resolve_comlrl_extra_fields(extra_fields, row_id=sample.sample_id)
    assert resolved["completion_id"] == "a0"
    assert resolved["tree_node_id"] == "root"
    assert resolved["joint_action_ids"] == ["j0"]
    assert resolved["joint_transition_ids"] == ["j0:transition"]
    assert resolved["joint_return_components"] == [1.0]
    assert resolved["projected_joint_return"] == 1.0


def test_credit_assigner_reads_nested_cross_mode_and_preserves_component_edges():
    trajectory = _credit_trajectory()
    trajectory.metadata = {"joint_tree": {"joint_mode": "cross"}}
    trajectory.joint_actions = [
        _projection_action("j00", "a0", "b0", 1.0),
        _projection_action("j01", "a0", "b1", 2.0),
        _projection_action("j10", "a1", "b0", 4.0),
        _projection_action("j11", "a1", "b1", 8.0),
    ]
    trajectory.joint_transitions = [
        JointTransition(f"{action.joint_action_id}:transition", action.joint_action_id, "root", 0, done=True)
        for action in trajectory.joint_actions
    ]

    samples = CoMLRLReinforceCreditAssigner(normalize=False).assign([trajectory], _team())
    by_id = {sample.completion_id: sample for sample in samples}

    assert by_id["a0"].metadata["joint_sampling_mode"] == "cross"
    assert by_id["a0"].joint_action_ids == ["j00", "j01"]
    assert by_id["a0"].joint_transition_ids == ["j00:transition", "j01:transition"]
    assert by_id["a0"].metadata["projected_joint_return"] == 3.0


def test_advantage_groups_are_isolated_by_episode_even_when_node_ids_repeat():
    first = _credit_trajectory()
    second = deepcopy(first)
    second.episode_id = "episode-1"
    second.joint_actions[0].shared_reward = 101.0
    second.joint_actions[1].shared_reward = 203.0

    samples = CoMLRLReinforceCreditAssigner(advantage_mode="mean", normalize=False).assign([first, second], _team())
    by_key = {(sample.episode_id, sample.completion_id): sample for sample in samples}

    assert by_key[("episode-0", "a0")].advantage == -1.0
    assert by_key[("episode-0", "a1")].advantage == 1.0
    assert by_key[("episode-1", "a0")].advantage == -51.0
    assert by_key[("episode-1", "a1")].advantage == 51.0
    assert by_key[("episode-1", "a0")].metadata["advantage_group"]["episode_id"] == "episode-1"
