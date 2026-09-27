import math

import pytest

from trajweave.core import SearchNode, TreeTrajectory
from trajweave.credit import TreeGroupBuilder, group_normalized_advantages, importance_correction_weights


def make_node(tree_id: str, node_id: int, reward: float, *, parent_idx: int | None = None) -> SearchNode:
    return SearchNode(
        tree_id=tree_id,
        prompt_id=f"prompt-{tree_id}",
        node_id=node_id,
        parent_idx=parent_idx,
        agent_name="generator",
        role="generator",
        policy_group="shared",
        prompt="write add_one",
        action_text="return x + 1",
        action_token_ids=[node_id + 10],
        rollout_logprobs=[-0.1],
        reward=reward,
    )


def test_tree_trajectory_validates_identity_and_parent_links():
    tree = TreeTrajectory(tree_id="tree-0", prompt_id="prompt-tree-0", task_id="task-0", rollout_group="group-0")
    tree.add_node(make_node("tree-0", 0, 1.0))
    tree.add_node(make_node("tree-0", 1, 0.0, parent_idx=0))

    assert [node.node_id for node in tree.nodes] == [0, 1]

    with pytest.raises(ValueError, match="missing parent_idx"):
        tree.add_node(make_node("tree-0", 2, 0.0, parent_idx=99))
    assert [node.node_id for node in tree.nodes] == [0, 1]


def test_fixed_contiguous_groups_match_explicit_tree_groups_for_marti_smoke_order():
    nodes = [
        make_node("tree-0", 0, 1.0),
        make_node("tree-0", 1, 0.0, parent_idx=0),
        make_node("tree-1", 0, 1.0),
        make_node("tree-1", 1, 1.0, parent_idx=0),
    ]

    builder = TreeGroupBuilder()

    assert builder.group_indices(nodes) == [[0, 1], [2, 3]]
    assert builder.fixed_size_matches_tree_groups(nodes, group_size=2)


def test_tree_group_builder_rejects_interleaved_tree_nodes():
    nodes = [
        make_node("tree-0", 0, 1.0),
        make_node("tree-1", 0, 1.0),
        make_node("tree-0", 1, 0.0, parent_idx=0),
        make_node("tree-1", 1, 1.0, parent_idx=0),
    ]

    with pytest.raises(ValueError, match="not contiguous"):
        TreeGroupBuilder().group_indices(nodes)

    assert TreeGroupBuilder(require_contiguous=False).group_indices(nodes) == [[0, 2], [1, 3]]


def test_group_normalized_advantages_match_marti_two_node_groups():
    rewards = [1.0, 0.0, 1.0, 1.0]
    groups = [[0, 1], [2, 3]]

    advantages = group_normalized_advantages(rewards, groups)

    assert advantages[0] == pytest.approx(1 / math.sqrt(2))
    assert advantages[1] == pytest.approx(-1 / math.sqrt(2))
    assert advantages[2:] == [0.0, 0.0]


def test_importance_correction_truncates_and_masks_finite_weights():
    old_logprobs = [[math.log(3.0), math.log(0.25), 0.0]]
    rollout_logprobs = [[0.0, 0.0, 0.0]]
    action_mask = [[1, 1, 0]]

    truncated = importance_correction_weights(
        old_logprobs=old_logprobs,
        rollout_logprobs=rollout_logprobs,
        action_mask=action_mask,
        level="token",
        mode="truncate",
        upper_threshold=2.0,
    )
    masked = importance_correction_weights(
        old_logprobs=old_logprobs,
        rollout_logprobs=rollout_logprobs,
        action_mask=action_mask,
        level="token",
        mode="mask",
        upper_threshold=2.0,
    )

    assert truncated == [[2.0, 0.25, 0.0]]
    assert masked == [[0.0, 0.0, 0.0]]
    assert all(math.isfinite(value) for row in truncated + masked for value in row)


def test_importance_correction_supports_sequence_level_weights():
    weights = importance_correction_weights(
        old_logprobs=[[math.log(2.0), math.log(0.5), 10.0]],
        rollout_logprobs=[[0.0, 0.0, 0.0]],
        action_mask=[[1, 1, 0]],
        level="sequence",
        mode="truncate",
        upper_threshold=2.0,
    )

    assert weights == [[1.0, 1.0, 0.0]]
