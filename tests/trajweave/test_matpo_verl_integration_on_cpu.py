import numpy as np
import pytest
import torch
from tensordict import TensorDict

from trajweave.backends.verl.dataproto import VerlDataProtoAdapter
from trajweave.backends.verl.extensions.common.hooks import extension_hooks_for_config
from trajweave.backends.verl.extensions.matpo import MATPOParentBroadcastHooks, MATPOParentChildIntegrityError
from trajweave.backends.verl.extensions.registry import _extension_names
from trajweave.recipes.matpo import run_smoke
from verl.protocol import DataProto


def _unreachable_fallback(main_data, **kwargs):
    raise AssertionError("fallback must not run when parent/child linkage validation fails")


def test_matpo_dataproto_contains_parent_child_fields():
    _, result = run_smoke(rollouts_per_task=1, max_turns=3)
    data = VerlDataProtoAdapter().build(result.samples)

    assert "is_from_subagent_tool" in data.batch
    assert "turn_count" in data.batch
    assert "reqs_id" in data.non_tensor_batch
    assert "parent_reqs_id" in data.non_tensor_batch
    assert data.batch["is_from_subagent_tool"].any()
    # MATPO-specific fields must also be present on the MATPO recipe's own DataProto.
    assert "matpo_tool_format_valid" in data.batch
    assert "matpo_tool_call_count" in data.batch


def test_matpo_specific_fields_are_isolated_from_other_recipes():
    # A non-MATPO recipe's DataProto must not pick up MATPO-only tensors, keeping
    # papers' schemas from bleeding into each other through the shared adapter.
    from trajweave.recipes.doctor_mas.math_smoke import build_doctor_mas_math_engine, default_math_tasks

    engine = build_doctor_mas_math_engine()
    result = engine.run(default_math_tasks(), rollouts_per_task=1)
    data = VerlDataProtoAdapter().build(result.samples)

    assert "matpo_tool_format_valid" not in data.batch
    assert "matpo_tool_call_count" not in data.batch


def test_matpo_extension_auto_selects_parent_broadcast():
    assert _extension_names({"trajweave": {"recipe": "matpo_browse"}}) == ("trajweave_matpo_parent_broadcast",)
    assert isinstance(
        extension_hooks_for_config({"trajweave": {"credit_allocator": "matpo_parent_broadcast_grpo"}}),
        MATPOParentBroadcastHooks,
    )


def test_matpo_hook_accepts_child_marker_from_tensor_batch():
    data = DataProto(
        batch=TensorDict(
            {
                "token_level_rewards": torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
                "response_mask": torch.tensor([[1, 1], [1, 0]], dtype=torch.long),
                "is_from_subagent_tool": torch.tensor([False, True]),
            },
            batch_size=[2],
        ),
        non_tensor_batch={
            "uid": np.array(["task", "task"], dtype=object),
            "reqs_id": np.array(["main", "child"], dtype=object),
            "parent_reqs_id": np.array(["", "main"], dtype=object),
        },
    )

    def fallback(main_data, **kwargs):
        main_data.batch["advantages"] = torch.tensor([[0.25, 0.25]])
        main_data.batch["returns"] = main_data.batch["advantages"].clone()
        return main_data

    output = MATPOParentBroadcastHooks().compute_advantage(data, adv_estimator="grpo", fallback=fallback)

    assert torch.equal(output.batch["advantages"][0], torch.tensor([0.25, 0.25]))
    assert torch.equal(output.batch["advantages"][1], torch.tensor([0.25, 0.0]))


def test_matpo_hook_broadcasts_main_advantage_to_child():
    data = DataProto(
        batch=TensorDict(
            {
                "token_level_rewards": torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
                "response_mask": torch.tensor([[1, 1], [1, 1], [1, 0]], dtype=torch.long),
            },
            batch_size=[3],
        ),
        non_tensor_batch={
            "uid": np.array(["task", "task", "task"], dtype=object),
            "reqs_id": np.array(["main_a", "main_b", "child_a"], dtype=object),
            "parent_reqs_id": np.array(["", "", "main_a"], dtype=object),
            "is_from_subagent_tool": np.array([False, False, True], dtype=object),
        },
    )

    def fallback(main_data, **kwargs):
        main_data.batch["advantages"] = torch.tensor([[0.5, 0.5], [-0.5, -0.5]])
        main_data.batch["returns"] = main_data.batch["advantages"].clone()
        return main_data

    output = MATPOParentBroadcastHooks().compute_advantage(
        data,
        adv_estimator="grpo",
        fallback=fallback,
        batch_keys=["a", "b", "c"],
    )

    assert torch.equal(output.batch["advantages"][0], torch.tensor([0.5, 0.5]))
    assert torch.equal(output.batch["advantages"][1], torch.tensor([-0.5, -0.5]))
    assert torch.equal(output.batch["advantages"][2], torch.tensor([0.5, 0.0]))


def test_matpo_hook_broadcasts_scalar_advantage_when_parent_shorter_than_child():
    # Parent response has 1 valid token, child response has 4 valid tokens.
    # The parent's scalar advantage must be broadcast to every valid child token,
    # not applied position-by-position (which would zero out most of the child).
    data = DataProto(
        batch=TensorDict(
            {
                "token_level_rewards": torch.zeros(2, 4),
                "response_mask": torch.tensor([[1, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.long),
            },
            batch_size=[2],
        ),
        non_tensor_batch={
            "uid": np.array(["task", "task"], dtype=object),
            "reqs_id": np.array(["main", "child"], dtype=object),
            "parent_reqs_id": np.array(["", "main"], dtype=object),
            "is_from_subagent_tool": np.array([False, True], dtype=object),
        },
    )

    def fallback(main_data, **kwargs):
        main_data.batch["advantages"] = torch.tensor([[0.5, 0.0, 0.0, 0.0]])
        main_data.batch["returns"] = main_data.batch["advantages"].clone()
        return main_data

    output = MATPOParentBroadcastHooks().compute_advantage(data, adv_estimator="grpo", fallback=fallback)

    assert torch.equal(output.batch["advantages"][0], torch.tensor([0.5, 0.0, 0.0, 0.0]))
    assert torch.equal(output.batch["advantages"][1], torch.tensor([0.5, 0.5, 0.5, 0.5]))
    assert torch.equal(output.batch["returns"][1], torch.tensor([0.5, 0.5, 0.5, 0.5]))


def test_matpo_hook_broadcasts_scalar_advantage_when_parent_longer_than_child():
    # Parent response has 4 valid tokens, child response has 1 valid token.
    # Only the child's own response_mask should gate how many tokens receive
    # the broadcast advantage; excess parent length must not leak in.
    data = DataProto(
        batch=TensorDict(
            {
                "token_level_rewards": torch.zeros(2, 4),
                "response_mask": torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0]], dtype=torch.long),
            },
            batch_size=[2],
        ),
        non_tensor_batch={
            "uid": np.array(["task", "task"], dtype=object),
            "reqs_id": np.array(["main", "child"], dtype=object),
            "parent_reqs_id": np.array(["", "main"], dtype=object),
            "is_from_subagent_tool": np.array([False, True], dtype=object),
        },
    )

    def fallback(main_data, **kwargs):
        main_data.batch["advantages"] = torch.tensor([[-0.25, -0.25, -0.25, -0.25]])
        main_data.batch["returns"] = main_data.batch["advantages"].clone()
        return main_data

    output = MATPOParentBroadcastHooks().compute_advantage(data, adv_estimator="grpo", fallback=fallback)

    assert torch.equal(output.batch["advantages"][0], torch.tensor([-0.25, -0.25, -0.25, -0.25]))
    assert torch.equal(output.batch["advantages"][1], torch.tensor([-0.25, 0.0, 0.0, 0.0]))
    assert torch.equal(output.batch["returns"][1], torch.tensor([-0.25, 0.0, 0.0, 0.0]))


def test_matpo_hook_rejects_duplicate_reqs_id():
    data = DataProto(
        batch=TensorDict(
            {
                "token_level_rewards": torch.zeros(3, 2),
                "response_mask": torch.tensor([[1, 1], [1, 1], [1, 0]], dtype=torch.long),
            },
            batch_size=[3],
        ),
        non_tensor_batch={
            "uid": np.array(["task", "task", "task"], dtype=object),
            "reqs_id": np.array(["main", "main", "child"], dtype=object),
            "parent_reqs_id": np.array(["", "", "main"], dtype=object),
            "is_from_subagent_tool": np.array([False, False, True], dtype=object),
            "traj_uid": np.array(["t0", "t1", "t2"], dtype=object),
        },
    )

    with pytest.raises(MATPOParentChildIntegrityError, match="Duplicate reqs_id"):
        MATPOParentBroadcastHooks().compute_advantage(data, adv_estimator="grpo", fallback=_unreachable_fallback)


def test_matpo_hook_rejects_orphan_child_with_empty_parent_reqs_id():
    data = DataProto(
        batch=TensorDict(
            {
                "token_level_rewards": torch.zeros(2, 2),
                "response_mask": torch.tensor([[1, 1], [1, 0]], dtype=torch.long),
            },
            batch_size=[2],
        ),
        non_tensor_batch={
            "uid": np.array(["task", "task"], dtype=object),
            "reqs_id": np.array(["main", "child"], dtype=object),
            "parent_reqs_id": np.array(["", ""], dtype=object),
            "is_from_subagent_tool": np.array([False, True], dtype=object),
            "traj_uid": np.array(["t0", "t1"], dtype=object),
        },
    )

    with pytest.raises(MATPOParentChildIntegrityError, match="Orphan child"):
        MATPOParentBroadcastHooks().compute_advantage(data, adv_estimator="grpo", fallback=_unreachable_fallback)


def test_matpo_hook_rejects_orphan_child_with_unresolvable_parent_reqs_id():
    data = DataProto(
        batch=TensorDict(
            {
                "token_level_rewards": torch.zeros(2, 2),
                "response_mask": torch.tensor([[1, 1], [1, 0]], dtype=torch.long),
            },
            batch_size=[2],
        ),
        non_tensor_batch={
            "uid": np.array(["task", "task"], dtype=object),
            "reqs_id": np.array(["main", "child"], dtype=object),
            "parent_reqs_id": np.array(["", "does_not_exist"], dtype=object),
            "is_from_subagent_tool": np.array([False, True], dtype=object),
            "traj_uid": np.array(["t0", "t1"], dtype=object),
        },
    )

    with pytest.raises(MATPOParentChildIntegrityError, match="no row in this batch has that reqs_id"):
        MATPOParentBroadcastHooks().compute_advantage(data, adv_estimator="grpo", fallback=_unreachable_fallback)


def test_matpo_hook_rejects_self_referencing_row():
    data = DataProto(
        batch=TensorDict(
            {
                "token_level_rewards": torch.zeros(2, 2),
                "response_mask": torch.tensor([[1, 1], [1, 0]], dtype=torch.long),
            },
            batch_size=[2],
        ),
        non_tensor_batch={
            "uid": np.array(["task", "task"], dtype=object),
            "reqs_id": np.array(["main", "child"], dtype=object),
            "parent_reqs_id": np.array(["", "child"], dtype=object),
            "is_from_subagent_tool": np.array([False, True], dtype=object),
            "traj_uid": np.array(["t0", "t1"], dtype=object),
        },
    )

    with pytest.raises(MATPOParentChildIntegrityError, match="Self-referencing"):
        MATPOParentBroadcastHooks().compute_advantage(data, adv_estimator="grpo", fallback=_unreachable_fallback)


def test_matpo_hook_rejects_chained_delegation_where_parent_is_itself_a_child():
    # main -> child_a -> child_b: child_b's parent (child_a) is itself a child,
    # which violates MATPO's single-level planner/worker protocol.
    data = DataProto(
        batch=TensorDict(
            {
                "token_level_rewards": torch.zeros(3, 2),
                "response_mask": torch.tensor([[1, 1], [1, 1], [1, 0]], dtype=torch.long),
            },
            batch_size=[3],
        ),
        non_tensor_batch={
            "uid": np.array(["task", "task", "task"], dtype=object),
            "reqs_id": np.array(["main", "child_a", "child_b"], dtype=object),
            "parent_reqs_id": np.array(["", "main", "child_a"], dtype=object),
            "is_from_subagent_tool": np.array([False, True, True], dtype=object),
            "traj_uid": np.array(["t0", "t1", "t2"], dtype=object),
        },
    )

    with pytest.raises(MATPOParentChildIntegrityError, match="single parent -> child level"):
        MATPOParentBroadcastHooks().compute_advantage(data, adv_estimator="grpo", fallback=_unreachable_fallback)


def test_matpo_hook_rejects_cyclic_main_row_reference():
    # Two "main" rows referencing each other via parent_reqs_id forms a cycle
    # that must be rejected even though neither row is marked as a child. A
    # third, legitimate child row is included so `is_child.any()` is True and
    # the validator actually runs instead of short-circuiting to fallback().
    data = DataProto(
        batch=TensorDict(
            {
                "token_level_rewards": torch.zeros(3, 2),
                "response_mask": torch.tensor([[1, 1], [1, 1], [1, 0]], dtype=torch.long),
            },
            batch_size=[3],
        ),
        non_tensor_batch={
            "uid": np.array(["task", "task", "task"], dtype=object),
            "reqs_id": np.array(["main_a", "main_b", "child_a"], dtype=object),
            "parent_reqs_id": np.array(["main_b", "main_a", "main_a"], dtype=object),
            "is_from_subagent_tool": np.array([False, False, True], dtype=object),
            "traj_uid": np.array(["t0", "t1", "t2"], dtype=object),
        },
    )

    with pytest.raises(MATPOParentChildIntegrityError, match="Cyclic MATPO parent/child reference"):
        MATPOParentBroadcastHooks().compute_advantage(data, adv_estimator="grpo", fallback=_unreachable_fallback)
