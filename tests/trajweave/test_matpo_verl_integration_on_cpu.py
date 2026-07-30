import numpy as np
import pytest
import torch
from tensordict import TensorDict

from trajweave.backends.verl.batch_padding import pad_session_batch
from trajweave.backends.verl.dataproto import VerlDataProtoAdapter
from trajweave.backends.verl.emitters.matpo import MATPOEmitterMixin
from trajweave.backends.verl.extensions.common.hooks import extension_hooks_for_config
from trajweave.backends.verl.extensions.matpo import MATPOParentBroadcastHooks, MATPOParentChildIntegrityError
from trajweave.backends.verl.extensions.registry import _extension_names
from trajweave.credit.matpo.parent_broadcast import combined_matpo_reward
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


def test_matpo_padding_remaps_copied_parent_child_ids_before_advantage_broadcast():
    fields = [
        {
            "uid": "task",
            "traj_uid": "main",
            "worker_group": "planner",
            "reqs_id": "planner-0",
            "parent_reqs_id": "",
            "is_from_subagent_tool": False,
            "response_mask": torch.tensor([1, 1]),
            "loss_mask": torch.tensor([1, 1]),
            "rm_scores": torch.tensor([1.0, 0.0]),
            "extra_fields": {"reqs_id": "planner-0", "parent_reqs_id": "", "is_from_subagent_tool": False},
        },
        {
            "uid": "task",
            "traj_uid": "child",
            "worker_group": "worker",
            "reqs_id": "worker-0",
            "parent_reqs_id": "planner-0",
            "is_from_subagent_tool": True,
            "response_mask": torch.tensor([1, 0]),
            "loss_mask": torch.tensor([1, 0]),
            "rm_scores": torch.tensor([0.0, 0.0]),
            "extra_fields": {"reqs_id": "worker-0", "parent_reqs_id": "planner-0", "is_from_subagent_tool": True},
        },
    ]
    keys = ["task_0_0", "task_0_1"]
    tags = [{"status": "success", "response_len": 2}, {"status": "success", "response_len": 1}]

    pad_session_batch(keys=keys, fields=fields, tags=tags, multiple=2, uid="task", session_id=0)

    assert len(fields) == 4
    assert len({field["reqs_id"] for field in fields}) == 4
    padded_parent = next(field for field in fields if field["reqs_id"].startswith("planner-0__trajweave_pad__"))
    padded_child = next(field for field in fields if field["reqs_id"].startswith("worker-0__trajweave_pad__"))
    assert padded_child["parent_reqs_id"] == padded_parent["reqs_id"]
    assert padded_child["extra_fields"]["parent_reqs_id"] == padded_parent["extra_fields"]["reqs_id"]
    assert padded_child["response_mask"].sum().item() == 0

    data = DataProto(
        batch=TensorDict(
            {
                "token_level_rewards": torch.stack([field["rm_scores"] for field in fields]),
                "response_mask": torch.stack([field["response_mask"] for field in fields]),
            },
            batch_size=[len(fields)],
        ),
        non_tensor_batch={
            "uid": np.array([field["uid"] for field in fields], dtype=object),
            "traj_uid": np.array([field["traj_uid"] for field in fields], dtype=object),
            "reqs_id": np.array([field["reqs_id"] for field in fields], dtype=object),
            "parent_reqs_id": np.array([field["parent_reqs_id"] for field in fields], dtype=object),
            "is_from_subagent_tool": np.array([field["is_from_subagent_tool"] for field in fields], dtype=object),
            "turn_count": np.arange(len(fields), dtype=object),
        },
    )

    def fallback(main_data, **kwargs):
        main_data.batch["advantages"] = torch.full_like(main_data.batch["token_level_rewards"], 0.25)
        main_data.batch["returns"] = main_data.batch["advantages"].clone()
        return main_data

    output = MATPOParentBroadcastHooks().compute_advantage(data, adv_estimator="grpo", fallback=fallback)
    assert torch.equal(output.batch["advantages"][1], torch.tensor([0.25, 0.0]))
    assert torch.equal(output.batch["advantages"][3], torch.zeros(2))


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
            "traj_uid": np.array(["trajectory", "trajectory"], dtype=object),
            "turn_count": np.array([0, 1], dtype=object),
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
            "traj_uid": np.array(["trajectory-a", "trajectory-b", "trajectory-a"], dtype=object),
            "turn_count": np.array([0, 0, 1], dtype=object),
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
            "traj_uid": np.array(["trajectory", "trajectory"], dtype=object),
            "turn_count": np.array([0, 1], dtype=object),
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
            "traj_uid": np.array(["trajectory", "trajectory"], dtype=object),
            "turn_count": np.array([0, 1], dtype=object),
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
            "turn_count": np.array([0, 1, 2], dtype=object),
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
            "turn_count": np.array([0, 1], dtype=object),
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
            "turn_count": np.array([0, 1], dtype=object),
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
            "turn_count": np.array([0, 1], dtype=object),
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
            "turn_count": np.array([0, 1, 2], dtype=object),
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
            "turn_count": np.array([0, 1, 2], dtype=object),
        },
    )

    with pytest.raises(MATPOParentChildIntegrityError, match="Cyclic MATPO parent/child reference"):
        MATPOParentBroadcastHooks().compute_advantage(data, adv_estimator="grpo", fallback=_unreachable_fallback)


def test_combined_matpo_reward_uses_strict_weighted_formula():
    assert combined_matpo_reward(accuracy=1.0, planner_format=1.0, worker_formats=(1.0, 0.0)) == pytest.approx(0.975)
    assert combined_matpo_reward(accuracy=0.0, planner_format=1.0, worker_formats=(1.0,)) == pytest.approx(0.1)
    with pytest.raises(ValueError, match="between 0 and 1"):
        combined_matpo_reward(accuracy=1.1, planner_format=1.0, worker_formats=(1.0,))
    with pytest.raises(ValueError, match="must sum to 1.0"):
        combined_matpo_reward(
            accuracy=1.0,
            planner_format=1.0,
            worker_formats=(1.0,),
            accuracy_reward_weight=0.8,
            tool_format_reward_weight=0.1,
        )


class _SyntheticMATPOEmitter(MATPOEmitterMixin):
    config = {
        "agent": {
            "orchestra": {
                "matpo": {
                    "planner_agent": "planner",
                    "worker_agent": "browsing_agent",
                    "tool_name": "search_and_browse",
                    "accuracy_reward_weight": 0.9,
                    "tool_format_reward_weight": 0.1,
                }
            }
        }
    }

    @staticmethod
    def _encode_prompt(raw_prompt):
        return [1, 2]

    @staticmethod
    def _encode_text(text):
        return list(text.encode("utf-8"))


def test_matpo_synthetic_emitter_mirrors_worker_call_summary_and_final_reward():
    emitter = _SyntheticMATPOEmitter()
    prompt = {
        "uid": "prompt-a",
        "raw_prompt": [{"role": "user", "content": "Question"}],
        "reward_model": {"ground_truth": "Paris"},
        "extra_info": {"search_query": "capital France"},
    }
    outputs = emitter._build_matpo_browse_outputs(prompt, session_id=0)

    assert [output.extra_fields["matpo_turn_role"] for output in outputs] == [
        "delegate",
        "worker_call",
        "worker_summary",
        "final",
    ]
    assert [output.reward_score for output in outputs] == [1.0, 1.0, 1.0, 1.0]
    reqs_ids = [output.extra_fields["reqs_id"] for output in outputs]
    assert len(reqs_ids) == len(set(reqs_ids))
    assert all("prompt-a" in reqs_id and "session:0" in reqs_id for reqs_id in reqs_ids)
    assert outputs[1].extra_fields["parent_reqs_id"] == outputs[0].extra_fields["reqs_id"]
    assert outputs[2].extra_fields["parent_reqs_id"] == outputs[0].extra_fields["reqs_id"]
    assert [output.extra_fields["response_text"] for output in outputs] == [
        "CALL browsing_agent: capital France",
        "CALL search_and_browse: capital France",
        "Evidence summary: Offline evidence for capital France: Paris. Suggested final answer: Paris",
        "Final answer: Paris",
    ]
    assert all(output.extra_fields["prompt_text"] for output in outputs)
    assert all(output.extra_fields["observation_text"] for output in outputs)
    assert {output.extra_fields["final_answer"] for output in outputs} == {"Final answer: Paris"}
    assert all(output.extra_fields["workflow_success"] is True for output in outputs)

    wrong_outputs = emitter._build_matpo_browse_outputs(prompt, session_id=1)
    assert [output.reward_score for output in wrong_outputs] == pytest.approx([0.1, 0.1, 0.1, 0.1])
    assert all(output.extra_fields["workflow_success"] is False for output in wrong_outputs)
    assert {output.extra_fields["reqs_id"] for output in outputs}.isdisjoint(
        output.extra_fields["reqs_id"] for output in wrong_outputs
    )


def _minimal_hook_data(*, markers=True, reqs=True, parents=True, child_values=(False,)):
    non_tensor_batch = {
        "uid": np.array(["task"] * len(child_values), dtype=object),
        "traj_uid": np.array([f"trajectory-{index}" for index in range(len(child_values))], dtype=object),
        "turn_count": np.arange(len(child_values), dtype=object),
    }
    if markers:
        non_tensor_batch["is_from_subagent_tool"] = np.array(child_values, dtype=object)
    if reqs:
        non_tensor_batch["reqs_id"] = np.array([f"row-{index}" for index in range(len(child_values))], dtype=object)
    if parents:
        non_tensor_batch["parent_reqs_id"] = np.array(
            ["" if not child else "missing-main" for child in child_values], dtype=object
        )
    return DataProto(
        batch=TensorDict(
            {
                "token_level_rewards": torch.zeros(len(child_values), 2),
                "response_mask": torch.ones(len(child_values), 2, dtype=torch.long),
            },
            batch_size=[len(child_values)],
        ),
        non_tensor_batch=non_tensor_batch,
    )


def test_matpo_hook_fails_closed_for_missing_marker_and_id_arrays():
    hooks = MATPOParentBroadcastHooks()
    with pytest.raises(MATPOParentChildIntegrityError, match="missing the is_from_subagent_tool"):
        hooks.compute_advantage(_minimal_hook_data(markers=False), adv_estimator="grpo", fallback=_unreachable_fallback)
    with pytest.raises(MATPOParentChildIntegrityError, match="missing reqs_id or parent_reqs_id"):
        hooks.compute_advantage(_minimal_hook_data(reqs=False), adv_estimator="grpo", fallback=_unreachable_fallback)
    with pytest.raises(MATPOParentChildIntegrityError, match="missing reqs_id or parent_reqs_id"):
        hooks.compute_advantage(_minimal_hook_data(parents=False), adv_estimator="grpo", fallback=_unreachable_fallback)


def test_matpo_hook_rejects_all_child_batch():
    data = _minimal_hook_data(child_values=(True, True))
    data.non_tensor_batch["parent_reqs_id"] = np.array(["row-1", "row-0"], dtype=object)
    with pytest.raises(MATPOParentChildIntegrityError):
        MATPOParentBroadcastHooks().compute_advantage(data, adv_estimator="grpo", fallback=_unreachable_fallback)


def test_matpo_hook_collapses_all_main_rows_to_equal_weight_trajectories():
    data = DataProto(
        batch=TensorDict(
            {
                "token_level_rewards": torch.tensor([[0.0], [1.0], [1.0]]),
                "response_mask": torch.ones(3, 1, dtype=torch.long),
            },
            batch_size=[3],
        ),
        non_tensor_batch={
            "uid": np.array(["task"] * 3, dtype=object),
            "traj_uid": np.array(["trajectory-low", "trajectory-high", "trajectory-high"], dtype=object),
            "turn_count": np.array([0, 0, 1], dtype=object),
            "reqs_id": np.array(["low", "high-early", "high-late"], dtype=object),
            "parent_reqs_id": np.array(["", "", ""], dtype=object),
            "is_from_subagent_tool": np.array([False, False, False], dtype=object),
        },
    )
    fallback_batch_keys = []

    def fallback(representative_data, *, batch_keys, **kwargs):
        fallback_batch_keys.extend(batch_keys)
        scores = representative_data.batch["token_level_rewards"].sum(dim=-1)
        normalized = (scores - scores.mean()) / scores.std(unbiased=True)
        representative_data.batch["advantages"] = normalized.unsqueeze(-1)
        representative_data.batch["returns"] = representative_data.batch["advantages"].clone()
        return representative_data

    output = MATPOParentBroadcastHooks().compute_advantage(
        data,
        adv_estimator="grpo",
        fallback=fallback,
        batch_keys=["low", "high-early", "high-late"],
    )

    assert fallback_batch_keys == ["low", "high-late"]
    assert output.batch["advantages"].squeeze(-1).tolist() == pytest.approx([-0.70710678, 0.70710678, 0.70710678])


@pytest.mark.parametrize("field", ["traj_uid", "turn_count"])
def test_matpo_hook_rejects_trajectory_metadata_length_mismatch(field):
    data = _minimal_hook_data(child_values=(False, False))
    data.non_tensor_batch[field] = data.non_tensor_batch[field][:1]

    with pytest.raises(MATPOParentChildIntegrityError, match="traj_uid and turn_count arrays"):
        MATPOParentBroadcastHooks().compute_advantage(data, adv_estimator="grpo", fallback=_unreachable_fallback)


def test_matpo_hook_retains_valid_all_main_fallback():
    data = _minimal_hook_data(child_values=(False, False))
    fallback_calls = []

    def fallback(main_data, **kwargs):
        fallback_calls.append(len(main_data))
        main_data.batch["advantages"] = torch.full_like(main_data.batch["token_level_rewards"], 0.5)
        main_data.batch["returns"] = main_data.batch["advantages"].clone()
        return main_data

    output = MATPOParentBroadcastHooks().compute_advantage(data, adv_estimator="grpo", fallback=fallback)
    assert fallback_calls == [2]
    assert torch.equal(output.batch["advantages"], torch.full((2, 2), 0.5))
