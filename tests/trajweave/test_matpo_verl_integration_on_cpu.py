import numpy as np
import torch
from tensordict import TensorDict

from trajweave.backends.verl.dataproto import VerlDataProtoAdapter
from trajweave.backends.verl.extensions.common.hooks import MATPOParentBroadcastHooks, extension_hooks_for_config
from trajweave.backends.verl.extensions.registry import _extension_names
from trajweave.recipes.matpo import run_smoke
from verl.protocol import DataProto


def test_matpo_dataproto_contains_parent_child_fields():
    _, result = run_smoke(rollouts_per_task=1, max_turns=3)
    data = VerlDataProtoAdapter().build(result.samples)

    assert "is_from_subagent_tool" in data.batch
    assert "turn_count" in data.batch
    assert "reqs_id" in data.non_tensor_batch
    assert "parent_reqs_id" in data.non_tensor_batch
    assert data.batch["is_from_subagent_tool"].any()


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
