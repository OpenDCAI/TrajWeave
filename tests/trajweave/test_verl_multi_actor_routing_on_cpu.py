from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from trajweave.backends.verl.routing import safe_actor_role_key, safe_worker_role_key, split_batch_by_group_values


class FakeKVBatch:
    def __init__(self, keys, tags=None, partition_id="train", extra_info=None):
        self.keys = list(keys)
        self.tags = list(tags or [{} for _ in self.keys])
        self.partition_id = partition_id
        self.fields = None
        self.extra_info = dict(extra_info or {})

    def select_keys(self, keys):
        indexes = [self.keys.index(key) for key in keys]
        return FakeKVBatch(
            keys,
            tags=[self.tags[index] for index in indexes],
            partition_id=self.partition_id,
            extra_info=self.extra_info,
        )


def test_split_batch_by_group_values_preserves_key_order():
    batch = FakeKVBatch(["k0", "k1", "k2", "k3"])

    routed = split_batch_by_group_values(batch, ["model_a", "model_b", "model_a", "model_b"])

    assert [item.group_id for item in routed] == ["model_a", "model_b"]
    assert routed[0].batch.keys == ["k0", "k2"]
    assert routed[1].batch.keys == ["k1", "k3"]


def test_split_batch_by_group_values_rejects_missing_values():
    batch = FakeKVBatch(["k0", "k1"])

    with pytest.raises(ValueError, match="must match batch keys"):
        split_batch_by_group_values(batch, ["model_a"])


def test_safe_worker_role_key_sanitizes_model_ids():
    assert safe_worker_role_key("qwen2.5/0.5b") == "maporl_actor_qwen2_5_0_5b"


def test_safe_actor_role_key_is_recipe_neutral():
    assert safe_actor_role_key("qwen2.5/0.5b") == "trajweave_actor_qwen2_5_0_5b"


def test_trajweave_maporl_multi_actor_trainer_is_registered():
    pytest.importorskip("transfer_queue")
    from trajweave.backends.verl.trainers import register_trajweave_trainers
    from verl.trainer.ppo.v1 import get_trainer_cls

    register_trajweave_trainers()

    trainer_cls = get_trainer_cls("trajweave_maporl_multi_actor_sync")
    assert trainer_cls.__name__ == "TrajWeaveMAPoRLMultiActorSyncTrainer"


def test_recipe_neutral_multi_actor_trainer_is_registered():
    pytest.importorskip("transfer_queue")
    from trajweave.backends.verl.trainers import register_trajweave_trainers
    from verl.trainer.ppo.v1 import get_trainer_cls

    register_trajweave_trainers()

    trainer_cls = get_trainer_cls("trajweave_multi_actor_sync")
    assert trainer_cls.__name__ == "TrajWeaveMultiActorSyncTrainer"


def test_multi_actor_balance_pads_each_worker_group_independently():
    tq = pytest.importorskip("transfer_queue")
    from transfer_queue import KVBatchMeta

    from trajweave.backends.verl.schema import to_python
    from trajweave.backends.verl.trainers.multi_actor_sync import TrajWeaveMultiActorSyncTrainer
    from verl.utils.tensordict_utils import list_of_dict_to_tensordict

    keys = ["a0", "b0", "b1", "b2"]
    groups = ["actor-a", "actor-b", "actor-b", "actor-b"]
    rows = []
    tags = []
    for key, group in zip(keys, groups, strict=True):
        rows.append(
            {
                "uid": key,
                "prompts": torch.tensor([1]),
                "responses": torch.tensor([2]),
                "input_ids": torch.tensor([1, 2]),
                "attention_mask": torch.tensor([1, 1]),
                "position_ids": torch.tensor([0, 1]),
                "response_mask": torch.tensor([1]),
                "loss_mask": torch.tensor([1]),
                "rm_scores": torch.tensor([0.0]),
                "rollout_log_probs": torch.tensor([0.0]),
                "num_turns": 1,
                "worker_group": group,
            }
        )
        tags.append({"prompt_len": 1, "response_len": 1, "seq_len": 2})

    trainer = object.__new__(TrajWeaveMultiActorSyncTrainer)
    trainer.actor_rollout_wgs = {
        "actor-a": SimpleNamespace(
            world_size=4,
            _dispatch_info={"actor": [0, 0, 1, 1]},
            _query_dispatch_info=lambda _role: [0, 0, 1, 1],
        ),
        "actor-b": SimpleNamespace(world_size=2),
    }
    trainer.tokenizer = SimpleNamespace(eos_token_id=0)
    batch = KVBatchMeta(
        keys=keys,
        tags=tags,
        partition_id="multi-actor-balance",
        fields=list(rows[0]),
        extra_info={},
    )

    tq.init()
    try:
        tq.kv_batch_put(
            keys=keys,
            partition_id=batch.partition_id,
            fields=list_of_dict_to_tensordict(rows),
            tags=tags,
        )
        balanced = trainer._balance_batch(batch, metrics={})
        fields = tq.kv_batch_get(
            keys=balanced.keys,
            partition_id=batch.partition_id,
            select_fields=["worker_group"],
        )
        balanced_groups = [str(to_python(fields["worker_group"][row])) for row in range(len(balanced.keys))]

        assert balanced_groups.count("actor-a") == 2
        assert balanced_groups.count("actor-b") == 4
        assert balanced_groups == ["actor-a"] * 2 + ["actor-b"] * 4
        assert len(balanced.keys) == 6
        assert {route.group_id: len(route.batch.keys) for route in trainer._route_batch(balanced)} == {
            "actor-a": 2,
            "actor-b": 4,
        }
    finally:
        tq.kv_clear(keys=balanced.keys if "balanced" in locals() else keys, partition_id=batch.partition_id)
        tq.close()
