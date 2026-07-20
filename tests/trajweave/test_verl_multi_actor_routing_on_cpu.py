from __future__ import annotations

import pytest

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
