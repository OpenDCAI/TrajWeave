from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def test_preference_trainer_precomputes_detached_other_deltas_and_updates_in_agent_order():
    tq = pytest.importorskip("transfer_queue")
    from transfer_queue import KVBatchMeta

    from trajweave.backends.verl.schema import to_python
    from trajweave.backends.verl.trainers.multi_actor_preference_sync import (
        TrajWeaveMultiActorPreferenceSyncTrainer,
    )
    from verl.utils.tensordict_utils import list_of_dict_to_tensordict

    rows = []
    values = {
        ("actor-a", "chosen"): 2.0,
        ("actor-a", "rejected"): 1.0,
        ("actor-b", "chosen"): 4.0,
        ("actor-b", "rejected"): 1.5,
    }
    for worker_group in ("actor-b", "actor-a"):
        for side in ("chosen", "rejected"):
            rows.append(
                {
                    "log_probs": torch.tensor([values[(worker_group, side)]]),
                    "response_mask": torch.ones(1, dtype=torch.long),
                    "worker_group": worker_group,
                    "preference_pair_id": "pair-0",
                    "preference_side": side,
                    "chosen_reward": 1.0,
                    "rejected_reward": 0.0,
                    "preference_loss_mask": 1.0,
                }
            )
    keys = [f"madpo-row-{index}" for index in range(4)]
    batch = KVBatchMeta(keys=keys, tags=[{"seq_len": 1}] * 4, partition_id="train")
    trainer = object.__new__(TrajWeaveMultiActorPreferenceSyncTrainer)
    trainer.multi_actor_trainable_group_ids = ["actor-a", "actor-b"]
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            actor=SimpleNamespace(ppo_epochs=1, data_loader_seed=0),
            rollout=SimpleNamespace(temperature=1.0),
        )
    )
    update_order = []

    class FakeWorkerGroup:
        world_size = 1

        def __init__(self, group_id):
            self.group_id = group_id

        @staticmethod
        def compute_log_prob(routed_batch):
            return routed_batch

        def update_actor(self, routed_batch):
            update_order.append((self.group_id, list(routed_batch.keys), dict(routed_batch.extra_info)))
            return {"metrics": {}}

    trainer.actor_rollout_wgs = {
        "actor-a": FakeWorkerGroup("actor-a"),
        "actor-b": FakeWorkerGroup("actor-b"),
    }

    tq.init()
    try:
        tq.kv_batch_put(
            keys=keys,
            partition_id="train",
            fields=list_of_dict_to_tensordict(rows),
            tags=batch.tags,
        )
        metrics = {}
        trainer._compute_old_log_prob(batch, metrics=metrics)
        trainer._compute_advantage(batch, metrics=metrics)
        output = tq.kv_batch_get(
            keys=keys,
            partition_id="train",
            select_fields=[
                "madpo_pair_index",
                "madpo_preference_side",
                "madpo_other_agent_delta",
            ],
        )
        assert [int(to_python(output["madpo_pair_index"][row])) for row in range(4)] == [0, 0, 0, 0]
        assert [float(to_python(output["madpo_preference_side"][row])) for row in range(4)] == [
            1.0,
            -1.0,
            1.0,
            -1.0,
        ]
        assert [float(to_python(output["madpo_other_agent_delta"][row])) for row in range(4)] == pytest.approx(
            [1.0, 1.0, 2.5, 2.5]
        )

        trainer._update_actor(batch, metrics)
        assert [group_id for group_id, _keys, _extra in update_order] == ["actor-a", "actor-b"]
        assert [routed_keys for _group, routed_keys, _extra in update_order] == [keys[2:], keys[:2]]
        assert all(extra["force_group_size"] == 2 for _group, _keys, extra in update_order)
        assert metrics["trajweave/madpo/actor_groups/updated"] == 2
        assert metrics["trajweave/madpo/pair_count"] == 1
    finally:
        tq.kv_clear(keys=keys, partition_id="train")
        tq.close()


def test_verl_microbatch_force_group_size_keeps_preference_pairs_together():
    from tensordict import TensorDict

    from verl.utils import tensordict_utils as tu
    from verl.workers.engine.utils import prepare_micro_batches

    data = TensorDict(
        {"pair_row": torch.tensor([0, 0, 1, 1])},
        batch_size=4,
    )
    tu.assign_non_tensor(
        data,
        use_dynamic_bsz=False,
        force_group_size=2,
        micro_batch_size_per_gpu=1,
    )

    micro_batches, _indices = prepare_micro_batches(data)

    assert [micro_batch["pair_row"].tolist() for micro_batch in micro_batches] == [[0, 0], [1, 1]]


def test_preference_trainer_registers_through_unified_entrypoint():
    pytest.importorskip("transfer_queue")
    from trajweave.backends.verl.trainers import register_trajweave_trainers
    from verl.trainer.ppo.v1.trainer_base import get_trainer_cls

    register_trajweave_trainers()
    assert get_trainer_cls("trajweave_multi_actor_preference_sync").__name__ == (
        "TrajWeaveMultiActorPreferenceSyncTrainer"
    )
