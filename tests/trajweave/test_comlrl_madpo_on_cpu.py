from __future__ import annotations

import math
import uuid
from types import SimpleNamespace

import pytest
import torch

from trajweave.backends.verl.extensions.comlrl.preference import (
    madpo_microbatch_loss,
    materialize_pair_response_logprobs,
    materialize_tq_preference_pairs,
    validate_preference_pairs,
)
from trajweave.credit.comlrl.madpo import (
    madpo_actor_loss,
    preference_logprob_delta,
    sequence_logprob_sum,
    snapshot_detached_deltas,
)


def _preference_fields(*, include_padding: bool = True):
    fields = {
        "preference_pair_id": ["pair-0"] * 4,
        "preference_side": ["chosen", "rejected", "chosen", "rejected"],
        "chosen_reward": [2.0] * 4,
        "rejected_reward": [1.0] * 4,
        "preference_loss_mask": [1.0] * 4,
        "worker_group": ["actor-a", "actor-a", "actor-b", "actor-b"],
        "response_mask": torch.tensor([[1, 1], [1, 0], [1, 1], [1, 1]], dtype=torch.long),
        "log_probs": torch.tensor([[-0.1, -0.2], [-0.5, 99.0], [-0.2, -0.3], [-0.7, -0.4]]),
    }
    if include_padding:
        for field, value in (
            ("preference_pair_id", "__padding__:pair"),
            ("preference_side", "__padding__"),
            ("chosen_reward", 0.0),
            ("rejected_reward", 0.0),
            ("preference_loss_mask", 0.0),
            ("worker_group", "actor-a"),
        ):
            fields[field].append(value)
        fields["response_mask"] = torch.cat([fields["response_mask"], torch.zeros(1, 2, dtype=torch.long)])
        fields["log_probs"] = torch.cat([fields["log_probs"], torch.full((1, 2), float("nan"))])
    return fields


def test_sequence_logprob_oracle_strict_mask_and_last_token():
    log_probs = torch.tensor([[math.log(0.5), math.log(0.25), float("nan")], [99.0, math.log(0.2), math.log(0.1)]])
    mask = torch.tensor([[1, 1, 0], [0, 1, 1]])

    actual = sequence_logprob_sum(log_probs, mask)

    assert actual.tolist() == pytest.approx([math.log(0.125), math.log(0.02)])
    assert actual[1].item() == pytest.approx(log_probs[1, 1:].sum().item())
    with pytest.raises(ValueError, match="empty completions"):
        sequence_logprob_sum(torch.zeros(1, 2), torch.zeros(1, 2))


def test_preference_delta_beta_oracle_and_gradient_only_through_own_actor():
    chosen = torch.tensor([[-0.1, -0.2]], requires_grad=True)
    rejected = torch.tensor([[-0.6, -0.4]], requires_grad=True)
    own_delta = preference_logprob_delta(chosen, torch.ones_like(chosen), rejected, torch.ones_like(rejected))
    other_source = torch.tensor([0.25], requires_grad=True)
    snapshot = snapshot_detached_deltas({"actor-a": own_delta, "actor-b": other_source})

    loss = madpo_actor_loss(own_delta, snapshot, actor_id="actor-a", beta=0.2)

    expected_joint_delta = (-0.3 - -1.0) + 0.25
    expected_loss = -torch.nn.functional.logsigmoid(torch.tensor(0.2 * expected_joint_delta)).item()
    assert loss.item() == pytest.approx(expected_loss)
    loss.backward()
    assert chosen.grad is not None and chosen.grad.abs().sum().item() > 0
    assert rejected.grad is not None and rejected.grad.abs().sum().item() > 0
    assert other_source.grad is None
    with pytest.raises(ValueError, match="beta"):
        madpo_actor_loss(own_delta.detach(), snapshot, actor_id="actor-a", beta=0)


def test_snapshot_is_detached_cloned_and_taken_before_sequential_mutation():
    actor_a = torch.tensor([1.0], requires_grad=True)
    actor_b = torch.tensor([2.0], requires_grad=True)
    snapshot = snapshot_detached_deltas({"actor-a": actor_a, "actor-b": actor_b})

    with torch.no_grad():
        actor_a.add_(10)
        actor_b.sub_(5)

    assert snapshot["actor-a"].item() == 1.0
    assert snapshot["actor-b"].item() == 2.0
    assert all(not value.requires_grad and value.grad_fn is None for value in snapshot.values())


def test_non_finite_inputs_validate_snapshots_and_keep_reference_helper_fallback():
    with pytest.raises(FloatingPointError, match="finite"):
        sequence_logprob_sum(torch.tensor([[float("nan")]]), torch.ones(1, 1))
    with pytest.raises(FloatingPointError, match="finite"):
        snapshot_detached_deltas({"actor-a": torch.tensor([float("inf")])})

    current = torch.tensor([float("inf")], requires_grad=True)
    loss = madpo_actor_loss(
        current,
        {"actor-a": torch.tensor([0.0]), "actor-b": torch.tensor([float("-inf")])},
        actor_id="actor-a",
        beta=0.1,
    )
    assert loss.item() == pytest.approx(0.1)
    loss.backward()
    assert current.grad is None


def test_microbatch_loss_accumulates_to_global_pair_mean_and_padding_backward():
    beta = 0.2
    log_probs = torch.tensor(
        [[-0.1], [-0.7], [-0.2], [-0.5], [-0.3], [-0.8], [-0.1], [-0.4]],
        requires_grad=True,
    )
    mask = torch.ones_like(log_probs, dtype=torch.long)
    pair_indices = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    side = torch.tensor([1.0, -1.0] * 4)
    other = torch.tensor([0.2, 0.2, -0.1, -0.1, 0.3, 0.3, 0.0, 0.0])
    active = torch.ones(8)

    whole_loss, whole_count = madpo_microbatch_loss(
        log_probs,
        mask,
        pair_indices,
        side,
        other,
        active,
        beta=beta,
        global_pair_count=4,
    )
    whole_loss.backward()
    whole_grad = log_probs.grad.detach().clone()

    split_log_probs = log_probs.detach().clone().requires_grad_(True)
    first_loss, first_count = madpo_microbatch_loss(
        split_log_probs[:4],
        mask[:4],
        pair_indices[:4],
        side[:4],
        other[:4],
        active[:4],
        beta=beta,
        global_pair_count=4,
    )
    second_loss, second_count = madpo_microbatch_loss(
        split_log_probs[4:],
        mask[4:],
        pair_indices[4:],
        side[4:],
        other[4:],
        active[4:],
        beta=beta,
        global_pair_count=4,
    )
    first_loss.backward()
    second_loss.backward()

    assert (whole_count, first_count, second_count) == (4, 2, 2)
    assert (first_loss + second_loss).item() == pytest.approx(whole_loss.item())
    torch.testing.assert_close(split_log_probs.grad, whole_grad)

    padding_log_probs = torch.tensor([[0.4], [-0.2]], requires_grad=True)
    padding_loss, padding_count = madpo_microbatch_loss(
        padding_log_probs,
        torch.zeros_like(padding_log_probs),
        torch.zeros(2, dtype=torch.long),
        torch.zeros(2),
        torch.zeros(2),
        torch.zeros(2),
        beta=beta,
        global_pair_count=4,
    )
    assert padding_count == 0
    assert padding_loss.item() == 0.0
    padding_loss.backward()
    torch.testing.assert_close(padding_log_probs.grad, torch.zeros_like(padding_log_probs))


def test_tq_pairing_excludes_padding_and_materializes_joint_deltas():
    fields = _preference_fields()

    materialized = materialize_pair_response_logprobs(
        fields,
        expected_worker_groups=("actor-a", "actor-b"),
    )

    assert [(pair.preference_pair_id, pair.worker_group) for pair in materialized.pairs] == [
        ("pair-0", "actor-a"),
        ("pair-0", "actor-b"),
    ]
    assert materialized.deltas[("pair-0", "actor-a")].item() == pytest.approx(0.2)
    assert materialized.deltas[("pair-0", "actor-b")].item() == pytest.approx(0.6)
    assert materialized.sequence_logps[-1].item() == 0.0


def test_tq_pairing_rejects_missing_actor_duplicate_side_invalid_side_and_reward_order():
    missing_actor = _preference_fields(include_padding=False)
    for field in missing_actor:
        missing_actor[field] = missing_actor[field][:2]
    with pytest.raises(ValueError, match="cover every trainable actor"):
        validate_preference_pairs(missing_actor, expected_worker_groups=("actor-a", "actor-b"))

    duplicate = _preference_fields(include_padding=False)
    duplicate["preference_side"][1] = "chosen"
    with pytest.raises(ValueError, match="duplicate"):
        validate_preference_pairs(duplicate, expected_worker_groups=("actor-a", "actor-b"))

    invalid = _preference_fields(include_padding=False)
    invalid["preference_side"][0] = "winner"
    with pytest.raises(ValueError, match="invalid preference_side"):
        validate_preference_pairs(invalid)

    reversed_reward = _preference_fields(include_padding=False)
    reversed_reward["chosen_reward"] = [0.0] * 4
    with pytest.raises(ValueError, match="strictly greater"):
        validate_preference_pairs(reversed_reward)


def test_real_transfer_queue_materialization_boundary():
    tq = pytest.importorskip("transfer_queue")
    from transfer_queue import KVBatchMeta

    from verl.utils.tensordict_utils import list_of_dict_to_tensordict

    fields = _preference_fields()
    rows = [
        {field: value[row] if isinstance(value, torch.Tensor) else value[row] for field, value in fields.items()}
        for row in range(5)
    ]
    keys = [f"madpo-{uuid.uuid4().hex}-{row}" for row in range(5)]
    batch = KVBatchMeta(keys=keys, tags=[{"seq_len": 2}] * 5, partition_id="train")
    tq.init()
    try:
        tq.kv_batch_put(
            keys=keys,
            partition_id="train",
            fields=list_of_dict_to_tensordict(rows),
            tags=batch.tags,
        )
        materialized = materialize_tq_preference_pairs(
            batch,
            expected_worker_groups=("actor-a", "actor-b"),
        )
        assert materialized.response_mask.shape == (5, 2)
        assert materialized.deltas[("pair-0", "actor-b")].item() == pytest.approx(0.6)
    finally:
        tq.kv_clear(keys=keys, partition_id="train")
        tq.close()


def test_all_tied_madpo_batch_is_a_safe_noop_with_parent_metric_fields():
    tq = pytest.importorskip("transfer_queue")
    from transfer_queue import KVBatchMeta

    from trajweave.backends.verl.trainers.joint_preference_sync import TrajWeaveJointPreferenceSyncTrainer
    from verl.utils.tensordict_utils import list_of_dict_to_tensordict

    rows = [
        {
            "preference_pair_id": f"all-tied-{group}",
            "preference_side": "__empty__",
            "chosen_reward": 0.0,
            "rejected_reward": 0.0,
            "preference_loss_mask": 0.0,
            "worker_group": group,
            "response_mask": torch.ones(1, dtype=torch.long),
        }
        for group in ("actor-a", "actor-b")
    ]
    keys = [f"all-tied-{uuid.uuid4().hex}-{row}" for row in range(len(rows))]
    batch = KVBatchMeta(keys=keys, tags=[{"seq_len": 2}] * len(rows), partition_id="train")
    trainer = object.__new__(TrajWeaveJointPreferenceSyncTrainer)
    trainer.multi_actor_trainable_group_ids = ["actor-a", "actor-b"]

    tq.init()
    try:
        tq.kv_batch_put(
            keys=keys,
            partition_id="train",
            fields=list_of_dict_to_tensordict(rows),
            tags=batch.tags,
        )
        metrics = {}
        balanced = trainer._balance_batch(batch, metrics)
        trainer._compute_old_log_prob(balanced, metrics)
        trainer._compute_advantage(balanced, metrics)
        trainer._update_actor(balanced, metrics)
        output = tq.kv_batch_get(
            keys=keys,
            partition_id="train",
            select_fields=["advantages", "returns"],
        )

        assert metrics["trajweave/madpo/skipped_all_tied_batch"] == 1
        assert metrics["trajweave/madpo/actor_groups/updated"] == 0
        assert torch.equal(output["advantages"].values(), torch.zeros_like(output["advantages"].values()))
        assert torch.equal(output["returns"].values(), torch.zeros_like(output["returns"].values()))
        assert torch.equal(output["advantages"].offsets(), output["returns"].offsets())
    finally:
        tq.kv_clear(keys=keys, partition_id="train")
        tq.close()


def test_madpo_balancing_keeps_pairs_atomic_per_actor_dp_rank():
    tq = pytest.importorskip("transfer_queue")
    from transfer_queue import KVBatchMeta

    from trajweave.backends.verl.schema import to_python
    from trajweave.backends.verl.trainers.joint_preference_sync import TrajWeaveJointPreferenceSyncTrainer
    from verl.utils.tensordict_utils import list_of_dict_to_tensordict

    rows = []
    for pair_index in range(3):
        for worker_group in ("actor-a", "actor-b"):
            for side in ("chosen", "rejected"):
                rows.append(
                    {
                        "preference_pair_id": f"pair-{pair_index}",
                        "preference_side": side,
                        "chosen_reward": 1.0,
                        "rejected_reward": 0.0,
                        "preference_loss_mask": 1.0,
                        "worker_group": worker_group,
                        "response_mask": torch.ones(1, dtype=torch.long),
                    }
                )
    keys = [f"madpo-balance-{uuid.uuid4().hex}-{row}" for row in range(len(rows))]
    batch = KVBatchMeta(keys=keys, tags=[{"seq_len": 2}] * len(rows), partition_id="train")

    events = []

    class FakeWorkerGroup:
        world_size = 2

        def __init__(self, group_id):
            self.group_id = group_id

        @staticmethod
        def compute_log_prob(routed_batch):
            fields = tq.kv_batch_get(
                keys=routed_batch.keys,
                partition_id=routed_batch.partition_id,
                select_fields=["response_mask"],
            )
            tq.kv_batch_put(
                keys=routed_batch.keys,
                partition_id=routed_batch.partition_id,
                fields=fields.select("response_mask")
                .apply(lambda value: torch.zeros_like(value, dtype=torch.float32))
                .rename_key_("response_mask", "log_probs"),
            )
            return routed_batch

        def update_actor(self, routed_batch):
            events.append((self.group_id, list(routed_batch.keys)))
            return {"metrics": {}}

    trainer = object.__new__(TrajWeaveJointPreferenceSyncTrainer)
    trainer.multi_actor_trainable_group_ids = ["actor-a", "actor-b"]
    trainer.actor_rollout_wgs = {
        "actor-a": FakeWorkerGroup("actor-a"),
        "actor-b": FakeWorkerGroup("actor-b"),
    }
    trainer.tokenizer = SimpleNamespace(eos_token_id=0)
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            actor=SimpleNamespace(data_loader_seed=0),
            rollout=SimpleNamespace(temperature=1.0),
        )
    )

    tq.init()
    try:
        tq.kv_batch_put(
            keys=keys,
            partition_id="train",
            fields=list_of_dict_to_tensordict(rows),
            tags=batch.tags,
        )
        metrics = {}
        balanced = trainer._balance_batch(batch, metrics)
        fields = tq.kv_batch_get(
            keys=balanced.keys,
            partition_id="train",
            select_fields=["worker_group", "preference_pair_id", "preference_loss_mask"],
        )
        groups = [str(to_python(fields["worker_group"][row])) for row in range(len(balanced.keys))]
        pair_ids = [str(to_python(fields["preference_pair_id"][row])) for row in range(len(balanced.keys))]
        active = [float(to_python(fields["preference_loss_mask"][row])) for row in range(len(balanced.keys))]

        assert groups == ["actor-a"] * 8 + ["actor-b"] * 8
        assert metrics == {
            "trajweave/madpo/padding_pairs": 2,
            "trajweave/madpo/balanced_rows": 16,
        }
        for group_start in (0, 8):
            for rank_start in (group_start, group_start + 4):
                rank_slice = slice(rank_start, rank_start + 4)
                rank_active_ids = [
                    pair_id
                    for pair_id, is_active in zip(pair_ids[rank_slice], active[rank_slice], strict=True)
                    if is_active
                ]
                assert rank_active_ids
                assert all(rank_active_ids.count(pair_id) == 2 for pair_id in set(rank_active_ids))
        routed = {item.group_id: item.batch for item in trainer._route_batch(balanced)}
        assert {group_id: len(routed_batch.keys) for group_id, routed_batch in routed.items()} == {
            "actor-a": 8,
            "actor-b": 8,
        }
        assert routed["actor-a"].keys == balanced.keys[:8]
        assert routed["actor-b"].keys == balanced.keys[8:]
        trainer._compute_old_log_prob(balanced, metrics={})
        trainer._compute_advantage(balanced, metrics={})
        trainer._update_actor(balanced, metrics={})
        assert [(group_id, len(group_keys)) for group_id, group_keys in events] == [
            ("actor-a", 8),
            ("actor-b", 8),
        ]
    finally:
        tq.kv_clear(keys=balanced.keys if "balanced" in locals() else keys, partition_id="train")
        tq.close()


def test_joint_preference_trainer_registration_and_dry_contract():
    pytest.importorskip("transfer_queue")
    from trajweave.backends.verl.trainers import register_trajweave_trainers
    from trajweave.backends.verl.trainers.joint_preference_sync import (
        validate_joint_preference_training_contract,
    )
    from verl.trainer.ppo.v1.trainer_base import get_trainer_cls

    register_trajweave_trainers()
    assert get_trainer_cls("trajweave_joint_preference_sync").__name__ == ("TrajWeaveJointPreferenceSyncTrainer")
    validate_joint_preference_training_contract(
        trainable_actor_groups=["actor-a", "actor-b"],
        use_critic=False,
        use_reference_policy=False,
        use_kl_in_reward=False,
        ppo_epochs=1,
        beta=0.1,
    )
    with pytest.raises(ValueError, match="ppo_epochs=1"):
        validate_joint_preference_training_contract(
            trainable_actor_groups=["actor-a", "actor-b"],
            use_critic=False,
            use_reference_policy=False,
            use_kl_in_reward=False,
            ppo_epochs=2,
            beta=0.1,
        )
    with pytest.raises(ValueError, match="reference-free"):
        validate_joint_preference_training_contract(
            trainable_actor_groups=["actor-a", "actor-b"],
            use_critic=False,
            use_reference_policy=True,
            use_kl_in_reward=False,
            ppo_epochs=1,
            beta=0.1,
        )
    with pytest.raises(ValueError, match="colocated reward model"):
        validate_joint_preference_training_contract(
            trainable_actor_groups=["actor-a", "actor-b"],
            use_critic=False,
            use_reference_policy=False,
            use_kl_in_reward=False,
            ppo_epochs=1,
            beta=0.1,
            reward_model_enabled=True,
        )


def test_update_actor_accepts_real_tensordict_and_preserves_padding_order():
    from transfer_queue import KVBatchMeta

    from trajweave.backends.verl.trainers.joint_preference_sync import TrajWeaveJointPreferenceSyncTrainer
    from verl.utils import tensordict_utils as tu

    events = []

    class FakeWorkerGroup:
        world_size = 1

        def __init__(self, group_id):
            self.group_id = group_id

        def update_actor(self, routed_batch):
            events.append((self.group_id, list(routed_batch.keys), dict(routed_batch.extra_info)))
            return tu.get_tensordict(tensor_dict={}, non_tensor_dict={"metrics": {"loss": [0.25]}})

    trainer = object.__new__(TrajWeaveJointPreferenceSyncTrainer)
    trainer.multi_actor_trainable_group_ids = ["actor-a", "actor-b"]
    trainer.actor_rollout_wgs = {
        "actor-a": FakeWorkerGroup("actor-a"),
        "actor-b": FakeWorkerGroup("actor-b"),
    }
    trainer._madpo_snapshot_deltas = {"actor-a": torch.tensor([0.0]), "actor-b": torch.tensor([0.0])}
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            actor=SimpleNamespace(data_loader_seed=0),
            rollout=SimpleNamespace(temperature=1.0),
        )
    )
    keys = ["a-chosen", "a-rejected", "a-pad-chosen", "a-pad-rejected", "b-chosen", "b-rejected"]
    batch = KVBatchMeta(
        keys=keys,
        tags=[{} for _ in keys],
        partition_id="train",
        extra_info={"madpo_pair_count": 1, "madpo_global_pair_count": 1},
    )
    trainer._route_batch = lambda actual_batch: [
        SimpleNamespace(group_id="actor-a", batch=actual_batch.select_keys(keys[:4])),
        SimpleNamespace(group_id="actor-b", batch=actual_batch.select_keys(keys[4:])),
    ]

    metrics = {}
    trainer._update_actor(batch, metrics)

    assert events[0][1] == keys[:4]
    assert events[1][1] == keys[4:]
    assert all(event[2]["madpo_global_pair_count"] == 1 for event in events)
    assert metrics["actor/actor-a/loss"] == pytest.approx(0.25)
    assert metrics["actor/actor-b/loss"] == pytest.approx(0.25)


def test_fresh_auto_resume_is_allowed_but_existing_checkpoint_fails(tmp_path):
    from trajweave.backends.verl.trainers.joint_preference_sync import TrajWeaveJointPreferenceSyncTrainer

    trainer = object.__new__(TrajWeaveJointPreferenceSyncTrainer)
    trainer.config = SimpleNamespace()
    trainer.config.trainer = SimpleNamespace(
        resume_mode="auto",
        default_local_dir=str(tmp_path),
        get=lambda key, default=None: {"resume_from_path": None}.get(key, default),
    )
    trainer._load_checkpoint()
    assert trainer.global_steps == 0

    (tmp_path / "latest_checkpointed_iteration.txt").write_text("2")
    with pytest.raises(ValueError, match="resume is not implemented"):
        trainer._load_checkpoint()


def test_trainer_snapshots_all_actors_before_sequential_updates():
    tq = pytest.importorskip("transfer_queue")
    from transfer_queue import KVBatchMeta

    from trajweave.backends.verl.trainers.joint_preference_sync import TrajWeaveJointPreferenceSyncTrainer
    from verl.utils.tensordict_utils import list_of_dict_to_tensordict

    fields = _preference_fields(include_padding=False)
    rows = [{field: value[row] for field, value in fields.items()} for row in range(4)]
    keys = [f"madpo-order-{uuid.uuid4().hex}-{row}" for row in range(4)]
    batch = KVBatchMeta(keys=keys, tags=[{"seq_len": 2}] * 4, partition_id="train")
    events: list[str] = []

    class FakeWorkerGroup:
        world_size = 1

        def __init__(self, group_id: str):
            self.group_id = group_id

        def compute_log_prob(self, routed_batch):
            events.append(f"snapshot:{self.group_id}")
            return routed_batch

        def update_actor(self, routed_batch):
            events.append(f"update:{self.group_id}")
            return {"metrics": {}}

    trainer = object.__new__(TrajWeaveJointPreferenceSyncTrainer)
    trainer.multi_actor_trainable_group_ids = ["actor-a", "actor-b"]
    trainer.actor_rollout_wgs = {
        "actor-a": FakeWorkerGroup("actor-a"),
        "actor-b": FakeWorkerGroup("actor-b"),
    }
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            actor=SimpleNamespace(data_loader_seed=0),
            rollout=SimpleNamespace(temperature=1.0),
        )
    )

    tq.init()
    try:
        tq.kv_batch_put(
            keys=keys,
            partition_id="train",
            fields=list_of_dict_to_tensordict(rows),
            tags=batch.tags,
        )
        trainer._compute_old_log_prob(batch, metrics={})
        trainer._compute_advantage(batch, metrics={})
        trainer._update_actor(batch, metrics={})
        assert events == [
            "snapshot:actor-a",
            "snapshot:actor-b",
            "update:actor-a",
            "update:actor-b",
        ]
        assert trainer._madpo_snapshot_deltas["actor-a"].item() == pytest.approx(0.2)
        assert trainer._madpo_snapshot_deltas["actor-b"].item() == pytest.approx(0.6)
    finally:
        tq.kv_clear(keys=keys, partition_id="train")
        tq.close()
