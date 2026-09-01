import json
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from trajweave.backends.verl.async_buffer import PolicyBufferCoordinator
from trajweave.backends.verl.trainers.maporl_multi_actor import (
    TrajWeaveMAPoRLMultiActorSyncTrainer,
    _multi_actor_worker_role,
    _namespace_rollout_replica_class,
)
from trajweave.backends.verl.weight_sync import (
    CheckpointEnginePolicyEndpoint,
    GroupedPolicyWeightTransport,
    MultiActorWeightSyncContract,
    PolicyWeightVersion,
)
from verl.trainer.ppo.utils import Role


def test_checkpoint_engine_policy_endpoint_adapts_success_and_failure():
    calls = []

    class Manager:
        def update_weights(self, *, global_steps):
            calls.append(global_steps)

    endpoint = CheckpointEnginePolicyEndpoint(Manager())
    assert endpoint.load_policy(PolicyWeightVersion(group_id="policy_a", global_step=4)) is True
    assert calls == [4]

    class FailingManager:
        def update_weights(self, *, global_steps):
            raise RuntimeError(f"step {global_steps} failed")

    with pytest.raises(RuntimeError, match="step 4 failed"):
        CheckpointEnginePolicyEndpoint(FailingManager()).load_policy(
            PolicyWeightVersion(group_id="policy_a", global_step=4)
        )


def test_multi_actor_weight_sync_tracks_pending_groups_and_checkpoint(tmp_path):
    contract = MultiActorWeightSyncContract(("policy_a", "policy_b"))
    assert contract.pending_groups == ("policy_a", "policy_b")
    assert contract.synchronized is False

    contract.record_actor_update("policy_a", 1)
    checkpoint = tmp_path / "policy_a.pt"
    checkpoint.write_bytes(b"weights-a")
    version = contract.record_checkpoint("policy_a", checkpoint)

    assert version.global_step == 1
    assert version.checkpoint_sha256
    assert contract.pending_groups == ("policy_a", "policy_b")

    contract.mark_rollout_sync("policy_a", 1)
    assert contract.pending_groups == ("policy_b",)
    assert contract.metric_fields()["trajweave/maporl/weight_sync/rollout_synced"] == 1


def test_multi_actor_weight_sync_requires_exact_version_and_known_groups():
    contract = MultiActorWeightSyncContract(("policy_a", "policy_b"))
    with pytest.raises(KeyError, match="Unknown weight sync group"):
        contract.record_actor_update("unknown", 1)
    contract.record_actor_update("policy_a", 2)
    with pytest.raises(ValueError, match="targets step 1"):
        contract.mark_rollout_sync("policy_a", 1)
    with pytest.raises(ValueError, match="moved backwards"):
        contract.record_actor_update("policy_a", 1)


def test_multi_actor_weight_sync_snapshot_is_json_serializable():
    contract = MultiActorWeightSyncContract(("policy_a",))
    contract.record_actor_update("policy_a", 3)
    snapshot = contract.as_dict()
    assert snapshot["synchronized"] is False
    assert snapshot["pending_groups"] == ["policy_a"]
    assert snapshot["versions"]["policy_a"]["global_step"] == 3

    contract.mark_rollout_sync("policy_a", 3)
    assert contract.as_dict()["synchronized"] is True


def test_multi_actor_weight_sync_transport_advances_only_acknowledged_groups(tmp_path):
    contract = MultiActorWeightSyncContract(("policy_a", "policy_b"))
    for group_id in contract.group_ids:
        contract.record_actor_update(group_id, 1)
        path = tmp_path / f"{group_id}.pt"
        path.write_bytes(group_id.encode())
        contract.record_checkpoint(group_id, path)

    class SelectiveTransport:
        def load_policy(self, version):
            return version.group_id == "policy_a"

    results = contract.sync_pending(SelectiveTransport())

    assert [(item.group_id, item.acknowledged) for item in results] == [
        ("policy_a", True),
        ("policy_b", False),
    ]
    assert contract.pending_groups == ("policy_b",)
    assert contract.as_dict()["versions"]["policy_a"]["rollout_synced_step"] == 1


def test_multi_actor_weight_sync_supports_live_transport_without_checkpoint():
    contract = MultiActorWeightSyncContract(("policy_a",))
    contract.record_actor_update("policy_a", 1)
    loaded = []

    class LiveTransport:
        def load_policy(self, version):
            loaded.append((version.group_id, version.global_step, version.checkpoint_path))
            return True

    results = contract.sync_pending(LiveTransport())

    assert loaded == [("policy_a", 1, None)]
    assert [(result.acknowledged, result.error) for result in results] == [(True, None)]
    assert contract.synchronized is True


def test_actor_update_clears_checkpoint_from_an_older_version(tmp_path):
    contract = MultiActorWeightSyncContract(("policy_a",))
    contract.record_actor_update("policy_a", 1)
    checkpoint = tmp_path / "policy_a.pt"
    checkpoint.write_bytes(b"step-1")
    contract.record_checkpoint("policy_a", checkpoint)

    version = contract.record_actor_update("policy_a", 2)

    assert version.checkpoint_path is None
    assert version.checkpoint_sha256 is None


def test_trainer_persists_acknowledged_weight_sync_state(tmp_path):
    contract = MultiActorWeightSyncContract(("policy_a", "policy_b"))
    for group_id in contract.group_ids:
        contract.record_actor_update(group_id, 1)
        checkpoint = tmp_path / f"{group_id}.pt"
        checkpoint.write_bytes(group_id.encode())
        contract.record_checkpoint(group_id, checkpoint)

    checkpoint_root = tmp_path / "checkpoints"
    (checkpoint_root / "global_step_1").mkdir(parents=True)

    class Transport:
        def load_policy(self, version):
            return True

    trainer = SimpleNamespace(
        global_steps=1,
        config=SimpleNamespace(trainer=SimpleNamespace(default_local_dir=str(checkpoint_root))),
        maporl_weight_sync=contract,
        maporl_weight_transport=Transport(),
    )
    trainer._write_weight_sync_manifest = lambda: (
        TrajWeaveMAPoRLMultiActorSyncTrainer._write_weight_sync_manifest(trainer)
    )

    TrajWeaveMAPoRLMultiActorSyncTrainer._sync_multi_actor_weights(trainer)

    manifest = json.loads(
        (checkpoint_root / "global_step_1" / "multi_actor_weight_sync.json").read_text(encoding="utf-8")
    )
    assert manifest["synchronized"] is True
    assert manifest["pending_groups"] == []


def test_trainer_refreshes_step_metrics_after_endpoint_acknowledgement():
    contract = MultiActorWeightSyncContract(("policy_a", "policy_b"))
    metrics = contract.metric_fields()
    coordinator = PolicyBufferCoordinator(("policy_a", "policy_b"))
    for group_id in contract.group_ids:
        contract.record_actor_update(group_id, 1)
        coordinator.update_actor_step(group_id, 1)

    assert metrics["trajweave/maporl/weight_sync/pending"] == 2

    contract.mark_rollout_sync("policy_a", 1)
    contract.mark_rollout_sync("policy_b", 1)
    coordinator.mark_rollout_sync("policy_a", 1)
    coordinator.mark_rollout_sync("policy_b", 1)
    trainer = SimpleNamespace(
        maporl_weight_sync=contract,
        maporl_buffer_coordinator=coordinator,
        _maporl_step_metrics=metrics,
    )

    TrajWeaveMAPoRLMultiActorSyncTrainer._refresh_weight_sync_metrics(trainer)

    assert metrics["trajweave/maporl/weight_sync/pending"] == 0
    assert metrics["trajweave/maporl/weight_sync/synchronized"] == 1
    assert metrics["trajweave/maporl/buffer/policy_a/rollout_synced_step"] == 1
    assert metrics["trajweave/maporl/buffer/policy_b/policy_lag"] == 0


def test_hf_local_sync_advances_buffer_rollout_version_without_transport():
    coordinator = PolicyBufferCoordinator(("policy_a", "policy_b"))
    coordinator.update_actor_step("policy_a", 3)
    coordinator.update_actor_step("policy_b", 2)

    class CheckpointManager:
        def __init__(self):
            self.steps = []

        def update_weights(self, step):
            self.steps.append(step)

    manager = CheckpointManager()
    trainer = SimpleNamespace(
        global_steps=3,
        maporl_weight_transport=None,
        checkpoint_manager=manager,
        maporl_buffer_coordinator=coordinator,
        maporl_trainable_group_ids=["policy_a", "policy_b"],
    )

    TrajWeaveMAPoRLMultiActorSyncTrainer._sync_multi_actor_weights(trainer)

    assert manager.steps == [3]
    assert coordinator.states["policy_a"].rollout_synced_step == 3
    assert coordinator.states["policy_b"].rollout_synced_step == 2


def test_trainer_initial_sync_does_not_require_checkpoint_files():
    contract = MultiActorWeightSyncContract(("policy_a", "policy_b"))

    class Transport:
        def __init__(self):
            self.loaded = []

        def load_policy(self, version):
            self.loaded.append((version.group_id, version.global_step, version.checkpoint_path))
            return True

    transport = Transport()
    trainer = SimpleNamespace(
        global_steps=0,
        maporl_worker_group_specs={"policy_a": object(), "policy_b": object()},
        maporl_weight_sync=contract,
        maporl_weight_transport=transport,
        _write_weight_sync_manifest=lambda: None,
    )

    TrajWeaveMAPoRLMultiActorSyncTrainer._sync_initial_multi_actor_weights(trainer)

    assert transport.loaded == [
        ("policy_a", 0, None),
        ("policy_b", 0, None),
    ]
    assert contract.synchronized is True


def test_grouped_vllm_replicas_can_share_local_rank_with_distinct_namespaces():
    class Replica:
        def __init__(self, *, replica_rank, name_suffix):
            self.replica_rank = replica_rank
            self.name_suffix = name_suffix

    manager_a = SimpleNamespace(rollout_replica_class=Replica)
    manager_b = SimpleNamespace(rollout_replica_class=Replica)
    _namespace_rollout_replica_class(manager_a, "policy_a")
    _namespace_rollout_replica_class(manager_b, "policy_b")

    replica_a = manager_a.rollout_replica_class(replica_rank=0)
    replica_b = manager_b.rollout_replica_class(replica_rank=0)

    assert replica_a.replica_rank == replica_b.replica_rank == 0
    assert replica_a.name_suffix != replica_b.name_suffix


def test_multi_actor_vllm_workers_include_the_rollout_role():
    enabled = OmegaConf.create({"trajweave": {"multi_actor": {"vllm": {"enabled": True}}}})
    disabled = OmegaConf.create({"trajweave": {"multi_actor": {"vllm": {"enabled": False}}}})

    assert _multi_actor_worker_role(enabled) is Role.ActorRollout
    assert _multi_actor_worker_role(disabled) is Role.Actor


def test_grouped_policy_transport_does_not_fallback_between_endpoints():
    calls = []

    class Endpoint:
        def __init__(self, group_id):
            self.group_id = group_id

        def load_policy(self, version):
            calls.append((self.group_id, version.group_id))
            return True

    transport = GroupedPolicyWeightTransport({"policy_a": Endpoint("server_a"), "policy_b": Endpoint("server_b")})

    assert transport.load_policy(PolicyWeightVersion(group_id="policy_b", global_step=1)) is True
    assert calls == [("server_b", "policy_b")]
    with pytest.raises(KeyError, match="No rollout endpoint"):
        transport.load_policy(PolicyWeightVersion(group_id="unknown", global_step=1))


def test_multi_actor_resume_loads_each_policy_group_and_dataloader(tmp_path):
    checkpoint_root = tmp_path / "global_step_3"
    groups = ("policy_a", "policy_b")
    for group_id in groups:
        actor_path = checkpoint_root / "actors" / group_id
        actor_path.mkdir(parents=True)
        (actor_path / "model_world_size_1_rank_0.pt").write_bytes(group_id.encode())
    torch.save({"epoch": 3}, checkpoint_root / "data.pt")

    class Worker:
        def __init__(self):
            self.loaded = []

        def load_checkpoint(self, **kwargs):
            self.loaded.append(kwargs)

    class Dataloader:
        def __init__(self):
            self.state = None

        def load_state_dict(self, state):
            self.state = state

    trainer = SimpleNamespace(
        config=SimpleNamespace(
            trainer=SimpleNamespace(
                resume_mode="resume_path",
                resume_from_path=str(checkpoint_root),
                del_local_ckpt_after_load=False,
            )
        ),
        actor_rollout_wgs={group_id: Worker() for group_id in groups},
        maporl_trainable_group_ids=list(groups),
        maporl_weight_sync=MultiActorWeightSyncContract(groups),
        train_dataloader=Dataloader(),
    )

    TrajWeaveMAPoRLMultiActorSyncTrainer._load_checkpoint(trainer)

    assert trainer.global_steps == 3
    assert all(worker.loaded for worker in trainer.actor_rollout_wgs.values())
    assert trainer.train_dataloader.state == {"epoch": 3}
    assert trainer.maporl_weight_sync.pending_groups == groups
