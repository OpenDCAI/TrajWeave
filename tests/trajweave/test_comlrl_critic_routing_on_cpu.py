from __future__ import annotations

import os

import pytest

from trajweave.backends.verl.multi_actor.critic_config import (
    normalize_critic_route_specs,
    resolve_actor_critic_settings,
)
from trajweave.backends.verl.routing import safe_critic_role_key


def _separate(group: str, actor: str, *, topology: str = "independent") -> dict:
    return {
        "id": group,
        "actor_groups": [actor],
        "topology": topology,
        "critic_type": "v",
        "model_path": f"/models/{group}",
        "tokenizer_path": "/tokenizer",
        "gpus": 1,
    }


def test_iac_requires_exactly_one_separate_critic_per_trainable_actor():
    routes = normalize_critic_route_specs(
        [_separate("critic-a", "actor-a"), _separate("critic-b", "actor-b")],
        trainable_actor_groups=["actor-a", "actor-b"],
    )

    assert [route.actor_groups for route in routes] == [("actor-a",), ("actor-b",)]
    assert all(route.topology == "independent" for route in routes)

    with pytest.raises(ValueError, match="exactly one critic"):
        normalize_critic_route_specs(
            [_separate("critic-a", "actor-a")],
            trainable_actor_groups=["actor-a", "actor-b"],
        )


def test_maac_requires_one_centralized_critic_covering_all_actors():
    route = {
        "id": "team-critic",
        "actor_groups": ["actor-b", "actor-a"],
        "topology": "centralized",
        "critic_type": "q",
        "model_path": "/models/critic",
        "tokenizer_path": "/tokenizer",
        "gpus": 2,
    }
    [normalized] = normalize_critic_route_specs([route], trainable_actor_groups=["actor-a", "actor-b"])
    assert normalized.critic_type == "q"
    assert normalized.gpus == 2

    with pytest.raises(ValueError, match="exactly one critic group"):
        normalize_critic_route_specs(
            [route, {**route, "id": "other"}],
            trainable_actor_groups=["actor-a", "actor-b"],
        )


def test_actor_critic_settings_support_only_two_explicit_runtime_paths():
    direct = {"trajweave": {"actor_critic": {"critic_routes": []}}}
    nested = {"trajweave": {"comlrl": {"actor_critic": {"critic_routes": []}}}}
    assert resolve_actor_critic_settings(direct) == {"critic_routes": []}
    assert resolve_actor_critic_settings(nested) == {"critic_routes": []}
    with pytest.raises(ValueError, match="algorithm.actor_critic"):
        resolve_actor_critic_settings({"algorithm": {"actor_critic": {}}}, required=True)
    with pytest.raises(ValueError, match="multi_actor.critic_routes"):
        resolve_actor_critic_settings({"trajweave": {"multi_actor": {"critic_routes": []}}}, required=True)
    with pytest.raises(ValueError, match="exactly one path"):
        resolve_actor_critic_settings(
            {
                "trajweave": {
                    "actor_critic": {},
                    "comlrl": {"actor_critic": {}},
                }
            }
        )


def test_critic_route_validates_model_tokenizer_gpus_length_and_shared_head_capability():
    base = _separate("critic-a", "actor-a")
    for field in ("model_path", "tokenizer_path"):
        with pytest.raises(ValueError, match=field):
            normalize_critic_route_specs([{**base, field: None}], trainable_actor_groups=["actor-a"])
    with pytest.raises(ValueError, match="positive integer"):
        normalize_critic_route_specs([{**base, "gpus": 0}], trainable_actor_groups=["actor-a"])
    with pytest.raises(ValueError, match="max_length.*positive integer"):
        normalize_critic_route_specs([{**base, "max_length": 0}], trainable_actor_groups=["actor-a"])
    [route] = normalize_critic_route_specs([base], trainable_actor_groups=["actor-a"])
    assert route.max_length == 2048
    [short_route] = normalize_critic_route_specs([base], trainable_actor_groups=["actor-a"], max_length=17)
    assert short_route.max_length == 17
    with pytest.raises(ValueError, match="real actor value-head worker"):
        normalize_critic_route_specs(
            [{**base, "shared_with_actor": True}],
            trainable_actor_groups=["actor-a"],
        )


def test_trainer_registers_through_unified_entrypoint_and_checkpoint_paths_are_isolated():
    pytest.importorskip("transfer_queue")
    from trajweave.backends.verl.trainers import register_trajweave_trainers
    from verl.trainer.ppo.v1.trainer_base import get_trainer_cls

    register_trajweave_trainers()
    registered = get_trainer_cls("trajweave_multi_actor_critic_sync")
    assert registered.__name__ == "TrajWeaveMultiActorCriticSyncTrainer"

    from trajweave.backends.verl.trainers.multi_actor_critic_sync import checkpoint_directory_map

    paths = checkpoint_directory_map(
        "/tmp/checkpoints/global_step_7",
        actor_group_ids=["actor/a", "actor-b"],
        critic_group_ids=["critic/a", "team-critic"],
    )
    assert paths["actors"]["actor/a"].endswith(os.path.join("actors", "actor_a"))
    assert paths["critics"]["critic/a"].endswith(os.path.join("critics", "critic_a"))
    assert set(paths["actors"].values()).isdisjoint(paths["critics"].values())
    assert safe_critic_role_key("Qwen/critic.1") == "trajweave_critic_qwen_critic_1"

    with pytest.raises(ValueError, match="collide"):
        checkpoint_directory_map(
            "/tmp/checkpoints",
            actor_group_ids=["actor/a", "actor-a"],
            critic_group_ids=[],
        )


def test_critic_runtime_rejects_multi_gpu_routes_until_padding_is_supported():
    from trajweave.backends.verl.trainers.multi_actor_critic_sync import (
        _validate_single_gpu_critic_routes,
    )

    routes = normalize_critic_route_specs(
        [{**_separate("critic-a", "actor-a"), "gpus": 2}],
        trainable_actor_groups=["actor-a"],
    )
    with pytest.raises(ValueError, match="requires gpus=1"):
        _validate_single_gpu_critic_routes(routes)


def test_auto_resume_starts_fresh_only_without_marker_or_explicit_path(tmp_path):
    from omegaconf import OmegaConf

    from trajweave.backends.verl.trainers.multi_actor_critic_sync import (
        TrajWeaveMultiActorCriticSyncTrainer,
    )

    trainer = object.__new__(TrajWeaveMultiActorCriticSyncTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "resume_mode": "auto",
                "resume_from_path": None,
                "default_local_dir": str(tmp_path),
            }
        }
    )
    trainer._load_checkpoint()
    assert trainer.global_steps == 0

    marker = tmp_path / "latest_checkpointed_iteration.txt"
    marker.write_text("3")
    with pytest.raises(ValueError, match="resume is not implemented"):
        trainer._load_checkpoint()
    marker.unlink()
    trainer.config.trainer.resume_from_path = str(tmp_path / "global_step_3")
    with pytest.raises(ValueError, match="resume is not implemented"):
        trainer._load_checkpoint()


def test_valid_existing_critic_token_fields_are_reused_without_retokenizing():
    from trajweave.backends.verl.trainers.multi_actor_critic_sync import (
        _validated_or_tokenized_critic_fields,
    )

    class MustNotRunTokenizer:
        def __call__(self, *_args, **_kwargs):
            raise AssertionError("valid supplied critic tokens must be reused")

    [route] = normalize_critic_route_specs([_separate("critic-a", "actor-a")], trainable_actor_groups=["actor-a"])
    fields = _validated_or_tokenized_critic_fields(
        {
            "critic_input_ids": [7, 8],
            "critic_attention_mask": [1, 1],
            "critic_position_ids": [0, 1],
        },
        text="ignored",
        tokenizer=MustNotRunTokenizer(),
        route=route,
        batch_key="existing",
    )
    assert fields["critic_input_ids"] == [7, 8]


def test_critic_tokenizer_truncates_to_route_limit_and_rejects_real_model_limit():
    from trajweave.backends.verl.trainers.multi_actor_critic_sync import (
        _validated_or_tokenized_critic_fields,
    )

    calls = []

    class FakeTokenizer:
        model_max_length = 8

        def __call__(self, text, **kwargs):
            calls.append((text, kwargs))
            return {"input_ids": list(range(20))[: kwargs["max_length"]]}

    [route] = normalize_critic_route_specs(
        [{**_separate("critic-a", "actor-a"), "max_length": 4}],
        trainable_actor_groups=["actor-a"],
    )
    fields = _validated_or_tokenized_critic_fields(
        {
            "critic_input_ids": [],
            "critic_attention_mask": [],
            "critic_position_ids": [],
        },
        text="long critic text",
        tokenizer=FakeTokenizer(),
        route=route,
        batch_key="long",
    )
    assert fields["critic_input_ids"] == [0, 1, 2, 3]
    assert calls == [
        (
            "long critic text",
            {"truncation": True, "max_length": 4, "add_special_tokens": False},
        )
    ]

    [too_long] = normalize_critic_route_specs(
        [{**_separate("critic-a", "actor-a"), "max_length": 9}],
        trainable_actor_groups=["actor-a"],
    )
    with pytest.raises(ValueError, match="exceeds.*model_max_length=8"):
        _validated_or_tokenized_critic_fields(
            {
                "critic_input_ids": [],
                "critic_attention_mask": [],
                "critic_position_ids": [],
            },
            text="text",
            tokenizer=FakeTokenizer(),
            route=too_long,
            batch_key="too-long",
        )

    sentinel = FakeTokenizer()
    sentinel.model_max_length = 10**30
    _validated_or_tokenized_critic_fields(
        {
            "critic_input_ids": [],
            "critic_attention_mask": [],
            "critic_position_ids": [],
        },
        text="text",
        tokenizer=sentinel,
        route=too_long,
        batch_key="sentinel",
    )


def test_ensure_iac_critic_fields_uses_each_route_tokenizer_with_real_tq(monkeypatch):
    tq = pytest.importorskip("transfer_queue")
    import torch
    from transfer_queue import KVBatchMeta

    from trajweave.backends.verl.trainers import multi_actor_critic_sync as trainer_module
    from trajweave.backends.verl.trainers.multi_actor_critic_sync import (
        TrajWeaveMultiActorCriticSyncTrainer,
    )
    from verl.utils.tensordict_utils import list_of_dict_to_tensordict

    loaded_paths = []

    class FakeTokenizer:
        model_max_length = 4096

        def __init__(self, path):
            self.path = path

        def __call__(self, text, *, truncation, max_length, add_special_tokens):
            assert truncation is True
            assert max_length == 2048
            assert add_special_tokens is False
            return {"input_ids": [len(self.path), len(text)][:max_length]}

    def fake_loader(path):
        loaded_paths.append(path)
        return FakeTokenizer(path)

    monkeypatch.setattr(trainer_module, "_load_critic_tokenizer", fake_loader)
    routes = normalize_critic_route_specs(
        [
            {**_separate("critic-a", "actor-a"), "tokenizer_path": "/critic-tokenizer-a"},
            {**_separate("critic-b", "actor-b"), "tokenizer_path": "/critic-tokenizer-b"},
        ],
        trainable_actor_groups=["actor-a", "actor-b"],
    )
    trainer = object.__new__(TrajWeaveMultiActorCriticSyncTrainer)
    trainer.critic_route_specs = {route.critic_group: route for route in routes}
    trainer.actor_to_critic_group = {route.actor_groups[0]: route.critic_group for route in routes}
    trainer.critic_topology = "independent"
    trainer.critic_type = "v"
    trainer.multi_actor_trainable_group_ids = ["actor-a", "actor-b"]

    rows = []
    for index, group in enumerate(("actor-a", "actor-b")):
        rows.append(
            {
                "worker_group": group,
                "agent_id": f"agent-{index}",
                "traj_uid": "trajectory",
                "turn_id": 0,
                "prompt_text": f"prompt-{index}",
                "response_text": f"response-{index}",
                "joint_action_ids": [f"action-{index}"],
                "joint_transition_ids": [f"transition-{index}"],
                "critic_group": f"__missing__:{index}",
                "critic_type": "__missing__",
                "critic_input_ids": [],
                "critic_attention_mask": [],
                "critic_position_ids": [],
                "critic_loss_mask": 0.0,
                "response_mask": torch.ones(1, dtype=torch.long),
            }
        )
    keys = ["iac-real-0", "iac-real-1"]
    tq.init()
    try:
        tq.kv_batch_put(
            keys=keys,
            partition_id="train",
            fields=list_of_dict_to_tensordict(rows),
            tags=[{"seq_len": 1}, {"seq_len": 1}],
        )
        batch = KVBatchMeta(keys=keys, tags=[{}, {}], partition_id="train")
        trainer._ensure_critic_fields(batch)
        output = tq.kv_batch_get(
            keys=keys,
            partition_id="train",
            select_fields=[
                "critic_group",
                "critic_type",
                "critic_input_ids",
                "critic_attention_mask",
                "critic_position_ids",
                "critic_loss_mask",
                "agent_id",
                "traj_uid",
            ],
        )
        from trajweave.backends.verl.schema import to_python

        assert to_python(output["critic_group"]) == ["critic-a", "critic-b"]
        assert to_python(output["critic_type"]) == ["v", "v"]
        input_ids = [to_python(output["critic_input_ids"][row]) for row in range(2)]
        attention_mask = [to_python(output["critic_attention_mask"][row]) for row in range(2)]
        position_ids = [to_python(output["critic_position_ids"][row]) for row in range(2)]
        assert input_ids == [
            [len("/critic-tokenizer-a"), len("prompt-0")],
            [len("/critic-tokenizer-b"), len("prompt-1")],
        ]
        assert attention_mask == [[1, 1], [1, 1]]
        assert position_ids == [[0, 1], [0, 1]]
        assert [float(to_python(output["critic_loss_mask"][row])) for row in range(2)] == [1.0, 1.0]
        assert trainer_module._unpack_advantage_tq_field(output["agent_id"]) == ["agent-0", "agent-1"]
        assert trainer_module._unpack_advantage_tq_field(output["traj_uid"]) == [
            "trajectory",
            "trajectory",
        ]
        assert loaded_paths == ["/critic-tokenizer-a", "/critic-tokenizer-b"]
    finally:
        tq.kv_clear(keys=keys, partition_id="train")
        tq.close()
