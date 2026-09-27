from __future__ import annotations

import json

import pytest
from omegaconf import OmegaConf

from trajweave.backends.verl.multi_actor import reconcile_multi_actor_global_assets
from trajweave.backends.verl.multi_actor.critic_config import normalize_critic_route_specs


@pytest.mark.parametrize("use_dict_config", [False, True])
def test_reconcile_multi_actor_global_assets_supports_dict_and_dict_config(use_dict_config):
    raw = {
        "trajweave": {"multi_actor": {"enabled": True}},
        "agent": {
            "worker_groups": [
                {
                    "id": "frozen",
                    "model_path": "/models/frozen",
                    "tokenizer_path": "/tokenizers/frozen",
                    "trainable": False,
                },
                {
                    "id": "actor-a",
                    "model_path": "/models/actor-a",
                    "tokenizer_path": "/tokenizers/shared",
                    "trainable": True,
                },
            ]
        },
        "actor_rollout_ref": {"model": {"path": "/models/stale", "tokenizer_path": "/tokenizers/stale"}},
    }
    config = OmegaConf.create(raw) if use_dict_config else raw

    assets = reconcile_multi_actor_global_assets(config)

    assert assets == {
        "worker_group": "actor-a",
        "model_path": "/models/actor-a",
        "tokenizer_path": "/tokenizers/shared",
    }
    resolved = OmegaConf.to_container(config, resolve=True) if use_dict_config else config
    assert resolved["actor_rollout_ref"]["model"] == {
        "path": "/models/actor-a",
        "tokenizer_path": "/tokenizers/shared",
    }


@pytest.mark.parametrize("use_dict_config", [False, True])
def test_reconcile_multi_actor_global_assets_disabled_is_noop(use_dict_config):
    raw = {
        "trajweave": {"multi_actor": {"enabled": False}},
        "actor_rollout_ref": {"model": {"path": "/models/original"}},
    }
    config = OmegaConf.create(raw) if use_dict_config else raw

    assert reconcile_multi_actor_global_assets(config) is None
    resolved = OmegaConf.to_container(config, resolve=True) if use_dict_config else config
    assert resolved == raw


@pytest.mark.parametrize("use_dict_config", [False, True])
def test_reconcile_multi_actor_global_assets_aligns_validation_critic(use_dict_config):
    raw = {
        "trajweave": {
            "multi_actor": {"enabled": True},
            "comlrl": {
                "actor_critic": {
                    "topology": "centralized",
                    "critic_type": "v",
                    "critic_routes": [
                        {
                            "critic_group": "team-critic",
                            "actor_groups": ["actor-a", "actor-b"],
                            "model_path": "/models/team-critic",
                            "tokenizer_path": "/tokenizers/critic",
                            "gpus": 1,
                        }
                    ],
                }
            },
        },
        "agent": {
            "worker_groups": [
                {
                    "id": group,
                    "model_path": f"/models/{group}",
                    "tokenizer_path": "/tokenizers/shared",
                    "trainable": True,
                }
                for group in ("actor-a", "actor-b")
            ],
        },
        "actor_rollout_ref": {"model": {"path": "/stale-actor"}},
        "critic": {"model": {"path": "/stale-critic"}},
    }
    config = OmegaConf.create(raw) if use_dict_config else raw

    reconcile_multi_actor_global_assets(config)

    resolved = OmegaConf.to_container(config, resolve=True) if use_dict_config else config
    assert resolved["critic"]["model"] == {
        "path": "/models/team-critic",
        "tokenizer_path": "/tokenizers/critic",
    }
    assert resolved["actor_rollout_ref"]["rollout"]["calculate_log_probs"] is False


class _CheckpointRecorder:
    def __init__(self, group_id: str, calls: list[tuple[str, str, int, int | None]]) -> None:
        self.group_id = group_id
        self.calls = calls

    def save_checkpoint(self, path, remote_path, global_step, *, max_ckpt_to_keep=None):
        assert remote_path is None
        self.calls.append((self.group_id, path, global_step, max_ckpt_to_keep))


class _StatefulDataloader:
    @staticmethod
    def state_dict():
        return {"cursor": 3}


def _checkpoint_trainer(tmp_path, routes):
    from trajweave.backends.verl.trainers.multi_actor_critic_sync import (
        TrajWeaveMultiActorCriticSyncTrainer,
    )

    actor_calls = []
    critic_calls = []
    trainer = object.__new__(TrajWeaveMultiActorCriticSyncTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "default_local_dir": str(tmp_path),
                "max_actor_ckpt_to_keep": 2,
                "max_critic_ckpt_to_keep": 4,
            }
        }
    )
    trainer.global_steps = 7
    trainer.multi_actor_trainable_group_ids = ["actor-a", "actor-b"]
    trainer.actor_rollout_wgs = {
        group_id: _CheckpointRecorder(group_id, actor_calls) for group_id in trainer.multi_actor_trainable_group_ids
    }
    trainer.critic_route_specs = {route.critic_group: route for route in routes}
    trainer.critic_wgs = {
        group_id: _CheckpointRecorder(group_id, critic_calls) for group_id in trainer.critic_route_specs
    }
    trainer.critic_topology = routes[0].topology
    trainer.critic_type = routes[0].critic_type
    trainer.train_dataloader = _StatefulDataloader()
    return trainer, actor_calls, critic_calls


def _read_actor_critic_manifest(tmp_path):
    from trajweave.backends.verl.trainers.multi_actor_critic_sync import ACTOR_CRITIC_CHECKPOINT_MANIFEST

    path = tmp_path / "global_step_7" / ACTOR_CRITIC_CHECKPOINT_MANIFEST
    return json.loads(path.read_text(encoding="utf-8"))


def test_iac_checkpoint_saves_every_critic_group_and_auditable_manifest(tmp_path):
    routes = normalize_critic_route_specs(
        [
            {
                "id": "critic-a",
                "actor_groups": ["actor-a"],
                "topology": "independent",
                "critic_type": "v",
                "model_path": "/models/critic-a",
                "tokenizer_path": "/tokenizer",
                "gpus": 1,
            },
            {
                "id": "critic-b",
                "actor_groups": ["actor-b"],
                "topology": "independent",
                "critic_type": "v",
                "model_path": "/models/critic-b",
                "tokenizer_path": "/tokenizer",
                "gpus": 1,
            },
        ],
        trainable_actor_groups=["actor-a", "actor-b"],
    )
    trainer, actor_calls, critic_calls = _checkpoint_trainer(tmp_path, routes)

    trainer._save_checkpoint()

    assert [call[0] for call in actor_calls] == ["actor-a", "actor-b"]
    assert [call[0] for call in critic_calls] == ["critic-a", "critic-b"]
    assert [call[1] for call in critic_calls] == [
        str(tmp_path / "global_step_7" / "critics" / "critic_a"),
        str(tmp_path / "global_step_7" / "critics" / "critic_b"),
    ]
    assert all(call[2:] == (7, 4) for call in critic_calls)
    manifest = _read_actor_critic_manifest(tmp_path)
    assert manifest["resume_supported"] is False
    assert manifest["critic_topology"] == "independent"
    assert manifest["critic_type"] == "v"
    assert manifest["actors"] == [
        {"group_id": "actor-a", "path": "actors/actor_a"},
        {"group_id": "actor-b", "path": "actors/actor_b"},
    ]
    assert manifest["critics"] == [
        {
            "group_id": "critic-a",
            "path": "critics/critic_a",
            "actor_groups": ["actor-a"],
            "topology": "independent",
            "critic_type": "v",
        },
        {
            "group_id": "critic-b",
            "path": "critics/critic_b",
            "actor_groups": ["actor-b"],
            "topology": "independent",
            "critic_type": "v",
        },
    ]
    assert (tmp_path / "global_step_7" / "data.pt").is_file()
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "7"


def test_maac_checkpoint_saves_single_centralized_critic_and_manifest(tmp_path):
    routes = normalize_critic_route_specs(
        [
            {
                "id": "team/critic",
                "actor_groups": ["actor-a", "actor-b"],
                "topology": "centralized",
                "critic_type": "q",
                "model_path": "/models/team-critic",
                "tokenizer_path": "/tokenizer",
                "gpus": 1,
            }
        ],
        trainable_actor_groups=["actor-a", "actor-b"],
    )
    trainer, _actor_calls, critic_calls = _checkpoint_trainer(tmp_path, routes)

    trainer._save_checkpoint()

    assert critic_calls == [
        (
            "team/critic",
            str(tmp_path / "global_step_7" / "critics" / "team_critic"),
            7,
            4,
        )
    ]
    manifest = _read_actor_critic_manifest(tmp_path)
    assert manifest["critic_topology"] == "centralized"
    assert manifest["critic_type"] == "q"
    assert manifest["critics"] == [
        {
            "group_id": "team/critic",
            "path": "critics/team_critic",
            "actor_groups": ["actor-a", "actor-b"],
            "topology": "centralized",
            "critic_type": "q",
        }
    ]
