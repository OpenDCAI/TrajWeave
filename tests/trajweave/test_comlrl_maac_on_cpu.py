from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import pytest
import torch

from trajweave.backends.verl.extensions.comlrl.actor_critic import prepare_actor_critic_batch
from trajweave.backends.verl.workflow_runtime import _trajectory_to_outputs
from trajweave.core.joint_trajectory import JointAction, JointTransition
from trajweave.core.specs import AgentSpec, PolicyGroupSpec, TeamSpec
from trajweave.core.trajectory import AgentTurn, MultiAgentTrajectory
from trajweave.credit.comlrl.actor_critic import (
    build_maac_q_critic_text,
    build_maac_v_critic_text,
    deduplicate_maac_critic_indices,
    deduplicate_maac_critic_samples,
)


def _team() -> TeamSpec:
    return TeamSpec(
        name="maac-test",
        agents=(
            AgentSpec(name="beta", role="solver", policy_group="actor-b"),
            AgentSpec(name="alpha", role="solver", policy_group="actor-a"),
        ),
        policy_groups=(
            PolicyGroupSpec(name="actor-a"),
            PolicyGroupSpec(name="actor-b"),
        ),
        orchestra="test",
        reward="test",
        credit="test",
    )


def test_maac_v_and_q_text_follow_team_order_and_q_appends_joint_actions():
    team = _team()
    prompts = {"alpha": "prompt-a", "beta": "prompt-b"}
    actions = {"alpha": "action-a", "beta": "action-b"}

    v_text = build_maac_v_critic_text(team, prompts)
    q_text = build_maac_q_critic_text(team, prompts, actions)

    assert "[Agent 0] prompt-b" in v_text
    assert "[Agent 1] prompt-a" in v_text
    assert v_text.index("prompt-b") < v_text.index("prompt-a")
    assert "Joint Action" not in v_text
    assert q_text.startswith(v_text)
    assert q_text.index("action-b") < q_text.index("action-a")

    with pytest.raises(ValueError, match="exactly match"):
        build_maac_v_critic_text(team, {"alpha": "missing beta"})


def test_maac_workflow_bridge_preserves_per_actor_prompt_and_response_text():
    team = _team()
    prompts = {"beta": "prompt-b", "alpha": "prompt-a"}
    actions = {"beta": "action-b", "alpha": "action-a"}
    turns = []
    for index, agent in enumerate(team.agents):
        turns.append(
            AgentTurn(
                episode_id="episode-maac",
                task_id="task-maac",
                turn_id=index,
                agent_name=agent.name,
                role=agent.role,
                policy_group=agent.policy_group,
                observation="observation",
                prompt=prompts[agent.name],
                action_text=actions[agent.name],
                action_token_ids=[index + 1],
                reward=1.0,
                done=True,
                completion_id=f"completion-{agent.name}",
                tree_node_id="node-joint",
                joint_action_ids=["action-joint"],
                joint_transition_ids=["transition-joint"],
            )
        )
    trajectory = MultiAgentTrajectory(
        episode_id="episode-maac",
        task_id="task-maac",
        rollout_group="group-maac",
        team_name=team.name,
        global_reward=1.0,
        success=True,
        turns=turns,
    )
    worker = SimpleNamespace(
        config={
            "trajweave": {
                "actor_critic": {
                    "topology": "centralized",
                    "critic_type": "q",
                    "critic_routes": [
                        {
                            "id": "team-critic",
                            "actor_groups": ["actor-a", "actor-b"],
                            "model_path": "/models/team-critic",
                            "tokenizer_path": "/tokenizers/team-critic",
                            "gpus": 1,
                        }
                    ],
                }
            }
        },
        _encode_prompt_text=lambda text: [ord(text[0])],
        _local_policy_version=lambda: 0,
    )

    outputs = _trajectory_to_outputs(worker, trajectory=trajectory, team=team)

    assert len(outputs) == 2
    assert [output.extra_fields["prompt_text"] for output in outputs] == ["prompt-b", "prompt-a"]
    assert [output.extra_fields["response_text"] for output in outputs] == ["action-b", "action-a"]
    assert all(output.extra_fields["critic_group"].startswith("__missing__") for output in outputs)
    assert all(output.extra_fields["critic_input_ids"] == [] for output in outputs)
    assert all(output.extra_fields["critic_loss_mask"] == 0.0 for output in outputs)
    assert all(output.extra_fields["joint_transition_ids"] == ["transition-joint"] for output in outputs)
    assert [output.extra_fields["turn_id"] for output in outputs] == [0, 0]

    prepared = prepare_actor_critic_batch(
        {
            "response_mask": torch.ones(2, 1, dtype=torch.long),
            "old_values": torch.tensor([0.5, 0.5]),
            "joint_action_ids": [output.extra_fields["joint_action_ids"] for output in outputs],
            "joint_transition_ids": [output.extra_fields["joint_transition_ids"] for output in outputs],
            "joint_reward": [output.extra_fields["joint_reward"] for output in outputs],
            "joint_done": [output.extra_fields["joint_done"] for output in outputs],
            "joint_truncated": [output.extra_fields["joint_truncated"] for output in outputs],
            "agent_id": [output.extra_fields["agent_id"] for output in outputs],
            "worker_group": [output.extra_fields["worker_group"] for output in outputs],
            "turn_id": [output.extra_fields["turn_id"] for output in outputs],
            "traj_uid": [output.extra_fields["traj_uid"] for output in outputs],
            "uid": ["group-maac", "group-maac"],
            "critic_group": ["team-critic", "team-critic"],
            "critic_type": ["q", "q"],
            "critic_input_ids": torch.ones(2, 1, dtype=torch.long),
            "critic_attention_mask": torch.ones(2, 1, dtype=torch.long),
            "critic_position_ids": torch.zeros(2, 1, dtype=torch.long),
            "critic_loss_mask": [1.0, 1.0],
        },
        topology="centralized",
        gamma=0.8,
        normalize_by_population_std=False,
    )
    assert prepared.critic_row_indices == (0,)


def test_maac_source_turn_survives_workflow_agent_loop_tq_and_critic_preparation(monkeypatch):
    tq = pytest.importorskip("transfer_queue")
    from tensordict import TensorDict
    from transfer_queue import KVBatchMeta

    from trajweave.backends.verl.agent_loop import TrajWeaveSyntheticAgentLoopWorkerTQ
    from trajweave.backends.verl.extensions.comlrl.actor_critic import ACTOR_CRITIC_TQ_FIELDS
    from trajweave.backends.verl.multi_actor.critic_config import normalize_critic_route_specs
    from trajweave.backends.verl.trainers import multi_actor_critic_sync as trainer_module
    from trajweave.backends.verl.trainers.multi_actor_critic_sync import (
        TrajWeaveMultiActorCriticSyncTrainer,
    )

    team = _team()
    turns = [
        AgentTurn(
            episode_id="episode-source-turn",
            task_id="task",
            turn_id=7 + index,
            agent_name=agent.name,
            role=agent.role,
            policy_group=agent.policy_group,
            observation="obs",
            prompt=f"prompt-{agent.policy_group}",
            action_text=f"action-{agent.policy_group}",
            action_token_ids=[index + 3],
            reward=1.0,
            done=True,
            completion_id=f"completion-{agent.name}",
            tree_node_id="node-source",
            joint_action_ids=["joint-action-source"],
            joint_transition_ids=["joint-transition-source"],
        )
        for index, agent in enumerate(team.agents)
    ]
    trajectory = MultiAgentTrajectory(
        episode_id="episode-source-turn",
        task_id="task",
        rollout_group="group",
        team_name=team.name,
        turns=turns,
        global_reward=1.0,
        success=True,
        joint_actions=[
            JointAction(
                joint_action_id="joint-action-source",
                tree_node_id="node-source",
                completion_ids={"beta": "completion-beta", "alpha": "completion-alpha"},
                candidate_indices={"beta": 0, "alpha": 0},
                joint_transition_id="joint-transition-source",
                shared_reward=1.0,
                done=True,
            )
        ],
        joint_transitions=[
            JointTransition(
                joint_transition_id="joint-transition-source",
                joint_action_id="joint-action-source",
                source_tree_node_id="node-source",
                source_turn=3,
                done=True,
            )
        ],
    )
    actor_critic = {
        "topology": "centralized",
        "critic_type": "v",
        "critic_routes": [
            {
                "id": "team-critic",
                "actor_groups": ["actor-a", "actor-b"],
                "model_path": "/models/team",
                "tokenizer_path": "/tokenizers/team",
                "gpus": 1,
                "max_length": 32,
            }
        ],
    }

    actor_class = TrajWeaveSyntheticAgentLoopWorkerTQ.__ray_actor_class__

    class ActorTokenizer:
        pad_token_id = 0
        eos_token_id = 1

    class FakeWorker:
        _put_outputs = actor_class._put_outputs

        def __init__(self):
            self.config = {"trajweave": {"actor_critic": actor_critic, "turn_padding_multiple": 1}}
            self.rollout_config = SimpleNamespace(prompt_length=8, response_length=4, temperature=1.0)
            self.tokenizer = ActorTokenizer()

        @staticmethod
        def _encode_prompt_text(text):
            return [ord(text[0])]

        @staticmethod
        def _local_policy_version():
            return 0

        @staticmethod
        def _compute_multi_modal_inputs(output, input_ids):
            return None

        @staticmethod
        def _compute_position_ids(input_ids, attention_mask, multi_modal_inputs):
            return torch.arange(input_ids.shape[-1], dtype=torch.long).unsqueeze(0)

        @staticmethod
        def _attach_worker_group_stats(rows):
            return None

        @staticmethod
        def _write_online_turns(runtime, rows):
            return None

    worker = FakeWorker()
    outputs = _trajectory_to_outputs(worker, trajectory=trajectory, team=team)
    assert [output.extra_fields["turn_id"] for output in outputs] == [3, 3]

    class CriticTokenizer:
        model_max_length = 64

        def __call__(self, text, **kwargs):
            assert kwargs == {"truncation": True, "max_length": 32, "add_special_tokens": False}
            return {"input_ids": [5, len(text)]}

    monkeypatch.setattr(trainer_module, "_load_critic_tokenizer", lambda _path: CriticTokenizer())
    [route] = normalize_critic_route_specs(
        actor_critic["critic_routes"],
        trainable_actor_groups=["actor-a", "actor-b"],
        topology="centralized",
        critic_type="v",
    )
    trainer = object.__new__(TrajWeaveMultiActorCriticSyncTrainer)
    trainer.critic_route_specs = {route.critic_group: route}
    trainer.actor_to_critic_group = {group: route.critic_group for group in route.actor_groups}
    trainer.critic_topology = "centralized"
    trainer.critic_type = "v"
    trainer.multi_actor_trainable_group_ids = ["actor-a", "actor-b"]

    uid = f"source-turn-{uuid.uuid4().hex}"
    keys = [f"{uid}_0_0", f"{uid}_0_1"]
    tq.init()
    try:
        asyncio.run(
            worker._put_outputs(
                outputs,
                validate=False,
                uid=uid,
                session_id=0,
                global_steps=0,
            )
        )
        turn_data = tq.kv_batch_get(keys=keys, partition_id="train", select_fields=["turn_id"])
        from trajweave.backends.verl.schema import to_python

        assert to_python(turn_data["turn_id"]) == [3, 3]
        batch = KVBatchMeta(keys=keys, tags=[{}, {}], partition_id="train")
        trainer._ensure_critic_fields(batch)
        tq.kv_batch_put(
            keys=keys,
            partition_id="train",
            fields=TensorDict({"old_values": torch.tensor([0.5, 0.5])}, batch_size=2),
        )
        selected = (*ACTOR_CRITIC_TQ_FIELDS, "response_mask", "old_values")
        data = tq.kv_batch_get(keys=keys, partition_id="train", select_fields=selected)
        mapping = {key: trainer_module._unpack_advantage_tq_field(data[key]) for key in selected}
        prepared = prepare_actor_critic_batch(
            mapping,
            topology="centralized",
            gamma=0.9,
            normalize_by_population_std=False,
        )
        assert prepared.critic_row_indices == (0,)
        assert prepared.scalar_targets.tolist() == [1.0, 1.0]
    finally:
        owned = [key for key in tq.kv_list(partition_id="train").get("train", {}) if key.startswith(uid)]
        if owned:
            tq.kv_clear(keys=owned, partition_id="train")
        tq.close()


def test_ensure_maac_critic_fields_aggregates_actor_groups_in_stable_order_with_real_tq(monkeypatch):
    tq = pytest.importorskip("transfer_queue")
    from transfer_queue import KVBatchMeta

    from trajweave.backends.verl.multi_actor.critic_config import normalize_critic_route_specs
    from trajweave.backends.verl.trainers import multi_actor_critic_sync as trainer_module
    from trajweave.backends.verl.trainers.multi_actor_critic_sync import (
        TrajWeaveMultiActorCriticSyncTrainer,
    )
    from verl.utils.tensordict_utils import list_of_dict_to_tensordict

    encoded_texts = []

    class FakeTokenizer:
        model_max_length = 4096

        def __call__(self, text, *, truncation, max_length, add_special_tokens):
            assert truncation is True
            assert max_length == 2048
            assert add_special_tokens is False
            encoded_texts.append(text)
            return {"input_ids": [11, len(text)][:max_length]}

    monkeypatch.setattr(trainer_module, "_load_critic_tokenizer", lambda _path: FakeTokenizer())
    [route] = normalize_critic_route_specs(
        [
            {
                "id": "team-critic",
                "actor_groups": ["actor-a", "actor-b"],
                "topology": "centralized",
                "critic_type": "q",
                "model_path": "/models/team",
                "tokenizer_path": "/tokenizers/team",
                "gpus": 1,
            }
        ],
        trainable_actor_groups=["actor-a", "actor-b"],
    )
    trainer = object.__new__(TrajWeaveMultiActorCriticSyncTrainer)
    trainer.critic_route_specs = {route.critic_group: route}
    trainer.actor_to_critic_group = {group: route.critic_group for group in route.actor_groups}
    trainer.critic_topology = "centralized"
    trainer.critic_type = "q"
    trainer.multi_actor_trainable_group_ids = ["actor-a", "actor-b"]

    rows = []
    for group in ("actor-b", "actor-a"):
        rows.append(
            {
                "worker_group": group,
                "agent_id": group,
                "traj_uid": "trajectory-0",
                "turn_id": 0,
                "prompt_text": f"prompt-{group}",
                "response_text": f"response-{group}",
                "joint_action_ids": ["joint-action"],
                "joint_transition_ids": ["local-transition"],
                "critic_group": "__missing__",
                "critic_type": "__missing__",
                "critic_input_ids": [],
                "critic_attention_mask": [],
                "critic_position_ids": [],
                "critic_loss_mask": 0.0,
                "response_mask": torch.ones(1, dtype=torch.long),
            }
        )
    keys = ["maac-real-b", "maac-real-a"]
    tq.init()
    try:
        tq.kv_batch_put(
            keys=keys,
            partition_id="train",
            fields=list_of_dict_to_tensordict(rows),
            tags=[{"seq_len": 1}, {"seq_len": 1}],
        )
        trainer._ensure_critic_fields(KVBatchMeta(keys=keys, tags=[{}, {}], partition_id="train"))
        output = tq.kv_batch_get(
            keys=keys,
            partition_id="train",
            select_fields=["critic_group", "critic_input_ids"],
        )
        from trajweave.backends.verl.schema import to_python

        assert to_python(output["critic_group"]) == ["team-critic", "team-critic"]
        assert len(encoded_texts) == 2
        assert encoded_texts[0] == encoded_texts[1]
        assert encoded_texts[0].index("prompt-actor-a") < encoded_texts[0].index("prompt-actor-b")
        assert encoded_texts[0].index("response-actor-a") < encoded_texts[0].index("response-actor-b")
        input_ids = [to_python(output["critic_input_ids"][row]) for row in range(2)]
        assert input_ids[0] == input_ids[1]
    finally:
        tq.kv_clear(keys=keys, partition_id="train")
        tq.close()


def test_maac_deduplicates_one_critic_sample_per_joint_transition_and_skips_padding():
    samples = [
        {
            "traj_uid": "trajectory-0",
            "joint_transition_ids": ["t0"],
            "critic_group": "team",
            "critic_type": "v",
            "critic_input_ids": [1, 2],
            "critic_attention_mask": [1, 1],
            "critic_position_ids": [0, 1],
            "critic_loss_mask": 1.0,
            "joint_reward": 1.0,
            "joint_done": False,
            "joint_truncated": False,
        },
        {
            "traj_uid": "trajectory-0",
            "joint_transition_ids": ["t0"],
            "critic_group": "team",
            "critic_type": "v",
            "critic_input_ids": [1, 2],
            "critic_attention_mask": [1, 1],
            "critic_position_ids": [0, 1],
            "critic_loss_mask": 1.0,
            "joint_reward": 1.0,
            "joint_done": False,
            "joint_truncated": False,
        },
        {
            "joint_transition_ids": ["__padding__:t"],
            "critic_loss_mask": 0.0,
        },
    ]

    assert deduplicate_maac_critic_indices(samples) == [0]
    [unique] = deduplicate_maac_critic_samples(samples)
    assert unique["joint_transition_id"] == "t0"

    different_trajectories = [dict(samples[0]), {**samples[1], "traj_uid": "trajectory-1"}]
    assert deduplicate_maac_critic_indices(different_trajectories) == [0, 1]

    inconsistent = [dict(samples[0]), {**samples[1], "joint_reward": 2.0}]
    with pytest.raises(ValueError, match="inconsistent"):
        deduplicate_maac_critic_samples(inconsistent)


def test_trainer_transition_keys_include_trajectory_id(monkeypatch):
    from types import SimpleNamespace

    from trajweave.backends.verl.trainers import multi_actor_critic_sync as trainer_module

    monkeypatch.setattr(
        trainer_module.tq,
        "kv_batch_get",
        lambda **_kwargs: {
            "traj_uid": ["trajectory-a", "trajectory-b"],
            "joint_transition_ids": [["local-t0"], ["local-t0"]],
        },
    )
    batch = SimpleNamespace(keys=["a", "b"], partition_id="train")
    assert trainer_module._transition_keys(batch) == [
        ("trajectory-a", "local-t0"),
        ("trajectory-b", "local-t0"),
    ]


def test_maac_batch_preparation_updates_central_critic_once_per_transition():
    fields = {
        "response_mask": torch.tensor([[1, 1], [1, 0], [1, 1], [1, 0], [0, 0]]),
        "old_values": torch.tensor([0.5, 0.5, 0.75, 0.75, 999.0]),
        "joint_reward": [1.0, 1.0, 2.0, 2.0, 999.0],
        "joint_done": [False, False, True, True, False],
        "joint_truncated": [False, False, False, False, False],
        "agent_id": ["a", "b", "a", "b", "padding"],
        "worker_group": ["actor-a", "actor-b", "actor-a", "actor-b", "actor-a"],
        "turn_id": [0, 0, 1, 1, 0],
        "traj_uid": ["trajectory"] * 4 + ["padding"],
        "uid": ["group"] * 4 + ["padding"],
        "joint_action_ids": [["a0"], ["a0"], ["a1"], ["a1"], []],
        "joint_transition_ids": [["t0"], ["t0"], ["t1"], ["t1"], ["__padding__:t"]],
        "critic_group": ["team"] * 5,
        "critic_type": ["v"] * 4 + ["__padding__"],
        "critic_input_ids": torch.tensor([[10, 11], [10, 11], [20, 21], [20, 21], [0, 0]]),
        "critic_attention_mask": torch.tensor([[1, 1], [1, 1], [1, 1], [1, 1], [0, 0]]),
        "critic_position_ids": torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1], [0, 0]]),
        "critic_loss_mask": [1.0, 1.0, 1.0, 1.0, 0.0],
    }
    prepared = prepare_actor_critic_batch(
        fields,
        topology="centralized",
        gamma=0.8,
        normalize_by_population_std=False,
    )

    assert prepared.critic_row_indices == (0, 2)
    assert torch.allclose(prepared.scalar_targets, torch.tensor([1.6, 1.6, 2.0, 2.0, 0.0]))
    assert torch.allclose(prepared.scalar_advantages, torch.tensor([1.1, 1.1, 1.25, 1.25, 0.0]))
    assert prepared.critic_fields["input_ids"].shape[0] == 2
    assert prepared.critic_fields["loss_mask"].sum().item() == 2
    assert prepared.actor_advantages[4].sum().item() == 0


def test_maac_truncated_transition_does_not_bootstrap():
    fields = {
        "response_mask": torch.ones(2, 1, dtype=torch.long),
        "old_values": torch.tensor([0.5, 0.5]),
        "joint_reward": [3.0, 3.0],
        "joint_done": [False, False],
        "joint_truncated": [True, True],
        "agent_id": ["a", "b"],
        "worker_group": ["actor-a", "actor-b"],
        "turn_id": [0, 0],
        "traj_uid": ["trajectory", "trajectory"],
        "uid": ["group", "group"],
        "joint_action_ids": [["a0"], ["a0"]],
        "joint_transition_ids": [["t0"], ["t0"]],
        "critic_group": ["team", "team"],
        "critic_type": ["q", "q"],
        "critic_input_ids": torch.tensor([[10], [10]]),
        "critic_attention_mask": torch.tensor([[1], [1]]),
        "critic_position_ids": torch.tensor([[0], [0]]),
        "critic_loss_mask": [1.0, 1.0],
    }
    prepared = prepare_actor_critic_batch(
        fields,
        topology="centralized",
        gamma=0.99,
        normalize_by_population_std=False,
    )
    assert torch.equal(prepared.scalar_targets, torch.tensor([3.0, 3.0]))


def test_maac_rejects_multiple_unique_transitions_at_same_trajectory_turn():
    fields = {
        "response_mask": torch.ones(2, 1, dtype=torch.long),
        "old_values": torch.tensor([0.5, 0.5]),
        "joint_reward": [1.0, 1.0],
        "joint_done": [False, False],
        "joint_truncated": [False, False],
        "agent_id": ["a", "b"],
        "worker_group": ["actor-a", "actor-b"],
        "turn_id": [0, 0],
        "traj_uid": ["trajectory", "trajectory"],
        "uid": ["group", "group"],
        "joint_action_ids": [["a0"], ["a1"]],
        "joint_transition_ids": [["t0"], ["t1"]],
        "critic_group": ["team", "team"],
        "critic_type": ["v", "v"],
        "critic_input_ids": torch.ones(2, 1, dtype=torch.long),
        "critic_attention_mask": torch.ones(2, 1, dtype=torch.long),
        "critic_position_ids": torch.zeros(2, 1, dtype=torch.long),
        "critic_loss_mask": [1.0, 1.0],
    }
    with pytest.raises(ValueError, match="one unique transition"):
        prepare_actor_critic_batch(fields, topology="centralized", gamma=0.9)
