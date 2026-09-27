from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from trajweave.backends.verl.extensions.comlrl.actor_critic import (
    prepare_actor_critic_batch,
    verl_unclipped_mse_critic_loss,
)
from trajweave.backends.verl.workflow_runtime import _trajectory_to_outputs
from trajweave.core.specs import AgentSpec, PolicyGroupSpec, TeamSpec
from trajweave.core.trajectory import AgentTurn, MultiAgentTrajectory
from trajweave.credit.comlrl.actor_critic import (
    build_iac_q_critic_text,
    build_iac_v_critic_text,
    iac_shared_head_clipped_value_loss,
    normalize_agent_turn_advantages,
    one_step_td_targets,
    ratio_free_sequence_actor_loss,
    unclipped_mse_critic_loss,
)


def test_iac_v_and_q_text_use_local_prompt_and_q_appends_action():
    v_text = build_iac_v_critic_text("local history", agent_name="alice")
    q_text = build_iac_q_critic_text("local history", "chosen action", agent_name="alice")

    assert v_text == "local history"
    assert q_text == v_text + "chosen action"
    assert "chosen action" not in v_text


def test_iac_workflow_bridge_preserves_text_without_using_actor_tokenizer_for_critic():
    team = TeamSpec(
        name="iac-runtime",
        agents=(AgentSpec(name="alice", role="solver", policy_group="actor-a", trainable=True),),
        policy_groups=(PolicyGroupSpec(name="actor-a", backend="local", trainable=True),),
        orchestra="test",
        reward="test",
        credit="test",
    )
    trajectory = MultiAgentTrajectory(
        episode_id="episode-iac",
        task_id="task-iac",
        rollout_group="group-iac",
        team_name=team.name,
        global_reward=1.0,
        success=True,
        turns=[
            AgentTurn(
                episode_id="episode-iac",
                task_id="task-iac",
                turn_id=0,
                agent_name="alice",
                role="solver",
                policy_group="actor-a",
                observation="observation",
                prompt="local history",
                action_text="chosen action",
                action_token_ids=[1],
                reward=1.0,
                done=True,
                completion_id="completion-a",
                tree_node_id="node-a",
                joint_action_ids=["action-a"],
                joint_transition_ids=["transition-a"],
            )
        ],
    )
    worker = SimpleNamespace(
        config={
            "trajweave": {
                "actor_critic": {
                    "topology": "independent",
                    "critic_type": "q",
                    "critic_routes": [
                        {
                            "id": "critic-a",
                            "actor_groups": ["actor-a"],
                            "model_path": "/models/critic-a",
                            "tokenizer_path": "/tokenizers/critic-a",
                            "gpus": 1,
                        }
                    ],
                }
            }
        },
        _encode_prompt_text=lambda text: [ord(text[0])],
        _local_policy_version=lambda: 0,
    )

    [output] = _trajectory_to_outputs(worker, trajectory=trajectory, team=team)

    assert output.extra_fields["prompt_text"] == "local history"
    assert output.extra_fields["response_text"] == "chosen action"
    assert output.extra_fields["critic_group"].startswith("__missing__")
    assert output.extra_fields["critic_input_ids"] == []
    assert output.extra_fields["critic_attention_mask"] == []
    assert output.extra_fields["critic_position_ids"] == []
    assert output.extra_fields["critic_loss_mask"] == 0.0
    assert output.extra_fields["joint_reward"] == 1.0
    assert output.extra_fields["joint_done"] is True


def test_terminal_aware_one_step_td_matches_two_turn_hand_calculation():
    result = one_step_td_targets(
        rewards=torch.tensor([1.0, 2.0, 3.0]),
        old_values=torch.tensor([0.5, 0.75, 1.0]),
        next_old_values=torch.tensor([0.75, 99.0, 99.0]),
        done=torch.tensor([False, True, False]),
        truncated=torch.tensor([False, False, True]),
        gamma=0.8,
    )

    assert torch.allclose(result.targets, torch.tensor([1.6, 2.0, 3.0]))
    assert torch.allclose(result.advantages, torch.tensor([1.1, 1.25, 2.0]))


def test_population_normalization_uses_agent_turn_groups_and_preserves_singletons():
    normalized = normalize_agent_turn_advantages(
        torch.tensor([1.0, 3.0, 10.0, 14.0, 7.0, 99.0]),
        agent_ids=["a", "a", "b", "b", "a", "padding"],
        turn_ids=[0, 0, 1, 1, 3, 0],
        active_mask=[True, True, True, True, True, False],
    )

    assert torch.allclose(normalized, torch.tensor([-1.0, 1.0, -1.0, 1.0, 7.0, 0.0]))
    raw = normalize_agent_turn_advantages(
        torch.tensor([1.0, 3.0]), agent_ids=["a", "a"], turn_ids=[0, 0], enabled=False
    )
    assert torch.equal(raw, torch.tensor([1.0, 3.0]))


def test_ratio_free_actor_and_separate_critic_losses_have_separate_gradients():
    actor = torch.nn.Parameter(torch.tensor([[0.2, 0.3], [0.4, 0.5]]))
    critic = torch.nn.Parameter(torch.tensor([0.0, 2.0]))
    mask = torch.tensor([[1, 1], [1, 0]])

    actor_loss = ratio_free_sequence_actor_loss(actor, torch.tensor([2.0, -1.0]), mask)
    assert actor_loss.item() == pytest.approx(-0.3)
    actor_loss.backward()
    assert actor.grad is not None
    assert critic.grad is None

    critic_loss = unclipped_mse_critic_loss(critic, torch.tensor([1.0, 0.0]))
    assert critic_loss.item() == pytest.approx(2.5)
    critic_loss.backward()
    assert critic.grad is not None

    padded = ratio_free_sequence_actor_loss(
        torch.tensor([[0.2, float("nan")]], requires_grad=True),
        torch.tensor([1.0]),
        torch.tensor([[1, 0]]),
    )
    assert padded.item() == pytest.approx(-0.2)


def test_iac_shared_value_helper_is_clipped_but_separate_critic_mse_is_not():
    prediction = torch.tensor([3.0], requires_grad=True)
    old = torch.tensor([0.0])
    target = torch.tensor([1.0])

    shared = iac_shared_head_clipped_value_loss(prediction, old, target, clip_range=0.2)
    separate = unclipped_mse_critic_loss(prediction, target)
    assert shared.item() == pytest.approx(4.0)
    assert separate.item() == pytest.approx(4.0)

    with pytest.raises(FloatingPointError, match="non-finite"):
        ratio_free_sequence_actor_loss(torch.tensor([[float("nan")]]), torch.tensor([1.0]), torch.ones(1, 1))


def test_iac_batch_preparation_excludes_padding_and_uses_critic_inputs():
    fields = {
        "response_mask": torch.tensor([[1, 1], [1, 0], [0, 0]]),
        "old_values": torch.tensor([0.5, 0.75, 999.0]),
        "joint_reward": [1.0, 2.0, 999.0],
        "joint_done": [False, True, False],
        "joint_truncated": [False, False, False],
        "agent_id": ["a", "a", "padding"],
        "worker_group": ["actor-a", "actor-a", "actor-a"],
        "turn_id": [0, 1, 0],
        "traj_uid": ["trajectory", "trajectory", "padding"],
        "uid": ["group", "group", "padding"],
        "joint_action_ids": [["a0"], ["a1"], []],
        "joint_transition_ids": [["t0"], ["t1"], ["__padding__:t"]],
        "critic_group": ["critic-a", "critic-a", "critic-a"],
        "critic_type": ["v", "v", "__padding__"],
        "critic_input_ids": torch.tensor([[10, 11], [20, 21], [0, 0]]),
        "critic_attention_mask": torch.tensor([[1, 1], [1, 1], [0, 0]]),
        "critic_position_ids": torch.tensor([[0, 1], [0, 1], [0, 0]]),
        "critic_loss_mask": [1.0, 1.0, 0.0],
    }
    prepared = prepare_actor_critic_batch(fields, topology="independent", gamma=0.8, normalize_by_population_std=False)

    assert torch.allclose(prepared.scalar_targets, torch.tensor([1.6, 2.0, 0.0]))
    assert prepared.critic_row_indices == (0, 1)
    assert torch.equal(prepared.critic_fields["input_ids"], fields["critic_input_ids"][:2])
    assert prepared.critic_fields["loss_mask"].sum().item() == 2
    assert prepared.actor_advantages[2].sum().item() == 0


def test_iac_2d_old_values_and_targets_use_last_valid_token_with_left_padding():
    fields = {
        "response_mask": torch.ones(2, 1, dtype=torch.long),
        "old_values": torch.tensor([[99.0, 0.5, 0.75], [99.0, 99.0, 1.25]]),
        "joint_reward": [1.0, 2.0],
        "joint_done": [True, True],
        "joint_truncated": [False, False],
        "agent_id": ["a", "b"],
        "worker_group": ["actor-a", "actor-b"],
        "turn_id": [0, 0],
        "traj_uid": ["trajectory", "trajectory"],
        "uid": ["group", "group"],
        "joint_action_ids": [["a0"], ["a1"]],
        "joint_transition_ids": [["t0"], ["t1"]],
        "critic_group": ["critic-a", "critic-b"],
        "critic_type": ["v", "v"],
        "critic_input_ids": torch.tensor([[0, 10, 11], [0, 0, 20]]),
        "critic_attention_mask": torch.tensor([[0, 1, 1], [0, 0, 1]]),
        "critic_position_ids": torch.tensor([[0, 0, 1], [0, 0, 0]]),
        "critic_loss_mask": [1.0, 1.0],
    }

    prepared = prepare_actor_critic_batch(
        fields,
        topology="independent",
        gamma=0.8,
        normalize_by_population_std=False,
    )

    assert torch.allclose(prepared.scalar_advantages, torch.tensor([0.25, 0.75]))
    assert torch.equal(prepared.critic_fields["loss_mask"], torch.tensor([[0.0, 0.0, 1.0]] * 2))
    assert torch.equal(prepared.critic_fields["returns"], torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]))


def test_iac_rejects_multiple_candidates_for_same_trajectory_agent_turn():
    fields = {
        "response_mask": torch.ones(2, 1, dtype=torch.long),
        "old_values": torch.zeros(2),
        "joint_reward": [1.0, 2.0],
        "joint_done": [False, True],
        "joint_truncated": [False, False],
        "agent_id": ["a", "a"],
        "worker_group": ["actor-a", "actor-a"],
        "turn_id": [0, 0],
        "traj_uid": ["trajectory", "trajectory"],
        "uid": ["group", "group"],
        "joint_action_ids": [["a0"], ["a1"]],
        "joint_transition_ids": [["t0"], ["t1"]],
        "critic_group": ["critic-a", "critic-a"],
        "critic_type": ["v", "v"],
        "critic_input_ids": torch.ones(2, 1, dtype=torch.long),
        "critic_attention_mask": torch.ones(2, 1, dtype=torch.long),
        "critic_position_ids": torch.zeros(2, 1, dtype=torch.long),
        "critic_loss_mask": [1.0, 1.0],
    }
    with pytest.raises(ValueError, match="one transition per .*agent, turn"):
        prepare_actor_critic_batch(fields, topology="independent", gamma=0.9)

    fields["turn_id"] = [0, 1]
    fields["joint_transition_ids"] = [["t0", "branch"], ["t1"]]
    with pytest.raises(ValueError, match="exactly one joint transition"):
        prepare_actor_critic_batch(fields, topology="independent", gamma=0.9)


def test_verl_critic_mse_uses_identical_dense_and_nested_token_positions():
    dense_values = torch.tensor([[0.0, 9.0], [3.0, 7.0]], requires_grad=True)
    dense_data = {
        "returns": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        "loss_mask": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
    }
    dense_loss, dense_metrics = verl_unclipped_mse_critic_loss(None, {"values": dense_values}, dense_data)
    assert dense_loss.item() == pytest.approx(1.5)
    assert dense_metrics["critic/mse_loss"] == pytest.approx(2.5)
    assert dense_metrics["critic/value_loss_coef"] == pytest.approx(0.6)
    assert dense_metrics["critic/vpred_mean"] == pytest.approx(1.5)

    nested_values = torch.nested.nested_tensor([torch.tensor([0.0, 9.0]), torch.tensor([3.0])])
    nested_data = {
        "returns": torch.nested.nested_tensor([torch.tensor([1.0, 0.0]), torch.tensor([1.0])]),
        "loss_mask": torch.nested.nested_tensor([torch.tensor([1.0, 0.0]), torch.tensor([1.0])]),
    }
    nested_loss, nested_metrics = verl_unclipped_mse_critic_loss(None, {"values": nested_values}, nested_data)
    assert nested_loss.item() == pytest.approx(dense_loss.item())
    assert nested_metrics["critic/vpred_mean"] == pytest.approx(1.5)


def test_scalar_critic_values_are_materialized_for_parent_verl_metrics():
    from trajweave.backends.verl.trainers.multi_actor_critic_sync import _scalar_values_to_response

    values = torch.tensor([0.25, 1.5])
    dense_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    dense = _scalar_values_to_response(values, dense_mask)
    assert torch.equal(dense, torch.tensor([[0.25, 0.25, 0.0], [1.5, 0.0, 0.0]]))

    nested_mask = torch.nested.as_nested_tensor(
        [torch.tensor([1, 1]), torch.tensor([1])],
        layout=torch.jagged,
    )
    nested = _scalar_values_to_response(values, nested_mask)
    assert nested.is_nested
    assert torch.equal(nested.values(), torch.tensor([0.25, 0.25, 1.5]))


def test_iac_rejects_missing_bootstrap_turns():
    fields = {
        "response_mask": torch.ones(2, 1, dtype=torch.long),
        "old_values": torch.tensor([0.5, 0.75]),
        "joint_reward": [1.0, 2.0],
        "joint_done": [False, True],
        "joint_truncated": [False, False],
        "agent_id": ["a", "a"],
        "worker_group": ["actor-a", "actor-a"],
        "turn_id": [0, 2],
        "traj_uid": ["trajectory", "trajectory"],
        "uid": ["group", "group"],
        "joint_action_ids": [["a0"], ["a2"]],
        "joint_transition_ids": [["t0"], ["t2"]],
        "critic_group": ["critic-a", "critic-a"],
        "critic_type": ["v", "v"],
        "critic_input_ids": torch.ones(2, 1, dtype=torch.long),
        "critic_attention_mask": torch.ones(2, 1, dtype=torch.long),
        "critic_position_ids": torch.zeros(2, 1, dtype=torch.long),
        "critic_loss_mask": [1.0, 1.0],
    }

    with pytest.raises(ValueError, match="contiguous"):
        prepare_actor_critic_batch(
            fields,
            topology="independent",
            gamma=0.9,
            normalize_by_population_std=False,
        )
