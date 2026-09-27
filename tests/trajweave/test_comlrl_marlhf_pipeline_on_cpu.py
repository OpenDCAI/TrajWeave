from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from trajweave.backends.verl.trainers.comlrl_staged import (
    MARLHFStagedWorkflow,
    MARLHFStageState,
    RewardModelTrainingStage,
    build_marlhf_preference_config,
    preference_pairs_from_tq_batch,
    prepare_marlhf_online_config,
    resolve_marlhf_rl_dispatch,
)
from trajweave.backends.verl.workers.scalar_head import (
    JointRewardModel,
    RewardModelWorker,
    serialize_joint_reward_text,
)
from trajweave.core.joint_trajectory import JointAction, JointCompletion, JointTransition, JointTreeNode
from trajweave.core.specs import AgentSpec, PolicyGroupSpec, TeamSpec
from trajweave.core.trajectory import MultiAgentTrajectory
from trajweave.credit.comlrl.marlhf import JointPreferenceBatch
from trajweave.credit.comlrl.preference import build_joint_preference_pairs, preference_pairs_to_training_samples


class TinyCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(n_embd=6)
        self.embed = nn.Embedding(128, 6)
        self.lm_head = nn.Linear(6, 128, bias=False)

    def forward(self, input_ids, attention_mask, output_hidden_states, use_cache):
        hidden = self.embed(input_ids)
        return SimpleNamespace(logits=self.lm_head(hidden), hidden_states=(hidden,))


class RecordingTokenizer:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, texts, *, padding, truncation, return_tensors, max_length=None):
        self.calls.append(list(texts))
        rows = [[ord(character) % 127 + 1 for character in text] for text in texts]
        if max_length is not None:
            rows = [row[:max_length] for row in rows]
        width = max(len(row) for row in rows)
        return {
            "input_ids": torch.tensor([row + [0] * (width - len(row)) for row in rows]),
            "attention_mask": torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in rows]),
        }


def _team() -> TeamSpec:
    return TeamSpec(
        name="marlhf-team",
        agents=(
            AgentSpec("alice", "solver", "actor-a"),
            AgentSpec("bob", "reviewer", "actor-b"),
        ),
        policy_groups=(PolicyGroupSpec("actor-a"), PolicyGroupSpec("actor-b")),
        orchestra="joint",
        reward="task",
        credit="marlhf",
    )


def test_joint_reward_text_matches_v141_format_and_team_order_exactly():
    text = serialize_joint_reward_text(
        _team(),
        {"bob": "prompt B", "alice": "prompt A"},
        {"bob": "response B", "alice": "response A"},
    )

    assert text == (
        "Joint multi-agent response\n\n"
        "Agent 1 prompt:\n"
        "prompt A\n\n"
        "Agent 1 response:\n"
        "response A\n\n"
        "Agent 2 prompt:\n"
        "prompt B\n\n"
        "Agent 2 response:\n"
        "response B"
    )


def _trajectory(task_reward) -> MultiAgentTrajectory:
    prompts = {"alice": "actual rollout prompt A", "bob": "actual rollout prompt B"}
    completions = []
    actions = []
    transitions = []
    for candidate_index in range(2):
        completion_ids = {}
        candidate_indices = {}
        for agent_name in ("alice", "bob"):
            completion = JointCompletion(
                completion_id=f"completion-{agent_name}-{candidate_index}",
                tree_node_id="root",
                agent_name=agent_name,
                candidate_index=candidate_index,
                text=f"{agent_name}-response-{candidate_index}",
                token_ids=[candidate_index + 1],
                logprobs=[-0.1],
                metadata={"prompt": prompts[agent_name]},
            )
            completions.append(completion)
            completion_ids[agent_name] = completion.completion_id
            candidate_indices[agent_name] = candidate_index
        action_id = f"action-{candidate_index}"
        transition_id = f"transition-{candidate_index}"
        actions.append(
            JointAction(
                joint_action_id=action_id,
                tree_node_id="root",
                completion_ids=completion_ids,
                candidate_indices=candidate_indices,
                joint_transition_id=transition_id,
                shared_reward=task_reward(candidate_index),
                done=True,
            )
        )
        transitions.append(
            JointTransition(
                joint_transition_id=transition_id,
                joint_action_id=action_id,
                source_tree_node_id="root",
                source_turn=0,
                done=True,
            )
        )
    return MultiAgentTrajectory(
        episode_id="episode-marlhf",
        task_id="task-marlhf",
        rollout_group="group-marlhf",
        team_name="marlhf-team",
        joint_nodes=[JointTreeNode("root", "root", 0)],
        joint_completions=completions,
        joint_actions=actions,
        joint_transitions=transitions,
        metadata={"joint_sampling_mode": "aligned"},
    )


def _workflow(task_calls: list[float] | None = None):
    calls = task_calls if task_calls is not None else []

    def task_reward(value):
        calls.append(float(value))
        return float(value)

    tokenizer = RecordingTokenizer()
    model = JointRewardModel(TinyCausalLM(), freeze_backbone=True)
    optimizer = torch.optim.AdamW(model.reward_head.parameters(), lr=0.05)
    worker = RewardModelWorker(model, tokenizer, _team(), optimizer=optimizer)
    workflow = MARLHFStagedWorkflow(
        task_reward=task_reward,
        reward_worker=worker,
        reward_model_stage=RewardModelTrainingStage(epochs=2, batch_size=1),
    )
    return workflow, tokenizer, calls


def test_transfer_queue_preference_rows_reconstruct_offline_reward_dataset():
    tq = pytest.importorskip("transfer_queue")
    from transfer_queue import KVBatchMeta

    from verl.utils.tensordict_utils import list_of_dict_to_tensordict

    trajectory = _trajectory(lambda candidate_index: float(candidate_index))
    [expected] = build_joint_preference_pairs(trajectory)
    samples = preference_pairs_to_training_samples(trajectory, [expected], _team())
    rows = [
        {
            "preference_pair_id": sample.metadata["preference_pair_id"],
            "preference_side": sample.metadata["preference_side"],
            "chosen_reward": sample.metadata["chosen_reward"],
            "rejected_reward": sample.metadata["rejected_reward"],
            "preference_loss_mask": sample.metadata["preference_loss_mask"],
            "worker_group": sample.policy_group,
            "response_mask": torch.ones(len(sample.response_token_ids), dtype=torch.long),
            "agent_name": sample.agent_name,
            "traj_uid": sample.episode_id,
            "prompt_text": sample.prompt,
            "response_text": sample.response,
            "tree_node_id": sample.tree_node_id,
            "joint_action_ids": sample.joint_action_ids,
            "candidate_mean": sample.metadata["candidate_mean"],
        }
        for sample in samples
    ]
    keys = [f"marlhf-pair-{row}" for row in range(len(rows))]
    batch = KVBatchMeta(keys=keys, tags=[{"seq_len": 2}] * len(rows), partition_id="train")

    tq.init()
    try:
        tq.kv_batch_put(
            keys=keys,
            partition_id="train",
            fields=list_of_dict_to_tensordict(rows),
            tags=batch.tags,
        )
        [actual] = preference_pairs_from_tq_batch(
            batch,
            expected_worker_groups=("actor-a", "actor-b"),
        )
    finally:
        tq.kv_clear(keys=keys, partition_id="train")
        tq.close()

    assert actual.preference_pair_id == expected.preference_pair_id
    assert actual.prompts_by_agent == expected.prompts_by_agent
    assert actual.chosen_by_agent == expected.chosen_by_agent
    assert actual.rejected_by_agent == expected.rejected_by_agent
    assert actual.chosen_reward == expected.chosen_reward
    assert actual.rejected_reward == expected.rejected_reward


def test_staged_workflow_uses_actual_prompts_and_switches_train_eval_rewards():
    workflow, tokenizer, task_calls = _workflow()

    pairs = workflow.collect_preferences(lambda task_reward: [_trajectory(task_reward)])
    assert workflow.state is MARLHFStageState.PREFERENCES_COLLECTED
    assert task_calls == [0.0, 1.0]
    pair = pairs[0]
    expected_chosen = serialize_joint_reward_text(_team(), pair.prompts_by_agent, pair.chosen_by_agent)
    expected_rejected = serialize_joint_reward_text(_team(), pair.prompts_by_agent, pair.rejected_by_agent)
    batch = JointPreferenceBatch.from_pairs(_team(), pairs)
    assert batch.chosen_texts == (expected_chosen,)
    assert batch.rejected_texts == (expected_rejected,)

    scorer = workflow.train_reward_model()
    assert workflow.state is MARLHFStageState.REWARD_MODEL_FROZEN
    assert scorer.is_frozen
    assert workflow.reward_worker.optimizer is None
    assert any(expected_chosen in call for call in tokenizer.calls)
    assert any(expected_rejected in call for call in tokenizer.calls)

    online_prompts = {"bob": "actual online prompt B", "alice": "actual online prompt A"}
    online_responses = {"bob": "online response B", "alice": "online response A"}
    expected_online_text = serialize_joint_reward_text(_team(), online_prompts, online_responses)
    calls_before_online = list(task_calls)

    def runner(*, dispatch, reward_router):
        learned_reward = reward_router.training_reward(online_prompts, online_responses)
        assert task_calls == calls_before_online
        eval_reward = reward_router.evaluation_reward(9.0)
        return dispatch, learned_reward, eval_reward

    dispatch, learned_reward, eval_reward = workflow.run_online_rl(algorithm="magrpo", runner=runner)

    assert dispatch.trainer == "trajweave_multi_actor_sync"
    assert dispatch.advantage_mode == "mean"
    assert isinstance(learned_reward, float)
    assert eval_reward == 9.0
    assert task_calls == [0.0, 1.0, 9.0]
    assert tokenizer.calls[-1] == [expected_online_text]
    assert workflow.state is MARLHFStageState.ONLINE_RL_COMPLETED


def test_state_machine_rejects_skips_repeats_and_prefrozen_rm():
    workflow, _tokenizer, _calls = _workflow()
    with pytest.raises(RuntimeError, match="expected 'preferences_collected'"):
        workflow.train_reward_model()
    with pytest.raises(RuntimeError, match="expected 'reward_model_frozen'"):
        workflow.run_online_rl(algorithm="magrpo", runner=lambda **_kwargs: None)

    workflow.collect_preferences(lambda task_reward: [_trajectory(task_reward)])
    with pytest.raises(RuntimeError, match="expected 'initial'"):
        workflow.collect_preferences(lambda task_reward: [_trajectory(task_reward)])
    workflow.train_reward_model()
    with pytest.raises(RuntimeError, match="expected 'preferences_collected'"):
        workflow.train_reward_model()
    workflow.run_online_rl(algorithm="mareinforce", runner=lambda **_kwargs: "done")
    with pytest.raises(RuntimeError, match="expected 'reward_model_frozen'"):
        workflow.run_online_rl(algorithm="mareinforce", runner=lambda **_kwargs: None)

    prefrozen, _tokenizer, _calls = _workflow()
    prefrozen.collect_preferences(lambda task_reward: [_trajectory(task_reward)])
    prefrozen.reward_worker.freeze_for_evaluation()
    with pytest.raises(RuntimeError, match="frozen before"):
        prefrozen.train_reward_model()


def test_marlhf_config_prepares_registered_online_trainer_and_offline_preference_phase():
    config = OmegaConf.create(
        {
            "trajweave": {
                "comlrl": {
                    "algorithm": "marlhf",
                    "joint_mode": "aligned",
                    "max_turns": 1,
                    "marlhf": {
                        "rl_algorithm": "magrpo",
                        "preference_num_candidates": 5,
                    },
                }
            },
            "actor_rollout_ref": {"rollout": {"n": 2, "val_kwargs": {"n": 2}}},
            "trainer": {"v1": {"trainer_mode": "unused"}},
        }
    )

    dispatch = prepare_marlhf_online_config(config)
    preference_config = build_marlhf_preference_config(config)

    assert dispatch is not None and dispatch.trainer == "trajweave_multi_actor_sync"
    assert config.trajweave.comlrl.algorithm == "magrpo"
    assert config.trajweave.comlrl.marlhf.enabled is True
    assert config.trainer.v1.trainer_mode == "trajweave_multi_actor_sync"
    assert config.actor_rollout_ref.actor.policy_loss.loss_mode == "gpg"
    assert config.actor_rollout_ref.actor.loss_agg_mode == "seq-mean-token-sum"
    assert config.actor_rollout_ref.actor.ppo_epochs == 1
    assert config.critic.enable is False
    assert config.algorithm.adv_estimator == "reinforce_plus_plus"
    assert config.algorithm.extension_hooks_class.endswith("CoMLRLReinforceHooks")
    assert list(config.trajweave.verl_extensions) == ["trajweave_comlrl_reinforce"]
    assert preference_config.trajweave.comlrl.algorithm == "madpo"
    assert preference_config.trajweave.comlrl.marlhf.reward_model_active is False
    assert preference_config.actor_rollout_ref.rollout.n == 5
    assert preference_config.actor_rollout_ref.rollout.val_kwargs.n == 5


def test_marlhf_rejects_legacy_verl_trainer_before_runtime_setup():
    config = OmegaConf.create(
        {
            "trajweave": {
                "comlrl": {
                    "algorithm": "marlhf",
                    "marlhf": {"rl_algorithm": "magrpo"},
                }
            },
            "trainer": {"use_v1": False},
        }
    )

    with pytest.raises(ValueError, match="trainer.use_v1=true"):
        prepare_marlhf_online_config(config)


@pytest.mark.parametrize(
    ("algorithm", "expected_topology", "expected_route_count"),
    [("iac", "independent", 2), ("maac", "centralized", 1)],
)
def test_marlhf_actor_critic_dispatch_materializes_required_trainer_contract(
    algorithm,
    expected_topology,
    expected_route_count,
):
    config = OmegaConf.create(
        {
            "trajweave": {
                "comlrl": {
                    "algorithm": "marlhf",
                    "joint_mode": "aligned",
                    "max_turns": 1,
                    "marlhf": {
                        "rl_algorithm": algorithm,
                        "critic_model_name": "/models/critic",
                    },
                }
            },
            "agent": {"model_ids": ["actor-a", "actor-b"]},
            "actor_rollout_ref": {
                "rollout": {"n": 1},
                "actor": {"policy_loss": {"loss_mode": "vanilla"}, "loss_agg_mode": "token-mean"},
            },
            "critic": {"enable": False},
            "trainer": {"v1": {"trainer_mode": "unused"}},
        }
    )

    dispatch = prepare_marlhf_online_config(config)

    assert dispatch is not None and dispatch.critic_topology == expected_topology
    assert config.critic.enable is True
    assert config.actor_rollout_ref.actor.policy_loss.loss_mode == "gpg"
    assert config.actor_rollout_ref.actor.loss_agg_mode == "seq-mean-token-sum"
    actor_critic = config.trajweave.comlrl.actor_critic
    assert actor_critic.topology == expected_topology
    assert len(actor_critic.critic_routes) == expected_route_count
    assert all(route.model_path == "/models/critic" for route in actor_critic.critic_routes)


@pytest.mark.parametrize(
    ("algorithm", "trainer", "advantage_mode", "critic_topology"),
    [
        ("magrpo", "trajweave_multi_actor_sync", "mean", None),
        ("mareinforce", "trajweave_multi_actor_sync", "raw", None),
        ("marloo", "trajweave_multi_actor_sync", "rloo", None),
        ("maremax", "trajweave_multi_actor_sync", "max", None),
        ("iac", "trajweave_multi_actor_critic_sync", None, "independent"),
        ("maac", "trajweave_multi_actor_critic_sync", None, "centralized"),
    ],
)
def test_six_way_rl_dispatch_is_explicit(algorithm, trainer, advantage_mode, critic_topology):
    dispatch = resolve_marlhf_rl_dispatch(algorithm)

    assert dispatch.algorithm == algorithm
    assert dispatch.trainer == trainer
    assert dispatch.advantage_mode == advantage_mode
    assert dispatch.critic_topology == critic_topology


def test_unknown_rl_dispatch_fails_fast():
    with pytest.raises(ValueError, match="Unsupported MARLHF RL algorithm"):
        resolve_marlhf_rl_dispatch("ppo")
