from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import pytest
import torch

from trajweave.backends.verl.batch_padding import pad_session_batch
from trajweave.backends.verl.schema import (
    COMLRL_EXTRA_FIELDS,
    MAS_EXTRA_FIELDS,
    resolve_comlrl_extra_fields,
    to_python,
)

COMLRL_VALUES = {
    "prompt_text": "prompt",
    "response_text": "answer",
    "completion_id": "completion-1",
    "tree_node_id": "tree-node-1",
    "joint_action_ids": ["joint-11", "joint-12"],
    "joint_return_components": [0.25, 0.75],
    "projected_joint_return": 1.5,
    "joint_advantage": 0.5,
    "effective_projected_joint_return": 1.25,
    "joint_transition_ids": ["transition-11", "transition-12"],
    "joint_reward": 0.8,
    "joint_done": False,
    "joint_truncated": False,
    "joint_stop_reason": "",
    "joint_sampling_mode": "joint",
    "critic_group": "critic-group-1",
    "critic_type": "centralized",
    "critic_input_ids": [21, 22],
    "critic_attention_mask": [1, 1],
    "critic_position_ids": [0, 1],
    "critic_loss_mask": 1.0,
    "preference_pair_id": "pair-1",
    "preference_side": "chosen",
    "chosen_reward": 0.9,
    "rejected_reward": 0.1,
    "candidate_mean": 0.5,
    "preference_loss_mask": 1.0,
    "raw_policy_reward": 0.9,
    "raw_comparator_reward": 0.8,
    "raw_candidate_rewards": [0.9, 0.8],
    "policy_provenance": {"kind": "current"},
    "comparator_provenance": {"kind": "history"},
    "winner_source": "current",
    "loser_source": "history",
}

ID_FIELDS = (
    "completion_id",
    "tree_node_id",
    "critic_group",
    "preference_pair_id",
)
LIST_FIELDS = (
    "joint_action_ids",
    "joint_transition_ids",
    "joint_return_components",
    "critic_input_ids",
    "critic_attention_mask",
    "critic_position_ids",
    "raw_candidate_rewards",
)


def test_comlrl_fields_are_part_of_shared_mas_schema():
    assert set(COMLRL_EXTRA_FIELDS) <= set(MAS_EXTRA_FIELDS)
    assert len(MAS_EXTRA_FIELDS) == len(set(MAS_EXTRA_FIELDS))


def test_comlrl_schema_falls_back_from_empty_ids_to_aliases_or_safe_defaults():
    resolved = resolve_comlrl_extra_fields(
        {"completion_id": None, "reqs_id": "request-1", "tree_node_id": "", "node_id": "node-1"},
        row_id="row-1",
    )

    assert resolved["completion_id"] == "request-1"
    assert resolved["tree_node_id"] == "node-1"
    missing = resolve_comlrl_extra_fields({"completion_id": None}, row_id="row-2")
    assert missing["completion_id"].startswith("__missing__:row-2")
    assert missing["prompt_text"] == ""
    assert missing["response_text"] == ""
    assert missing["critic_input_ids"] == []
    assert missing["critic_loss_mask"] == 0.0


def test_comlrl_schema_normalizes_ids_and_rejects_misaligned_return_components():
    resolved = resolve_comlrl_extra_fields(
        {
            "joint_action_ids": [1, 2],
            "joint_transition_ids": [3, 4],
            "joint_return_components": [0.25, 0.75],
        },
        row_id="row-1",
    )
    assert resolved["joint_action_ids"] == ["1", "2"]
    assert resolved["joint_transition_ids"] == ["3", "4"]

    with pytest.raises(ValueError, match="equal lengths"):
        resolve_comlrl_extra_fields(
            {"joint_action_ids": ["action-1"], "joint_return_components": [0.25, 0.75]},
            row_id="row-bad",
        )


def test_comlrl_schema_requires_action_transition_and_return_alignment():
    with pytest.raises(ValueError, match="action_ids and joint_transition_ids"):
        resolve_comlrl_extra_fields(
            {"joint_action_ids": ["a1", "a2"], "joint_transition_ids": ["t1"]},
            row_id="misaligned-transitions",
        )
    with pytest.raises(ValueError, match="action_ids and joint_return_components"):
        resolve_comlrl_extra_fields(
            {
                "joint_action_ids": ["a1", "a2"],
                "joint_transition_ids": ["t1", "t2"],
                "joint_return_components": [1.0],
            },
            row_id="misaligned-returns",
        )
    with pytest.raises(ValueError, match="transition_ids must be unique"):
        resolve_comlrl_extra_fields(
            {"joint_action_ids": ["a1", "a2"], "joint_transition_ids": ["t1", "t1"]},
            row_id="duplicate-transitions",
        )


def test_comlrl_padding_rows_use_isolated_safe_values_without_copying_lists():
    real_field = {
        "worker_group": "actor-a",
        "response_mask": torch.ones(3, dtype=torch.int64),
        "loss_mask": torch.ones(3, dtype=torch.int64),
        "rm_scores": torch.ones(3, dtype=torch.float32),
        "raw_score": 1.0,
        **COMLRL_VALUES,
        "extra_fields": {"worker_group": "actor-a", **COMLRL_VALUES},
    }
    keys = ["real_0_0"]
    fields = [real_field]
    tags = [{"seq_len": 6, "response_len": 3}]

    pad_session_batch(keys=keys, fields=fields, tags=tags, multiple=3, uid="real", session_id=0)

    assert fields[0]["joint_action_ids"] == ["joint-11", "joint-12"]
    padding_rows = fields[1:]
    assert len(padding_rows) == 2
    assert all(row["worker_group"] == "actor-a" for row in padding_rows)
    assert all(row["loss_mask"].sum().item() == 0 for row in padding_rows)
    assert all(row["raw_score"] == 0.0 for row in padding_rows)
    assert all(row["joint_done"] is True for row in padding_rows)
    assert all(row["joint_truncated"] is True for row in padding_rows)
    assert all(row["joint_stop_reason"] == "padding" for row in padding_rows)
    assert all(row["joint_advantage"] == 0.0 for row in padding_rows)
    assert all(row["effective_projected_joint_return"] == 0.0 for row in padding_rows)
    assert all(row["critic_loss_mask"] == 0.0 for row in padding_rows)
    assert all(row["preference_loss_mask"] == 0.0 for row in padding_rows)
    for field in LIST_FIELDS:
        assert [row[field] for row in padding_rows] == [[], []]
        assert all(row["extra_fields"][field] == [] for row in padding_rows)
    for field in ID_FIELDS:
        values = [row[field] for row in padding_rows]
        assert len(set(values)) == len(values)
        assert all(value.startswith("__padding__") for value in values)
        assert COMLRL_VALUES[field] not in values
        assert [row["extra_fields"][field] for row in padding_rows] == values
    assert all(row["joint_sampling_mode"] == "__padding__" for row in padding_rows)
    assert all(row["critic_type"] == "__padding__" for row in padding_rows)
    assert all(row["prompt_text"] == "" for row in padding_rows)
    assert all(row["response_text"] == "" for row in padding_rows)
    assert all(row["preference_side"] == "__padding__" for row in padding_rows)
    assert all(row["policy_provenance"] == {} for row in padding_rows)
    assert all(row["comparator_provenance"] == {} for row in padding_rows)
    assert all(row["winner_source"] == "__padding__" for row in padding_rows)
    assert all(row["loser_source"] == "__padding__" for row in padding_rows)
    assert all(tag["is_padding"] for tag in tags[1:])


def test_trajectory_metadata_reaches_agent_loop_output_with_safe_missing_defaults():
    from trajweave.backends.verl.workflow_runtime import _trajectory_to_outputs
    from trajweave.core.specs import AgentSpec, PolicyGroupSpec, TeamSpec
    from trajweave.core.trajectory import AgentTurn, MultiAgentTrajectory

    team = TeamSpec(
        name="comlrl-test",
        agents=(AgentSpec(name="actor", role="solver", policy_group="shared", trainable=True),),
        policy_groups=(PolicyGroupSpec(name="shared", backend="local", trainable=True),),
        orchestra="test",
        reward="test",
        credit="test",
        max_turns=1,
    )
    trajectory = MultiAgentTrajectory(
        episode_id="episode-1",
        task_id="task-1",
        rollout_group="group-1",
        team_name=team.name,
        global_reward=1.0,
    )
    trajectory.add_turn(
        AgentTurn(
            episode_id=trajectory.episode_id,
            task_id=trajectory.task_id,
            turn_id=0,
            agent_name="actor",
            role="solver",
            policy_group="shared",
            observation="question",
            prompt="prompt",
            action_text="answer",
            action_token_ids=[1, 2],
            completion_id=COMLRL_VALUES["completion_id"],
            tree_node_id=COMLRL_VALUES["tree_node_id"],
            joint_action_ids=COMLRL_VALUES["joint_action_ids"],
            joint_transition_ids=COMLRL_VALUES["joint_transition_ids"],
            metadata={
                key: value
                for key, value in {**COMLRL_VALUES, "round_id": 4}.items()
                if key not in {"completion_id", "tree_node_id", "joint_action_ids", "joint_transition_ids"}
            },
        )
    )
    worker = SimpleNamespace(
        _encode_prompt_text=lambda text: [ord(char) for char in text],
        _local_policy_version=lambda: 3,
    )

    output = _trajectory_to_outputs(worker, trajectory=trajectory, team=team)[0]

    assert {field: output.extra_fields[field] for field in COMLRL_EXTRA_FIELDS} == COMLRL_VALUES
    assert output.extra_fields["round_id"] == 4

    trajectory.turns[0].metadata = {}
    trajectory.turns[0].completion_id = None
    trajectory.turns[0].tree_node_id = None
    trajectory.turns[0].joint_action_ids = []
    trajectory.turns[0].joint_transition_ids = []
    defaulted = _trajectory_to_outputs(worker, trajectory=trajectory, team=team)[0].extra_fields
    assert all(defaulted[field].startswith("__missing__") for field in ID_FIELDS)
    assert all(defaulted[field] == [] for field in LIST_FIELDS)
    assert defaulted["critic_loss_mask"] == 0.0
    assert defaulted["preference_loss_mask"] == 0.0


def test_training_sample_top_level_joint_fields_share_metadata_bridge():
    from trajweave.backends.verl.workflow_runtime import _metadata_from_turn_or_sample
    from trajweave.core.trajectory import TrainingSample

    sample = TrainingSample(
        sample_id="sample-1",
        episode_id="episode-1",
        task_id="task-1",
        rollout_group="group-1",
        turn_id=0,
        agent_name="actor",
        role="solver",
        policy_group="shared",
        prompt="prompt",
        response="answer",
        response_token_ids=[1],
        response_logprobs=[-0.1],
        reward=1.0,
        completion_id="completion-sample",
        tree_node_id="tree-node-sample",
        joint_action_ids=["joint-action-sample"],
        joint_transition_ids=["transition-sample"],
        metadata={"joint_reward": 0.75},
    )

    metadata = _metadata_from_turn_or_sample(sample)

    assert metadata["completion_id"] == "completion-sample"
    assert metadata["tree_node_id"] == "tree-node-sample"
    assert metadata["joint_action_ids"] == ["joint-action-sample"]
    assert metadata["joint_transition_ids"] == ["transition-sample"]
    assert metadata["joint_reward"] == 0.75

    sample.metadata.update(
        {
            "completion_id": "stale-completion",
            "tree_node_id": None,
            "joint_action_ids": [],
            "joint_transition_ids": ["stale-transition"],
        }
    )
    canonical = _metadata_from_turn_or_sample(sample)
    assert canonical["completion_id"] == "completion-sample"
    assert canonical["tree_node_id"] == "tree-node-sample"
    assert canonical["joint_action_ids"] == ["joint-action-sample"]
    assert canonical["joint_transition_ids"] == ["transition-sample"]

    sample.completion_id = None
    sample.tree_node_id = None
    sample.joint_action_ids = []
    sample.joint_transition_ids = []
    cleared = _metadata_from_turn_or_sample(sample)
    assert cleared["completion_id"] is None
    assert cleared["tree_node_id"] is None
    assert cleared["joint_action_ids"] == []
    assert cleared["joint_transition_ids"] == []


def test_verl_dataproto_adapter_preserves_comlrl_fields():
    from trajweave.backends.verl.dataproto import VerlDataProtoAdapter
    from trajweave.core.trajectory import TrainingSample

    sample = TrainingSample(
        sample_id="sample-dataproto",
        episode_id="episode-dataproto",
        task_id="task-dataproto",
        rollout_group="group-dataproto",
        turn_id=0,
        agent_name="actor",
        role="solver",
        policy_group="shared",
        prompt="prompt",
        response="answer",
        response_token_ids=[1],
        response_logprobs=[-0.1],
        reward=1.0,
        completion_id=COMLRL_VALUES["completion_id"],
        tree_node_id=COMLRL_VALUES["tree_node_id"],
        joint_action_ids=COMLRL_VALUES["joint_action_ids"],
        joint_transition_ids=COMLRL_VALUES["joint_transition_ids"],
        metadata={
            key: value
            for key, value in COMLRL_VALUES.items()
            if key not in {"completion_id", "tree_node_id", "joint_action_ids", "joint_transition_ids"}
        },
    )

    data = VerlDataProtoAdapter().build([sample])

    assert {field: to_python(data.non_tensor_batch[field][0]) for field in COMLRL_EXTRA_FIELDS} == COMLRL_VALUES


def test_credit_assigners_propagate_joint_identifiers():
    from trajweave.core.specs import AgentSpec, PolicyGroupSpec, TeamSpec
    from trajweave.core.trajectory import AgentTurn, MultiAgentTrajectory
    from trajweave.credit.agentflow.planner_only import FlowGRPOPlannerOnlyCreditAssigner
    from trajweave.credit.gigpo.hierarchical import GiGPOCreditAssigner
    from trajweave.credit.global_broadcast import GlobalBroadcastCreditAssigner
    from trajweave.credit.maporl.score_rules import MAPoRLPPOScoreRuleCreditAssigner

    team = TeamSpec(
        name="joint-credit-test",
        agents=(AgentSpec(name="planner", role="planner", policy_group="shared", trainable=True),),
        policy_groups=(PolicyGroupSpec(name="shared", backend="local", trainable=True),),
        orchestra="test",
        reward="test",
        credit="test",
        max_turns=1,
    )
    trajectory = MultiAgentTrajectory(
        episode_id="episode-credit",
        task_id="task-credit",
        rollout_group="group-credit",
        team_name=team.name,
        global_reward=1.0,
        success=True,
        metadata={"finished_round": 0},
    )
    trajectory.add_turn(
        AgentTurn(
            episode_id=trajectory.episode_id,
            task_id=trajectory.task_id,
            turn_id=0,
            agent_name="planner",
            role="planner",
            policy_group="shared",
            observation="question",
            prompt="prompt",
            action_text="answer",
            action_token_ids=[1],
            completion_id="completion-credit",
            tree_node_id="tree-credit",
            joint_action_ids=["action-credit"],
            joint_transition_ids=["transition-credit"],
            metadata={
                "agentflow_stage": "planner_next_step",
                "round_id": 0,
                "agent_index": 0,
                "raw_score": 1.0,
                "correctness": 1.0,
            },
        )
    )

    assigners = (
        GlobalBroadcastCreditAssigner(),
        FlowGRPOPlannerOnlyCreditAssigner(),
        MAPoRLPPOScoreRuleCreditAssigner(),
        GiGPOCreditAssigner(),
    )
    for assigner in assigners:
        [sample] = assigner.assign([trajectory], team)
        assert sample.completion_id == "completion-credit"
        assert sample.tree_node_id == "tree-credit"
        assert sample.joint_action_ids == ["action-credit"]
        assert sample.joint_transition_ids == ["transition-credit"]


def test_comlrl_output_fields_round_trip_through_tq_and_online_rows_on_cpu():
    tq = pytest.importorskip("transfer_queue")
    from trajweave.backends.verl.agent_loop import TrajWeaveSyntheticAgentLoopWorkerTQ
    from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput

    actor_class = TrajWeaveSyntheticAgentLoopWorkerTQ.__ray_actor_class__

    class _Tokenizer:
        pad_token_id = 0
        eos_token_id = 1

    class _Worker:
        _put_outputs = actor_class._put_outputs

        def __init__(self):
            self.config = SimpleNamespace(
                trajweave=SimpleNamespace(recipe=None, turn_padding_multiple=2),
            )
            self.rollout_config = SimpleNamespace(prompt_length=4, response_length=4, temperature=1.0)
            self.tokenizer = _Tokenizer()
            self.online_turn_rows = []

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

        def _write_online_turns(self, runtime, rows):
            self.online_turn_rows.extend(rows)

    uid = uuid.uuid4().hex
    worker = _Worker()
    output = AgentLoopOutput(
        prompt_ids=[1, 2],
        response_ids=[3, 4],
        response_mask=[1, 1],
        reward_score=1.0,
        num_turns=1,
        metrics=AgentLoopMetrics(
            generate_sequences=1.0,
            tool_calls=0.0,
            compute_score=0.0,
            num_preempted=-1,
        ),
        extra_fields={
            "trajweave_agent_name": "actor",
            "trajweave_role": "solver",
            "policy_group": "shared",
            "worker_group": "shared",
            "round_id": 7,
            **COMLRL_VALUES,
        },
    )

    tq.init()
    try:
        asyncio.run(
            worker._put_outputs(
                [output],
                validate=False,
                uid=uid,
                session_id=0,
                global_steps=0,
            )
        )
        metadata = tq.kv_list(partition_id="train")["train"]
        row_keys = sorted(key for key in metadata if key.startswith(f"{uid}_"))
        assert row_keys == [f"{uid}_0_-1", f"{uid}_0_0"]
        data = tq.kv_batch_get(
            keys=[f"{uid}_0_0", f"{uid}_0_-1"],
            partition_id="train",
            select_fields=(*COMLRL_EXTRA_FIELDS, "round_id", "loss_mask"),
        )
        assert {field: to_python(data[field][0]) for field in COMLRL_EXTRA_FIELDS} == COMLRL_VALUES
        assert int(data["round_id"][0]) == 7
        assert data["loss_mask"][1].sum().item() == 0
        assert to_python(data["joint_action_ids"][1]) == []
        assert to_python(data["critic_input_ids"][1]) == []
        assert float(data["critic_loss_mask"][1]) == 0.0
        assert float(data["preference_loss_mask"][1]) == 0.0
        assert str(data["critic_group"][1]).startswith("__padding__")
        assert str(data["preference_pair_id"][1]).startswith("__padding__")

        assert len(worker.online_turn_rows) == 1
        online_metadata = worker.online_turn_rows[0]["metadata"]
        assert {field: online_metadata[field] for field in COMLRL_EXTRA_FIELDS} == COMLRL_VALUES
        assert online_metadata["round_id"] == 7
    finally:
        keys = list(tq.kv_list(partition_id="train").get("train", {}).keys())
        owned_keys = [key for key in keys if key == uid or key.startswith(f"{uid}_")]
        if owned_keys:
            tq.kv_clear(keys=owned_keys, partition_id="train")
        tq.close()
