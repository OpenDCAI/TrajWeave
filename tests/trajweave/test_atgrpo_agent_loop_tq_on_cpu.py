import asyncio
import uuid
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def _runtime_config(branch_factor: int):
    return SimpleNamespace(
        trajweave=SimpleNamespace(
            recipe="atgrpo_solver_verifier_math",
            agent_loop_backend="synthetic_tq",
            turn_padding_multiple=1,
        ),
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(n=branch_factor, val_kwargs=SimpleNamespace(n=branch_factor))
        ),
        agent=SimpleNamespace(
            orchestra=SimpleNamespace(
                atgrpo=SimpleNamespace(
                    max_turns=3,
                    mixed_reward=SimpleNamespace(enabled=False, alpha=1.0, verifier_local_reward=1.0),
                )
            )
        ),
    )


def test_atgrpo_agent_loop_writes_and_reads_turn_and_tree_schema_through_real_tq():
    tq = pytest.importorskip("transfer_queue")
    from trajweave.backends.verl.agent_loop import TrajWeaveSyntheticAgentLoopWorkerTQ
    from trajweave.backends.verl.emitters.atgrpo import ATGRPOEmitterMixin
    from trajweave.backends.verl.extensions.common.hooks import ATGRPOHooks

    actor_class = TrajWeaveSyntheticAgentLoopWorkerTQ.__ray_actor_class__

    class _Tokenizer:
        pad_token_id = 0
        eos_token_id = 1

    class _Worker(actor_class):
        def __init__(self):
            self.config = _runtime_config(branch_factor=2)
            self.rollout_config = SimpleNamespace(prompt_length=32, response_length=16, temperature=1.0)
            self.tokenizer = _Tokenizer()
            self.online_turn_rows = []

        @staticmethod
        def _encode_prompt(value):
            return [10, 11]

        @staticmethod
        def _encode_text(value):
            return [ord(char) % 127 for char in value] or [1]

        @staticmethod
        def _local_policy_version():
            return 0

        @staticmethod
        def _compute_multi_modal_inputs(output, input_ids):
            return None

        @staticmethod
        def _compute_position_ids(input_ids, attention_mask, multi_modal_inputs):
            return torch.arange(input_ids.shape[-1], dtype=torch.long).unsqueeze(0)

        def _write_online_turns(self, runtime, rows):
            self.online_turn_rows.extend(rows)

    tq.init()
    uid = uuid.uuid4().hex
    val_uid = uuid.uuid4().hex
    worker = _Worker()
    try:
        asyncio.run(
            worker._run_prompt(
                {
                    "uid": uid,
                    "raw_prompt": [{"role": "user", "content": "What is 2 + 2?"}],
                    "reward_model": {"ground_truth": "4"},
                    "global_steps": 0,
                },
                trajectory={"validate": False},
            )
        )
        metadata = tq.kv_list(partition_id="train")["train"]
        row_keys = sorted(key for key in metadata if key.startswith(f"{uid}_"))
        assert len(row_keys) == 4

        fields = ATGRPOHooks().tq_select_fields("advantage", config={"group_by_agent_id": True})
        selected_fields = (*fields, "local_score", "branch_index")
        data = tq.kv_batch_get(keys=row_keys, partition_id="train", select_fields=selected_fields)
        turn_ids = [int(value) for value in data["turn_id"]]
        node_ids = [str(value) for value in data["node_id"]]
        observation_groups = [str(value) for value in data["observation_group_id"]]
        assert set(turn_ids) == {0, 1}
        assert len(set(node_ids)) == 4
        assert len(set(observation_groups)) == 2
        assert all(observation_groups.count(group_id) == 2 for group_id in set(observation_groups))
        online_groups = {
            group_id: {
                row["turn_id"] for row in worker.online_turn_rows if row["metadata"]["observation_group_id"] == group_id
            }
            for group_id in set(observation_groups)
        }
        assert all(len(logical_turn_ids) == 1 for logical_turn_ids in online_groups.values())
        torch.testing.assert_close(data["rm_scores"].sum(dim=-1), torch.ones(4))
        assert sorted(float(value) for value in data["local_score"]) == [-1.0, 0.0, 1.0, 1.0]

        from verl.protocol import DataProto
        from verl.trainer.ppo.core_algos import AdvantageEstimator

        proto = DataProto.from_dict(
            tensors={
                "token_level_rewards": data["rm_scores"].float(),
                "response_mask": data["response_mask"],
            },
            non_tensors={
                "uid": np.array([str(value) for value in data["uid"]], dtype=object),
                "agent_id": np.array([str(value) for value in data["agent_id"]], dtype=object),
                "traj_uid": np.array([str(value) for value in data["traj_uid"]], dtype=object),
                "turn_id": np.array(turn_ids, dtype=object),
                "observation_group_id": np.array(observation_groups, dtype=object),
            },
        )
        result = ATGRPOHooks().compute_advantage(
            proto,
            batch_keys=row_keys,
            adv_estimator=AdvantageEstimator.GRPO,
            config={"group_by_agent_id": True},
            fallback=None,
        )
        assert torch.isfinite(result.batch["advantages"]).all()

        worker.config.agent.orchestra.atgrpo.mixed_reward.enabled = True
        mixed_outputs = worker._build_atgrpo_solver_verifier_outputs(
            {
                "uid": "mixed",
                "raw_prompt": [{"role": "user", "content": "What is 2 + 2?"}],
                "reward_model": {"ground_truth": "4"},
                "__atgrpo_branch_factor__": 2,
            }
        )
        assert [output.extra_fields["local_score"] for output in mixed_outputs] == [1.0, 0.0, 1.0, -1.0]
        assert [output.reward_score for output in mixed_outputs] == [2.0, 1.0, 2.0, 0.0]
        worker.config.agent.orchestra.atgrpo.mixed_reward.enabled = False

        asyncio.run(
            worker._run_prompt(
                {
                    "uid": val_uid,
                    "raw_prompt": [{"role": "user", "content": "What is 2 + 2?"}],
                    "reward_model": {"ground_truth": "4"},
                    "global_steps": 0,
                },
                trajectory={"validate": True},
            )
        )
        val_metadata = tq.kv_list(partition_id="val")["val"]
        val_row_keys = sorted(key for key in val_metadata if key.startswith(f"{val_uid}_"))
        assert len(val_row_keys) == 4
        val_data = tq.kv_batch_get(keys=val_row_keys, partition_id="val", select_fields=selected_fields)
        val_roots = [str(value) for value in val_data["root_id"]]
        val_groups = [str(value) for value in val_data["observation_group_id"]]
        assert len(set(val_roots)) == 2
        assert all(val_roots.count(root_id) == 2 for root_id in set(val_roots))
        assert len(set(val_groups)) == 4
        assert {int(value) for value in val_data["branch_index"]} == {0}
    finally:
        for partition_id, owned_uid in (("train", uid), ("val", val_uid)):
            keys = list(tq.kv_list(partition_id=partition_id).get(partition_id, {}).keys())
            owned_keys = [key for key in keys if key == owned_uid or key.startswith(f"{owned_uid}_")]
            if owned_keys:
                tq.kv_clear(keys=owned_keys, partition_id=partition_id)
        tq.close()
