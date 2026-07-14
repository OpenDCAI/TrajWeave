from __future__ import annotations

from collections import UserDict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from trajweave.backends.policy import StableByteTokenizer
from trajweave.backends.verl.batch_padding import pad_session_batch
from trajweave.backends.verl.schema import flatten_token_ids
from trajweave.backends.verl.trainers.maporl_multi_actor import _worker_groups_from_config
from trajweave.backends.verl.weight_sync import sync_hf_local_rollout_weights
from trajweave.backends.verl.workflow_runtime import build_hf_workflow_outputs


class FakeWorkflowWorker:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.tokenizer = StableByteTokenizer()
        self.model_config = SimpleNamespace(local_path="/models/default")
        self.config = {
            "agent": {
                "agent_ids": ["agent_0", "agent_1"],
                "model_ids": ["group_0", "group_1"],
                "orchestra": {
                    "math": {"max_loop_num": 2},
                    "search": {"max_loop_num": 1},
                    "maporl": {
                        "max_rounds": 1,
                        "consensus_threshold": 2,
                        "early_stop": True,
                        "criteria_for_consensus_percentage": 0.75,
                        "criteria_for_consensus_reward_threshold": 0.6,
                        "policy_separation": False,
                        "collaboration_separation": False,
                        "task_training": True,
                    },
                    "agentflow": {"max_steps": 1, "enabled_tools": ["base_generator"]},
                    "gigpo": {"max_steps": 2},
                    "comas": {
                        "num_rounds": 1,
                        "num_references": 1,
                        "task_name": "math",
                        "assignment_seed": 7,
                    },
                    "matpo": {"max_turns": 3},
                },
            }
        }

    def _encode_prompt_text(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def _decode_response_ids(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def _generate_local_response_ids(self, prompt_ids: list[int], **kwargs) -> list[int]:
        del prompt_ids, kwargs
        if not self.responses:
            raise AssertionError("Fake workflow exhausted its configured responses.")
        return self.tokenizer.encode(self.responses.pop(0))

    def _local_policy_version(self) -> int:
        return 7

    def _maporl_worker_group_model_path(self, group_id: str) -> str:
        return f"/models/{group_id}"

    def _worker_group_model_path(self, group_id: str) -> str:
        return f"/models/{group_id}"

    def _maporl_agent_ids(self) -> list[str]:
        return ["agent_0", "agent_1"]

    def _maporl_model_ids(self, *, default_agent_ids: list[str]) -> list[str]:
        assert default_agent_ids == ["agent_0", "agent_1"]
        return ["group_0", "group_1"]

    def _maporl_max_rounds(self) -> int:
        return 1

    def _maporl_consensus_threshold(self, *, default: int) -> int:
        assert default == 2
        return 2

    def _maporl_early_stop(self) -> bool:
        return True

    def _maporl_reward_feedback(self) -> bool:
        return False

    def _maporl_consensus_percentage(self) -> float:
        return 0.75

    def _maporl_consensus_reward_threshold(self) -> float:
        return 0.6

    def _maporl_policy_separation(self) -> bool:
        return False

    def _maporl_collaboration_separation(self) -> bool:
        return False

    def _maporl_task_training(self) -> bool:
        return True

    def _agentflow_max_steps(self) -> int:
        return 1

    def _agentflow_enabled_tools(self) -> list[str]:
        return ["base_generator"]

    def _comas_agent_ids(self) -> list[str]:
        return ["agent_0", "agent_1"]

    def _comas_model_ids(self, *, default_agent_ids: list[str]) -> list[str]:
        assert default_agent_ids == ["agent_0", "agent_1"]
        return ["group_0", "group_1"]

    def _comas_num_rounds(self) -> int:
        return 1

    def _comas_num_references(self) -> int:
        return 1

    def _comas_task_name(self) -> str:
        return "math"

    def _comas_assignment_seed(self) -> int:
        return 7


def _math_prompt() -> dict:
    return {
        "uid": "math-1",
        "raw_prompt": [{"role": "user", "content": "What is 1 + 1?"}],
        "reward_model": {"ground_truth": "2"},
        "extra_info": {},
    }


def _search_prompt() -> dict:
    return {
        "uid": "browse-1",
        "raw_prompt": [{"role": "user", "content": "Which city is the capital of France?"}],
        "reward_model": {"ground_truth": "Paris"},
        "extra_info": {
            "search_query": "capital France",
            "documents": [
                {"title": "France", "text": "France is a country in Europe. Its capital city is Paris."},
                {"title": "Germany", "text": "Germany's capital city is Berlin."},
            ],
        },
    }


def test_flatten_token_ids_accepts_tokenizer_batch_encoding_shape():
    encoded = UserDict({"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]})
    assert flatten_token_ids(encoded) == [1, 2, 3]


def test_dynamic_turn_padding_has_zero_loss_and_distinct_credit_group():
    field = {
        "response_mask": torch.ones(4, dtype=torch.int64),
        "loss_mask": torch.ones(4, dtype=torch.int64),
        "rm_scores": torch.ones(4),
        "rollout_log_probs": torch.ones(4),
        "extra_fields": {"agent_id": "answer", "traj_uid": "real"},
    }
    keys, fields, tags = ["real_0_0"], [field], [{"seq_len": 8, "response_len": 4}]

    pad_session_batch(keys=keys, fields=fields, tags=tags, multiple=4, uid="real", session_id=0)

    assert len(fields) == 4
    assert all(item["loss_mask"].sum().item() == 0 for item in fields[1:])
    assert all(item["rm_scores"].sum().item() == 0 for item in fields[1:])
    assert all(item["agent_id"] == "__padding__" for item in fields[1:])
    assert all(tag["is_padding"] for tag in tags[1:])
    assert keys[1:] == ["real_0_-1", "real_0_-2", "real_0_-3"]


def test_dynamic_turn_padding_preserves_real_worker_group_for_multi_actor_routing():
    fields = []
    for group in ("actor_a", "actor_b"):
        fields.append(
            {
                "worker_group": group,
                "policy_group": group,
                "response_mask": torch.ones(2, dtype=torch.int64),
                "loss_mask": torch.ones(2, dtype=torch.int64),
                "rm_scores": torch.ones(2),
                "raw_score": 1.0,
                "correctness": 1.0,
                "extra_fields": {"worker_group": group},
            }
        )
    keys = ["real_0_0", "real_0_1"]
    tags = [{"seq_len": 4}, {"seq_len": 4}]

    pad_session_batch(keys=keys, fields=fields, tags=tags, multiple=2, uid="real", session_id=0)

    assert [field["worker_group"] for field in fields] == ["actor_a", "actor_b", "actor_a", "actor_b"]
    assert [field["agent_id"] for field in fields[2:]] == ["__padding__", "__padding__"]
    assert [field["raw_score"] for field in fields[2:]] == [0.0, 0.0]
    assert keys[2:] == ["real_0_-1", "real_0_-2"]


def test_hf_drmas_reward_comes_from_generated_answer_not_session_parity():
    outputs = build_hf_workflow_outputs(
        FakeWorkflowWorker(["Final answer: 2", "APPROVED"]),
        recipe="doctor_mas_math",
        prompt=_math_prompt(),
        session_id=1,
    )

    assert [output.reward_score for output in outputs] == [1.0, 1.0]
    assert [output.extra_fields["trajweave_agent_name"] for output in outputs] == ["solver", "verifier"]
    assert {output.extra_fields["policy_version"] for output in outputs} == {7}
    assert outputs[0].extra_fields["prompt_text"].startswith("Task:")
    assert outputs[0].extra_fields["response_text"] == "Final answer: 2"
    assert outputs[0].extra_fields["final_answer"] == "Final answer: 2"


def test_hf_maporl_keeps_real_turn_scores_and_worker_routing():
    outputs = build_hf_workflow_outputs(
        FakeWorkflowWorker(["Reasoning. Final answer: 2", "Final answer: 2"]),
        recipe="maporl_debate_math",
        prompt=_math_prompt(),
        session_id=3,
    )

    assert [output.extra_fields["worker_group"] for output in outputs] == ["group_0", "group_1"]
    assert [output.extra_fields["worker_group_model_path"] for output in outputs] == [
        "/models/group_0",
        "/models/group_1",
    ]
    assert [output.extra_fields["raw_score"] for output in outputs] == [1.0, 1.0]
    assert all(output.extra_fields["finished_round"] == 0 for output in outputs)
    assert all(output.extra_fields["policy_separation"] is False for output in outputs)
    assert all(output.extra_fields["collaboration_separation"] is False for output in outputs)
    assert all(output.extra_fields["task_training"] is True for output in outputs)


def test_hf_search_requires_public_documents_and_never_uses_ground_truth_as_evidence():
    prompt = {
        "uid": "search-1",
        "raw_prompt": [{"role": "user", "content": "Which city is the capital of France?"}],
        "reward_model": {"ground_truth": "Paris"},
        "extra_info": {"search_query": "capital France"},
    }

    with pytest.raises(ValueError, match="extra_info.documents"):
        build_hf_workflow_outputs(
            FakeWorkflowWorker([]),
            recipe="doctor_mas_search",
            prompt=prompt,
            session_id=0,
        )


def test_hf_matpo_emits_parent_child_metadata_for_planner_worker_browse():
    outputs = build_hf_workflow_outputs(
        FakeWorkflowWorker([
            "CALL search_and_browse: capital France",
            "Evidence summary: France capital is Paris. Suggested final answer: Paris",
            "Final answer: Paris",
        ]),
        recipe="matpo_browse",
        prompt=_search_prompt(),
        session_id=0,
    )

    assert [output.extra_fields["trajweave_agent_name"] for output in outputs] == [
        "browsing_agent",
        "planner",
    ]
    assert [output.reward_score for output in outputs] == [1.0, 1.0]
    child = outputs[0].extra_fields
    parent_ids = {outputs[1].extra_fields["reqs_id"]}
    assert child["is_from_subagent_tool"] is True
    assert child["parent_reqs_id"] in parent_ids
    assert outputs[1].extra_fields["matpo_tool_call_count"] == 1


def test_hf_agentflow_projects_only_trainable_planner_and_keeps_frozen_trace():
    outputs = build_hf_workflow_outputs(
        FakeWorkflowWorker(["Context: arithmetic\nSub-Goal: solve 1 + 1\nTool Name: base_generator"]),
        recipe="agentflow_planner_tool",
        prompt=_math_prompt(),
        session_id=0,
    )

    assert len(outputs) == 1
    assert outputs[0].extra_fields["trajweave_agent_name"] == "planner"
    assert outputs[0].reward_score == 1.0
    assert {event["role"] for event in outputs[0].extra_fields["agentflow_trace"]} == {
        "executor",
        "tool",
        "verifier",
    }


def test_hf_agentflow_invalid_planner_action_cannot_receive_success_reward():
    outputs = build_hf_workflow_outputs(
        FakeWorkflowWorker(["I will solve it somehow."]),
        recipe="agentflow_planner_tool",
        prompt=_math_prompt(),
        session_id=0,
    )

    assert outputs[0].extra_fields["plan_valid"] is False
    assert outputs[0].reward_score == 0.0
    assert "Tool error" in outputs[0].extra_fields["tool_result"]
    tool_events = [event for event in outputs[0].extra_fields["agentflow_trace"] if event["role"] == "tool"]
    assert [event["agent_name"] for event in tool_events] == ["invalid_tool"]


def test_hf_agentflow_accepts_allowed_tool_without_requiring_redundant_context_label():
    outputs = build_hf_workflow_outputs(
        FakeWorkflowWorker(["base_generator\nSub-Goal: calculate 1 + 1"]),
        recipe="agentflow_planner_tool",
        prompt=_math_prompt(),
        session_id=0,
    )

    assert outputs[0].extra_fields["plan_valid"] is True
    assert outputs[0].extra_fields["tool_name"] == "base_generator"
    assert outputs[0].reward_score == 1.0


def test_hf_gigpo_emits_solver_steps_with_sparse_reward_and_transition_fields():
    outputs = build_hf_workflow_outputs(
        FakeWorkflowWorker(["Final answer: 0", "REVISE", "Final answer: 2"]),
        recipe="gigpo_solver_verifier_math",
        prompt=_math_prompt(),
        session_id=0,
    )

    assert len(outputs) == 2
    assert [output.extra_fields["agent_id"] for output in outputs] == ["solver", "solver"]
    assert [output.extra_fields["step_reward"] for output in outputs] == [0.0, 1.0]
    assert all(output.extra_fields["active_mask"] == 1.0 for output in outputs)
    assert "verifier_feedback" in outputs[0].extra_fields["anchor_obs"]
    assert "REVISE" in outputs[0].extra_fields["next_obs"]


def test_hf_gigpo_does_not_let_verifier_approve_a_wrong_math_answer():
    outputs = build_hf_workflow_outputs(
        FakeWorkflowWorker(["Final answer: 0", "APPROVED", "Final answer: 2"]),
        recipe="gigpo_solver_verifier_math",
        prompt=_math_prompt(),
        session_id=0,
    )

    assert len(outputs) == 2
    assert outputs[-1].reward_score == 1.0
    assert outputs[-1].extra_fields["step_reward"] == 1.0


def test_hf_comas_emits_source_aligned_interaction_rewards_and_worker_routing():
    outputs = build_hf_workflow_outputs(
        FakeWorkflowWorker(
            [
                "Reasoning. \\boxed{2}",
                "Reasoning. \\boxed{3}",
                "No fatal issue.",
                "The answer is incorrect.",
                "Looks correct. <score>3</score>",
                "Fatal error. <score>1</score>",
            ]
        ),
        recipe="comas_peer_review_math",
        prompt=_math_prompt(),
        session_id=0,
    )

    assert len(outputs) == 6
    assert [output.extra_fields["comas_stage"] for output in outputs] == [
        "solver",
        "solver",
        "evaluator",
        "evaluator",
        "scorer",
        "scorer",
    ]
    assert [output.reward_score for output in outputs] == [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    for stage in ("solver", "evaluator", "scorer"):
        groups = {
            output.extra_fields["worker_group"] for output in outputs if output.extra_fields["comas_stage"] == stage
        }
        assert groups == {"group_0", "group_1"}
    assert {output.extra_fields["workflow_evaluation_reward"] for output in outputs} == {0.5}
    assert all(output.extra_fields["reward_source"] == "comas_interaction_only" for output in outputs)


def test_agentflow_planner_prompt_uses_the_configured_tool_name():
    from trajweave.orchestration.agentflow import AgentFlowMemory, AgentFlowPlannerToolOrchestra

    prompt = AgentFlowPlannerToolOrchestra(tool_name="python_stub")._planner_prompt(
        "What is 1 + 1?",
        AgentFlowMemory(),
        step_id=1,
        max_steps=1,
    )

    assert "Tool Name: python_stub" in prompt
    assert "Tool Name: base_generator" not in prompt


class FakeActorWorkerGroup:
    def export_hf_rollout_snapshot(self, path: str, global_step: int, max_ckpt_to_keep: int):
        assert global_step == 4
        assert max_ckpt_to_keep == 2
        hf_path = Path(path) / "huggingface"
        hf_path.mkdir(parents=True)
        (hf_path / "config.json").write_text("{}", encoding="utf-8")
        (hf_path / "model.safetensors").write_bytes(b"weights")


class FakeAgentLoopManager:
    def __init__(self):
        self.calls = []

    def release_local_models(self):
        self.calls.append(("release",))
        return [{"released_groups": ["shared"], "policy_version": 3}]

    def reload_local_models(self, model_paths: dict[str, str], *, policy_version: int):
        self.calls.append(("reload", model_paths, policy_version))
        return [{"policy_version": policy_version}]


def test_hf_local_weight_sync_exports_and_reloads_current_actor(tmp_path: Path):
    manager = FakeAgentLoopManager()
    trainer = type("Trainer", (), {})()
    trainer.config = OmegaConf.create(
        {
            "trajweave": {"agent_loop_backend": "hf_local_tq"},
            "trainer": {"default_local_dir": str(tmp_path)},
        }
    )
    trainer.global_steps = 4
    trainer.actor_rollout_wg = FakeActorWorkerGroup()
    trainer.agent_loop_manager = manager

    paths = sync_hf_local_rollout_weights(trainer)

    assert paths == {"__default__": str(tmp_path / "rollout_sync/global_step_4/actor/huggingface")}
    assert manager.calls == [("release",), ("reload", paths, 4)]


def test_maporl_worker_group_gpu_count_is_parsed_and_validated():
    config = OmegaConf.create(
        {
            "agent": {
                "worker_groups": {
                    "qwen-a": {
                        "model_path": "/models/a",
                        "tokenizer_path": "/models/a",
                        "trainable": True,
                        "gpus": 2,
                    }
                }
            }
        }
    )
    assert _worker_groups_from_config(config)[0].gpus == 2

    config.agent.worker_groups["qwen-a"].gpus = 0
    with pytest.raises(ValueError, match="positive integer"):
        _worker_groups_from_config(config)
