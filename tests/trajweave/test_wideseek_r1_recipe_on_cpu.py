from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from trajweave.backends.policy import StableByteTokenizer
from trajweave.backends.verl.emitters.wideseek_r1 import WideSeekR1EmitterMixin
from trajweave.backends.verl.extensions.common.hooks import extension_hooks_for_config
from trajweave.backends.verl.extensions.registry import _extension_names
from trajweave.backends.verl.extensions.wideseek_r1 import WideSeekR1GRPOHooks
from trajweave.backends.verl.workflow_runtime import build_hf_workflow_outputs
from trajweave.credit.wideseek_r1 import apply_wideseek_trajectory_reward
from trajweave.envs.search import SearchAnswerEnvironment
from trajweave.recipes.registry import resolve_recipe
from trajweave.recipes.wideseek_r1 import (
    default_wideseek_r1_tasks,
    default_wideseek_r1_team,
    run_wideseek_r1_smoke,
)
from trajweave.recipes.wideseek_r1.config import (
    build_wideseek_r1_launch_overrides,
    resolve_wideseek_r1_settings,
)
from trajweave.recipes.wideseek_r1.validation import validate_wideseek_training_result
from trajweave.runner import run_from_config


def test_wideseek_registry_and_smoke_preserve_width_scaling_contract():
    recipe = resolve_recipe("wideseek-r1")
    summary, result = run_wideseek_r1_smoke(rollouts_per_task=2, max_parallel_subagents=3)

    assert recipe.name == "wideseek_r1.broad_search"
    assert recipe.runtime_recipe == "wideseek_r1_broad_search"
    assert summary.trajectories == 4
    assert summary.success_rate == 0.5
    assert summary.lead_samples == 8
    assert summary.subagent_samples == 36
    assert {sample.policy_group for sample in result.samples} == {"shared_policy"}
    assert sorted({sample.reward for sample in result.samples}) == pytest.approx([0.15, 1.15])

    for trajectory in result.trajectories:
        assert trajectory.metadata["active_subagents"] == 3
        assert trajectory.metadata["agent_count"] == 4
        assert trajectory.metadata["subagent_contexts_isolated"] is True
        worker_turns = [turn for turn in trajectory.turns if turn.role == "subagent"]
        assert {turn.metadata["wideseek_parallel_wave"] for turn in worker_turns} == {1}
        assert len({turn.metadata["wideseek_agent_instance"] for turn in worker_turns}) == 3
        assert all(turn.metadata["wideseek_agent_count"] == 4 for turn in trajectory.turns)


def test_wideseek_subagent_contexts_do_not_leak_sibling_evidence():
    _summary, result = run_wideseek_r1_smoke(rollouts_per_task=1, max_parallel_subagents=3)
    trajectory = result.trajectories[0]
    worker_turns = [turn for turn in trajectory.turns if turn.role == "subagent"]
    by_agent = {
        agent: [turn for turn in worker_turns if turn.agent_name == agent]
        for agent in {turn.agent_name for turn in worker_turns}
    }
    for agent, turns in by_agent.items():
        sibling_names = set(by_agent) - {agent}
        visible = "\n".join(f"{turn.prompt}\n{turn.observation}" for turn in turns)
        assert all(sibling not in visible for sibling in sibling_names)
        assert {turn.metadata["wideseek_subtrajectory_id"] for turn in turns} != {0}


def test_wideseek_team_uses_one_shared_policy_and_role_specific_tools():
    team = default_wideseek_r1_team(max_parallel_subagents=4)

    assert len(team.policy_groups) == 1
    assert {agent.policy_group for agent in team.agents} == {"shared_policy"}
    assert team.agent("lead_agent").tools == ("subagent",)
    assert all(agent.tools == ("search", "access") for agent in team.agents[1:])


def test_search_environment_accepts_wideseek_access_tool():
    task = default_wideseek_r1_tasks()[0]
    environment = SearchAnswerEnvironment()

    assert "Paris" in environment.execute_tool("access", task, "France capital")


def test_wideseek_credit_broadcasts_one_advantage_to_every_agent_in_a_trajectory():
    _summary, result = run_wideseek_r1_smoke(rollouts_per_task=2, max_parallel_subagents=2)
    task_samples = [sample for sample in result.samples if sample.task_id == "wideseek_capital_france"]
    by_episode: dict[str, set[float]] = {}
    for sample in task_samples:
        by_episode.setdefault(sample.episode_id, set()).add(float(sample.advantage))

    assert all(len(values) == 1 for values in by_episode.values())
    assert len({next(iter(values)) for values in by_episode.values()}) == 2
    assert all(sample.metadata["wideseek_shared_trajectory_advantage"] for sample in task_samples)


def test_wideseek_reward_is_gated_by_final_answer_format_not_intermediate_tool_format():
    _summary, result = run_wideseek_r1_smoke(rollouts_per_task=1, max_parallel_subagents=1)
    trajectory = result.trajectories[0]
    trajectory.global_reward = 1.0
    trajectory.turns[1].metadata["wideseek_format_valid"] = False

    assert apply_wideseek_trajectory_reward(trajectory) == pytest.approx(1.15)
    assert trajectory.metadata["wideseek_final_format_valid"] is True
    assert trajectory.metadata["wideseek_all_formats_valid"] is False

    trajectory.turns[-1].metadata["wideseek_format_valid"] = False
    assert apply_wideseek_trajectory_reward(trajectory) == 0.0
    assert trajectory.metadata["wideseek_final_format_valid"] is False


def test_wideseek_dual_level_reweighting_equalizes_agent_token_mass():
    # Two trajectories, each with one lead and one subagent.  The first
    # subagent spans two turns; its combined token mass must still match every
    # other agent's contribution.
    response_mask = torch.tensor(
        [
            [1, 1, 0, 0],  # trajectory 1 lead: 2 tokens
            [1, 0, 0, 0],  # trajectory 1 subagent turn 1: 1 token
            [1, 1, 1, 0],  # trajectory 1 subagent turn 2: 3 tokens
            [1, 0, 0, 0],  # trajectory 2 lead: 1 token
            [1, 0, 0, 0],  # trajectory 2 subagent: 1 token
        ],
        dtype=torch.float32,
    )
    data = SimpleNamespace(
        batch={
            "response_mask": response_mask,
            "advantages": response_mask.clone(),
            "returns": response_mask.clone(),
        },
        non_tensor_batch={
            "traj_uid": ["t1", "t1", "t1", "t2", "t2"],
            "wideseek_agent_instance": ["lead", "worker", "worker", "lead", "worker"],
            "wideseek_agent_count": [2, 2, 2, 2, 2],
        },
        meta_info={},
    )

    WideSeekR1GRPOHooks._apply_dual_level_reweighting(data)

    contributions: dict[tuple[str, str], float] = {}
    for row, (trajectory, agent) in enumerate(
        zip(data.non_tensor_batch["traj_uid"], data.non_tensor_batch["wideseek_agent_instance"], strict=True)
    ):
        contributions[(trajectory, agent)] = contributions.get((trajectory, agent), 0.0) + float(
            data.batch["advantages"][row].sum()
        )
    assert list(contributions.values()) == pytest.approx([2.0] * 4)
    assert data.meta_info["wideseek_dual_level_reweighting"] is True


def test_wideseek_dual_level_reweighting_rejects_missing_agent_rows():
    data = SimpleNamespace(
        batch={
            "response_mask": torch.ones((1, 1)),
            "advantages": torch.ones((1, 1)),
            "returns": torch.ones((1, 1)),
        },
        non_tensor_batch={
            "traj_uid": ["t1"],
            "wideseek_agent_instance": ["lead"],
            "wideseek_agent_count": [2],
        },
        meta_info={},
    )

    with pytest.raises(ValueError, match="declares 2 agents but emitted 1"):
        WideSeekR1GRPOHooks._apply_dual_level_reweighting(data)


def test_wideseek_extension_hook_and_schema_are_registered():
    config = {
        "trajweave": {
            "recipe": "wideseek_r1_broad_search",
            "credit_allocator": "wideseek_r1_multi_agent_grpo",
        }
    }

    assert _extension_names(config) == ("trajweave_wideseek_r1_grpo",)
    assert isinstance(extension_hooks_for_config(config), WideSeekR1GRPOHooks)
    fields = WideSeekR1GRPOHooks().batch_schema_fields("advantage")
    assert {"wideseek_agent_instance", "wideseek_agent_count", "wideseek_subtrajectory_id"} <= set(fields)


def test_wideseek_config_wires_shared_actor_and_algorithm_contract():
    config = {
        "wideseek_r1": {"max_parallel_subagents": 2},
        "verl": {"overrides": ["trainer.use_v1=false", "actor_rollout_ref.actor.ppo_epochs=9"]},
    }
    command = build_wideseek_r1_launch_overrides(config, config_path="configs/wideseek_r1/example.yaml")

    assert "trainer.use_v1=true" in command
    assert "trainer.use_v1=false" not in command
    assert "actor_rollout_ref.actor.ppo_epochs=1" in command
    assert "actor_rollout_ref.actor.ppo_epochs=9" not in command
    assert '+agent.agent_ids=["lead_agent","subagent_0","subagent_1"]' in command
    assert '+agent.model_ids=["shared_policy","shared_policy","shared_policy"]' in command
    assert "+agent.model_sharing=true" in command
    assert "++algorithm.group_by_agent_id=false" in command
    assert "+trajweave.verl_extensions=[trajweave_wideseek_r1_grpo]" in command


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"max_parallel_subagents": 0}, "positive integer"),
        ({"max_parallel_subagents": 33}, "must not exceed 32"),
        ({"agent_loop_backend": "verl_tq"}, "synthetic_tq or hf_local_tq"),
        ({"format_reward": -0.1}, "non-negative"),
    ],
)
def test_wideseek_config_rejects_invalid_contract(settings, message):
    with pytest.raises(ValueError, match=message):
        resolve_wideseek_r1_settings({"wideseek_r1": settings})


def test_wideseek_smoke_and_verl_plan_run_through_recipe_plugin(tmp_path):
    smoke = run_from_config(
        {
            "recipe": "wideseek_r1.broad_search",
            "mode": "smoke",
            "output_dir": str(tmp_path / "runs"),
            "wideseek_r1": {"max_parallel_subagents": 2},
            "rollout": {"rollouts_per_task": 2},
        }
    )
    plan = run_from_config(
        {
            "recipe": "wideseek_r1.broad_search",
            "mode": "verl_plan",
            "output_dir": str(tmp_path / "runs"),
            "prepare": {"tiny_verl_assets": {"enabled": False}},
            "wideseek_r1": {"agent_loop_backend": "synthetic_tq", "max_parallel_subagents": 2},
            "verl": {"enabled": True, "execute": False, "overrides": []},
        },
        config_path="wideseek-plan.yaml",
    )

    assert smoke["samples"] == 32
    assert plan["verl_launch"]["status"] == "dry_run"
    assert "+trajweave.recipe=wideseek_r1_broad_search" in plan["verl_launch"]["command"]


def test_wideseek_training_validation_rejects_zero_advantage_and_gradient(tmp_path):
    stdout_path = tmp_path / "verl_stdout.log"
    stdout_path.write_text(
        "step:1 - trajweave/wideseek_r1/active_agent_instances:2.0 - "
        "trajweave/wideseek_r1/loss_scale_mean:1.0 - critic/advantages/min:0.0 - "
        "critic/advantages/max:0.0 - actor/grad_norm:0.0\n",
        encoding="utf-8",
    )
    output = {
        "verl_launch": {
            "status": "ok",
            "stdout_path": str(stdout_path),
            "expected_training_steps": 1,
        }
    }

    validate_wideseek_training_result(output, run_dir=tmp_path, mode="verl_train")

    assert output["verl_launch"]["status"] == "failed"
    assert "advantages were zero" in output["verl_launch"]["validation_error"]
    assert "gradient norm stayed zero" in output["verl_launch"]["validation_error"]


def test_wideseek_training_validation_accepts_nonzero_update_and_policy_reload(tmp_path):
    stdout_path = tmp_path / "verl_stdout.log"
    stdout_path.write_text(
        "step:2 - trajweave/wideseek_r1/active_agent_instances:2.0 - "
        "trajweave/wideseek_r1/loss_scale_mean:1.2 - critic/advantages/min:-0.5 - "
        "critic/advantages/max:0.5 - actor/grad_norm:3.0\n",
        encoding="utf-8",
    )
    turns = tmp_path / "trajectories" / "online_turns"
    turns.mkdir(parents=True)
    (turns / "worker.jsonl").write_text(
        '{"metadata":{"policy_version":0}}\n{"metadata":{"policy_version":1}}\n',
        encoding="utf-8",
    )
    output = {
        "verl_launch": {
            "status": "ok",
            "stdout_path": str(stdout_path),
            "expected_training_steps": 2,
        }
    }

    validate_wideseek_training_result(output, run_dir=tmp_path, mode="verl_train")

    assert output["verl_launch"]["status"] == "ok"


class FakeWideSeekEmitter(WideSeekR1EmitterMixin):
    def __init__(self):
        self.tokenizer = StableByteTokenizer()
        self.config = {
            "agent": {
                "agent_ids": ["lead_agent", "subagent_0", "subagent_1"],
                "orchestra": {
                    "wideseek_r1": {
                        "max_parallel_subagents": 2,
                        "shared_model_id": "shared_policy",
                    }
                },
            }
        }

    def _encode_prompt(self, prompt):
        return self.tokenizer.encode(str(prompt))

    def _encode_text(self, text):
        return self.tokenizer.encode(text)


class FakeWideSeekWorkflowWorker(FakeWideSeekEmitter):
    def __init__(self, responses: list[str]):
        super().__init__()
        self.responses = list(responses)
        self.model_config = SimpleNamespace(local_path="/models/shared")

    def _encode_prompt_text(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def _decode_response_ids(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def _generate_local_response_ids(self, prompt_ids: list[int], **kwargs) -> list[int]:
        del prompt_ids, kwargs
        return self.tokenizer.encode(self.responses.pop(0))

    def _local_policy_version(self) -> int:
        return 7

    @staticmethod
    def _worker_group_model_path(group_id: str) -> str:
        return f"/models/{group_id}"


def test_wideseek_synthetic_emitter_outputs_all_shared_model_subtrajectories():
    outputs = FakeWideSeekEmitter()._build_wideseek_r1_outputs(
        {
            "raw_prompt": [{"role": "user", "content": "Which city is the capital of France?"}],
            "reward_model": {"ground_truth": "Paris"},
            "extra_info": {"search_query": "capital France"},
        },
        session_id=0,
    )

    assert len(outputs) == 8
    assert {output.extra_fields["worker_group"] for output in outputs} == {"shared_policy"}
    assert [output.extra_fields["wideseek_parallel_wave"] for output in outputs] == [0, 1, 1, 1, 1, 1, 1, 2]
    assert {output.extra_fields["wideseek_agent_count"] for output in outputs} == {3}
    assert list({output.reward_score for output in outputs}) == pytest.approx([1.15])


def test_hf_wideseek_workflow_preserves_isolated_shared_model_rows():
    worker = FakeWideSeekWorkflowWorker(
        [
            "CALL subagent: France capital\nCALL subagent: Paris city",
            "CALL search: France capital",
            "CALL access: France capital",
            "Subagent result: France's capital is Paris.",
            "CALL search: Paris city",
            "CALL access: Paris city",
            "Subagent result: Paris is the target city.",
            "Final answer: Paris",
        ]
    )
    outputs = build_hf_workflow_outputs(
        worker,
        recipe="wideseek_r1_broad_search",
        prompt={
            "uid": "wideseek-hf-1",
            "raw_prompt": [{"role": "user", "content": "Which city is the capital of France?"}],
            "reward_model": {"ground_truth": "Paris"},
            "extra_info": {
                "search_query": "France capital",
                "documents": [{"title": "France", "text": "The capital of France is Paris."}],
            },
        },
        session_id=0,
    )

    assert len(outputs) == 8
    assert {output.extra_fields["worker_group"] for output in outputs} == {"shared_policy"}
    assert {output.extra_fields["worker_group_model_path"] for output in outputs} == {"/models/shared_policy"}
    assert {output.extra_fields["policy_version"] for output in outputs} == {7}
    assert [output.extra_fields["wideseek_subtrajectory_id"] for output in outputs] == [0, 1, 1, 1, 2, 2, 2, 0]
    assert list({output.reward_score for output in outputs}) == pytest.approx([1.15])
