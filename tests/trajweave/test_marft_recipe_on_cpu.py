from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from trajweave.backends.local import extract_final_int
from trajweave.backends.policy import PolicyRequest, PolicyResponse, StableByteTokenizer
from trajweave.backends.verl.batch_padding import pad_session_batch
from trajweave.backends.verl.emitters.marft import MARFTEmitterMixin
from trajweave.backends.verl.emitters.registry import EMITTER_ROUTES
from trajweave.backends.verl.extensions.common.hooks import extension_hooks_for_config
from trajweave.backends.verl.extensions.marft import MARFT_PPO_TQ_FIELDS, MARFTPPOHooks
from trajweave.backends.verl.extensions.registry import _extension_names
from trajweave.backends.verl.trainers import register_trajweave_trainers
from trajweave.backends.verl.trainers.marft_critic_sync import _projected_return_tensor
from trajweave.backends.verl.workflow_runtime import build_hf_workflow_outputs
from trajweave.core.trajectory import AgentTurn, MultiAgentTrajectory
from trajweave.credit.marft import apply_marft_trajectory_credit
from trajweave.envs.math import MathTask
from trajweave.orchestration.marft import MARFTWorkflowGraph, MARFTWorkflowOrchestra
from trajweave.recipes.marft import build_marft_launch_overrides, default_marft_team
from trajweave.recipes.marft.config import resolve_marft_settings
from trajweave.recipes.marft.math_workflow import run_marft_smoke
from trajweave.runner import run_from_config
from verl.protocol import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.trainer.ppo.v1 import get_trainer_cls


@dataclass
class _RecordingBackend:
    prompts: list[tuple[str, str]]

    def generate(self, request: PolicyRequest) -> PolicyResponse:
        self.prompts.append((request.agent.name, request.prompt))
        text = f"output-{request.agent.name}"
        return PolicyResponse(text=text, token_ids=[len(self.prompts)], logprobs=[0.0])


def _trajectory() -> MultiAgentTrajectory:
    trajectory = MultiAgentTrajectory(
        episode_id="episode",
        task_id="task",
        rollout_group="group",
        team_name="marft_math_workflow",
        global_reward=1.0,
    )
    for index, role in enumerate(("planner", "solver", "verifier")):
        trajectory.add_turn(
            AgentTurn(
                episode_id="episode",
                task_id="task",
                turn_id=index,
                agent_name=role,
                role=role,
                policy_group="shared",
                observation="1 + 1",
                prompt=f"prompt-{role}",
                action_text=f"output-{role}",
                action_token_ids=[index + 1],
                joint_action_ids=[f"action-{index}"],
                joint_transition_ids=[f"transition-{index}"],
                done=index == 2,
            )
        )
    return trajectory


def _marft_config(*, independent_critic: str | None = None) -> dict:
    shared = independent_critic is None
    model_ids = ["shared", "shared"] if shared else ["policy_planner", "policy_solver"]
    groups = {
        group: {
            "model_path": f"/models/{group}",
            "tokenizer_path": "/tokenizer",
            "trainable": True,
            "gpus": 1,
        }
        for group in dict.fromkeys(model_ids)
    }
    return {
        "marft": {
            "role_names": ["planner", "solver"],
            "model_ids": model_ids,
            "shared_policy": shared,
            "use_multi_lora": True,
            "shared_lora": shared,
            "independent_critic": independent_critic,
            "kl_coef": 0.0 if independent_critic is not None else 0.1,
            "agent_loop_backend": "synthetic_tq",
            "worker_groups": groups,
        },
        "verl": {"overrides": ["trainer.n_gpus_per_node=4"]},
    }


def test_marft_custom_dag_uses_frozen_history_for_parallel_layer():
    graph = MARFTWorkflowGraph.from_config(
        {
            "nodes": [
                {"id": "plan", "role_name": "planner"},
                {"id": "solve", "role_name": "solver"},
                {"id": "verify", "role_name": "verifier"},
            ],
            "edges": [["plan", "solve"], ["plan", "verify"]],
        }
    )
    backend = _RecordingBackend(prompts=[])
    team = default_marft_team(role_names=("planner", "solver", "verifier"))
    task = MathTask(task_id="math", question="1 + 1", answer=2)
    trajectory = MARFTWorkflowOrchestra(
        graph=graph,
        role_prompts={agent.name: str(agent.prompt_template) for agent in team.agents},
    ).run(
        episode_id="episode",
        rollout_group="group",
        task=task,
        team=team,
        observation=task.question,
        policy_backend=backend,
    )

    assert graph.execution_layers()[1][0].node_id == "solve"
    solve_prompt = backend.prompts[1][1]
    verify_prompt = backend.prompts[2][1]
    assert "planner: output-planner" in solve_prompt
    assert "planner: output-planner" in verify_prompt
    assert "solver: output-solver" not in verify_prompt
    assert trajectory.metadata["shared_history"] == [
        ("planner", "output-planner"),
        ("solver", "output-solver"),
        ("verifier", "output-verifier"),
    ]


@pytest.mark.parametrize(
    ("strategy", "discount", "gamma", "expected_events", "expected_returns"),
    [
        ("equal", 1.0, 0.5, [0.0, 0.0, 1.0], [0.25, 0.5, 1.0]),
        ("step_discount", 0.5, 1.0, [0.25, 0.5, 1.0], [1.75, 1.5, 1.0]),
    ],
)
def test_marft_credit_preserves_sparse_events_and_projects_returns(
    strategy, discount, gamma, expected_events, expected_returns
):
    trajectory = _trajectory()

    returns = apply_marft_trajectory_credit(
        trajectory,
        strategy=strategy,
        discount=discount,
        gamma=gamma,
    )

    assert list(returns) == pytest.approx(expected_returns)
    assert trajectory.metadata["marft_step_rewards"] == pytest.approx(expected_events)
    assert [turn.step_reward for turn in trajectory.turns] == pytest.approx(expected_events)
    assert [turn.reward for turn in trajectory.turns] == pytest.approx(expected_returns)


def test_marft_per_step_credit_uses_role_override_and_global_fallback():
    trajectory = _trajectory()

    apply_marft_trajectory_credit(
        trajectory,
        strategy="per_step",
        gamma=0.5,
        step_reward_fn=lambda **kwargs: kwargs["step"].step_index + 1,
        per_agent_reward_fns={"solver": lambda **kwargs: 10.0},
    )

    assert trajectory.metadata["marft_step_rewards"] == [1.0, 10.0, 3.0]
    assert trajectory.metadata["marft_projected_returns"] == pytest.approx([6.75, 11.5, 3.0])


def test_marft_per_step_callable_receives_upstream_compatible_step_environment_and_data():
    trajectory = _trajectory()
    observed = []

    def reward_fn(*, step, team_reward, env, data):
        observed.append(
            {
                "span": (step.token_start, step.token_end),
                "messages": list(env.messages),
                "metadata": dict(env.metadata),
                "answer": data["answer"],
                "team_reward": team_reward,
            }
        )
        return float(step.token_end - step.token_start)

    apply_marft_trajectory_credit(
        trajectory,
        strategy="per_step",
        step_reward_fn=reward_fn,
        data={
            "raw_prompt": [{"role": "user", "content": "What is 1 + 1?"}],
            "reward_model": {"ground_truth": "2"},
        },
    )

    assert [item["span"] for item in observed] == [(0, 1), (1, 2), (2, 3)]
    assert observed[0]["messages"][0] == {"role": "user", "content": "What is 1 + 1?"}
    assert [message["agent_name"] for message in observed[0]["messages"][1:]] == [
        "planner",
        "solver",
        "verifier",
    ]
    assert observed[0]["answer"] == "2"
    assert "trajectory_metadata" in observed[0]["metadata"]
    assert observed[0]["team_reward"] == 1.0


def test_marft_registry_and_smoke_pipeline_are_executable():
    result = run_from_config(
        {
            "recipe": "marft.math",
            "mode": "smoke",
            "marft": {"role_names": ["planner", "solver"]},
            "backend": {"type": "rule", "device": "cpu"},
            "rollout": {"rollouts_per_task": 1},
        }
    )

    assert result["canonical_recipe"] == "marft.cooperative_math"
    assert result["trajectories"] == 2
    assert result["samples"] == 4
    assert result["success_rate"] == 1.0


@pytest.mark.parametrize(
    "roles",
    [
        ("planner", "solver"),
        ("planner", "solver", "verifier"),
        ("planner", "solver", "reflector", "verifier"),
    ],
)
def test_marft_rule_smoke_evaluates_the_shared_completion_for_supported_team_sizes(roles):
    summary, result = run_marft_smoke(role_names=roles, rollouts_per_task=1)

    assert summary.success_rate == 1.0
    assert all("solver: Reasoning:" in trajectory.final_answer for trajectory in result.trajectories)


class _FakeMARFTWorker(MARFTEmitterMixin):
    def __init__(
        self,
        *,
        responses: list[str] | None = None,
        role_names: tuple[str, ...] = ("planner", "solver"),
    ):
        self.responses = list(responses or ["Plan: add the values.", "Final answer: 2"])
        self.generation_configs = []
        self.tokenizer = StableByteTokenizer()
        self.model_config = SimpleNamespace(local_path="/models/shared")
        role_configs = {
            role: {
                "system_prompt": f"Act as the {role}.",
                **({"temperature": 0.2, "top_p": 0.9} if role == "solver" else {}),
            }
            for role in role_names
        }
        self.config = {
            "agent": {
                "agent_ids": list(role_names),
                "model_ids": ["shared"] * len(role_names),
                "orchestra": {
                    "marft": {
                        "role_names": list(role_names),
                        "role_configs": role_configs,
                        "graph_config": {
                            "nodes": [{"id": role, "role_name": role} for role in role_names],
                            "edges": [list(edge) for edge in zip(role_names, role_names[1:], strict=False)],
                        },
                        "credit_strategy": "equal",
                        "credit_discount": 1.0,
                        "return_gamma": 0.5,
                    }
                },
            }
        }

    def _encode_prompt_text(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def _decode_response_ids(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def _generate_local_response_ids(self, prompt_ids: list[int], **kwargs) -> list[int]:
        del prompt_ids
        self.generation_configs.append(kwargs["generation_config"])
        return self.tokenizer.encode(self.responses.pop(0))

    def _local_policy_version(self) -> int:
        return 3

    def _worker_group_model_path(self, group_id: str) -> str:
        return f"/models/{group_id}"


def test_marft_hf_emitter_builds_shared_history_rows_and_projected_rewards():
    worker = _FakeMARFTWorker()
    outputs = build_hf_workflow_outputs(
        worker,
        recipe="marft_math_workflow",
        prompt={
            "uid": "math-1",
            "raw_prompt": [{"role": "user", "content": "What is 1 + 1?"}],
            "reward_model": {"ground_truth": "2"},
        },
        session_id=0,
    )

    assert "marft_math_workflow" in EMITTER_ROUTES
    assert [output.extra_fields["trajweave_agent_name"] for output in outputs] == ["planner", "solver"]
    assert [output.reward_score for output in outputs] == pytest.approx([0.5, 1.0])
    assert [output.extra_fields["marft_step_reward"] for output in outputs] == [0.0, 1.0]
    assert "planner: Plan: add the values." in outputs[1].extra_fields["prompt_text"]
    assert all(output.extra_fields["joint_action_ids"] for output in outputs)
    assert all(output.extra_fields["joint_transition_ids"] for output in outputs)
    assert worker.generation_configs[1] == {"temperature": 0.2, "top_p": 0.9}


def test_marft_math_judge_uses_verifier_self_correction_from_shared_completion():
    worker = _FakeMARFTWorker(
        role_names=("planner", "solver", "verifier"),
        responses=[
            "Plan: compute carefully.",
            "Final answer: 4",
            "Correction: the solver made an arithmetic error. Final answer: 5",
        ],
    )

    outputs = build_hf_workflow_outputs(
        worker,
        recipe="marft_math_workflow",
        prompt={
            "uid": "math-correction",
            "raw_prompt": [{"role": "user", "content": "What is 2 + 3?"}],
            "reward_model": {"ground_truth": "5"},
        },
        session_id=0,
    )

    assert extract_final_int(outputs[-1].extra_fields["final_answer"]) == 5
    assert outputs[-1].reward_score == 1.0
    assert outputs[-1].extra_fields["marft_step_reward"] == 1.0


@pytest.mark.parametrize(
    ("ground_truth", "prediction"),
    [
        (r"\boxed{\frac{1}{2}}", "Final answer: 0.5"),
        (r"\sqrt{4}", "Final answer: 2"),
    ],
)
def test_marft_hf_runtime_preserves_string_ground_truth_for_math_verification(ground_truth, prediction):
    worker = _FakeMARFTWorker(responses=["Plan: simplify the expression.", prediction])

    outputs = build_hf_workflow_outputs(
        worker,
        recipe="marft_math_workflow",
        prompt={
            "uid": "math-symbolic",
            "raw_prompt": [{"role": "user", "content": "Evaluate the expression."}],
            "reward_model": {"ground_truth": ground_truth},
        },
        session_id=0,
    )

    assert outputs[-1].reward_score == 1.0
    assert outputs[-1].extra_fields["marft_step_reward"] == 1.0


def _marft_hook_batch() -> DataProto:
    response_mask = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.long)
    rewards = torch.zeros(2, 3)
    rewards[:, 1] = torch.tensor([0.5, 1.0])
    non_tensors = {
        field: np.array(
            {
                "agent_id": ["planner", "solver"],
                "policy_group": ["shared", "shared"],
                "worker_group": ["shared", "shared"],
                "traj_uid": ["trajectory", "trajectory"],
                "turn_id": [0, 1],
                "marft_node_id": ["plan", "solve"],
                "marft_layer": [0, 1],
                "marft_role_index": [0, 1],
                "marft_credit_strategy": ["equal", "equal"],
                "marft_credit_discount": [1.0, 1.0],
                "marft_return_gamma": [0.5, 0.5],
                "marft_step_reward": [0.0, 1.0],
                "marft_projected_return": [0.5, 1.0],
            }[field],
            dtype=object,
        )
        for field in MARFT_PPO_TQ_FIELDS
    }
    return DataProto.from_dict(
        tensors={"response_mask": response_mask, "token_level_rewards": rewards},
        non_tensors=non_tensors,
    )


def test_marft_hook_validates_credit_then_delegates_to_gae():
    called = {}

    def fallback(data, **kwargs):
        called.update(kwargs)
        data.batch["advantages"] = torch.ones_like(data.batch["token_level_rewards"])
        data.batch["returns"] = torch.ones_like(data.batch["token_level_rewards"])
        return data

    hooks = MARFTPPOHooks()
    result = hooks.compute_advantage(
        _marft_hook_batch(),
        batch_keys=["math_0_0", "math_0_1"],
        adv_estimator=AdvantageEstimator.GAE,
        gamma=0.5,
        lam=1.0,
        num_repeat=1,
        norm_adv_by_std_in_grpo=True,
        config={},
        fallback=fallback,
    )

    assert called["gamma"] == 0.5
    assert "advantages" in result.batch
    selected_fields = hooks.tq_select_fields("advantage")
    assert "marft_projected_return" in selected_fields
    assert {"uid", "response_mask", "rm_scores", "old_log_probs", "ref_log_prob", "values"} <= set(selected_fields)
    assert isinstance(
        extension_hooks_for_config({"trajweave": {"credit_allocator": "marft_ctde"}}),
        MARFTPPOHooks,
    )


def test_marft_hook_rejects_reward_contamination():
    data = _marft_hook_batch()
    data.batch["token_level_rewards"][0, 1] = 9.0

    with pytest.raises(ValueError, match="differ from marft_projected_return"):
        MARFTPPOHooks().compute_advantage(
            data,
            batch_keys=["math_0_0", "math_0_1"],
            adv_estimator=AdvantageEstimator.GAE,
            gamma=0.5,
            lam=1.0,
            num_repeat=1,
            norm_adv_by_std_in_grpo=True,
            config={},
            fallback=lambda data, **kwargs: data,
        )


def test_marft_hook_validates_pre_kl_scores_and_preserves_kl_rewards():
    data = _marft_hook_batch()
    data.batch["token_level_scores"] = data.batch["token_level_rewards"].clone()
    data.batch["token_level_rewards"] -= 0.05 * data.batch["response_mask"]
    expected = torch.tensor([[-0.05, 0.35, 0.0], [-0.05, 0.95, 0.0]])

    result = MARFTPPOHooks().compute_advantage(
        data,
        batch_keys=["math_0_0", "math_0_1"],
        adv_estimator=AdvantageEstimator.GAE,
        gamma=1.0,
        lam=1.0,
        num_repeat=1,
        norm_adv_by_std_in_grpo=True,
        config={},
        fallback=lambda data, **kwargs: data,
    )

    torch.testing.assert_close(result.batch["token_level_rewards"], expected)


def test_marft_hook_projects_future_kl_by_trajectory_after_batch_reordering_and_ignores_padding():
    first = _marft_hook_batch()
    second = _marft_hook_batch()
    second.non_tensor_batch["traj_uid"][:] = "trajectory-2"
    second.batch["token_level_rewards"][:, 1] = torch.tensor([2.0, 3.0])
    second.non_tensor_batch["marft_projected_return"][:] = [2.0, 3.0]
    padded = first.select_idxs([0])
    padded.batch["response_mask"].zero_()
    padded.batch["token_level_rewards"].zero_()
    padded.non_tensor_batch["marft_projected_return"][:] = 0.0
    data = DataProto.concat([first, second, padded]).select_idxs([3, 0, 4, 2, 1])
    data.batch["token_level_scores"] = data.batch["token_level_rewards"].clone()
    row_adjustments = torch.tensor([-0.4, -0.1, 0.0, -0.3, -0.2])
    data.batch["token_level_rewards"][:, 0] += row_adjustments

    result = MARFTPPOHooks().compute_advantage(
        data,
        batch_keys=["trajectory-2-1", "trajectory-1-0", "padding", "trajectory-2-0", "trajectory-1-1"],
        adv_estimator=AdvantageEstimator.GAE,
        gamma=1.0,
        lam=1.0,
        num_repeat=1,
        norm_adv_by_std_in_grpo=True,
        config={},
        fallback=lambda data, **kwargs: data,
    )

    expected = data.batch["token_level_scores"].clone()
    expected[:, 0] += row_adjustments
    expected[1, 1] += -0.2
    expected[3, 1] += -0.4
    torch.testing.assert_close(result.batch["token_level_rewards"], expected)


def test_marft_padding_clears_credit_metadata_and_does_not_bias_metrics():
    source = {
        "worker_group": "shared",
        "reqs_id": "request",
        "response_mask": torch.ones(3, dtype=torch.long),
        "loss_mask": torch.ones(3, dtype=torch.long),
        "rm_scores": torch.tensor([0.0, 0.0, 1.0]),
        "marft_node_id": "solve",
        "marft_layer": 1,
        "marft_role_index": 1,
        "marft_credit_strategy": "equal",
        "marft_credit_discount": 1.0,
        "marft_return_gamma": 1.0,
        "marft_step_reward": 1.0,
        "marft_projected_return": 1.0,
        "extra_fields": {
            "reqs_id": "request",
            "marft_node_id": "solve",
            "marft_step_reward": 1.0,
            "marft_projected_return": 1.0,
        },
    }
    keys = ["math_0_0"]
    fields = [source]
    tags = [{"seq_len": 3, "response_len": 3}]

    pad_session_batch(keys=keys, fields=fields, tags=tags, multiple=2, uid="math", session_id=0)

    assert len(fields) == 2
    padding = fields[1]
    assert not bool(padding["response_mask"].any())
    assert padding["marft_node_id"].startswith("__trajweave_pad__")
    assert padding["marft_layer"] == -1
    assert padding["marft_step_reward"] == 0.0
    assert padding["marft_projected_return"] == 0.0
    assert padding["extra_fields"]["marft_step_reward"] == 0.0
    assert padding["extra_fields"]["marft_projected_return"] == 0.0

    data = _marft_hook_batch()
    padded_data = DataProto.concat([data, data.select_idxs([0])])
    padded_data.batch["response_mask"][-1].zero_()
    padded_data.non_tensor_batch["marft_step_reward"][-1] = 100.0
    padded_data.non_tensor_batch["marft_projected_return"][-1] = 100.0
    metrics = MARFTPPOHooks().compute_extra_metrics(padded_data, {}, "advantage")
    assert metrics["trajweave/marft/step_reward_mean"] == pytest.approx(0.5)
    assert metrics["trajweave/marft/projected_return_mean"] == pytest.approx(0.75)


def test_marft_shared_and_independent_critic_launch_plans_are_routed():
    shared = build_marft_launch_overrides(_marft_config(), config_path="marft.yaml")
    independent = build_marft_launch_overrides(
        _marft_config(independent_critic="lora"),
        config_path="marft.yaml",
    )

    assert "trainer.v1.trainer_mode=trajweave_marft_shared_critic_sync" in shared
    assert "actor_rollout_ref.model.lora_rank=32" in shared
    assert "critic.model.lora_rank=32" in shared
    assert "algorithm.use_kl_in_reward=true" in shared
    assert "algorithm.kl_penalty=low_var_kl" in shared
    assert "algorithm.kl_ctrl.kl_coef=0.1" in shared
    assert "algorithm.lam=1.0" in shared
    assert "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1" in shared
    assert "trainer.v1.trainer_mode=trajweave_marft_independent_critic_sync" in independent
    assert any(value.startswith("+trajweave.actor_critic=") for value in independent)
    assert _extension_names({"trajweave": {"recipe": "marft_math_workflow"}}) == ("trajweave_marft_ctde",)


def test_marft_independent_critic_registration_and_projected_target_validation():
    register_trajweave_trainers()

    assert get_trainer_cls("trajweave_marft_shared_critic_sync").__name__ == "MARFTSharedCriticSyncTrainer"
    assert get_trainer_cls("trajweave_marft_multi_actor_sync").__name__ == "MARFTMultiActorSyncTrainer"
    assert get_trainer_cls("trajweave_marft_independent_critic_sync").__name__ == ("MARFTIndependentCriticSyncTrainer")
    tensor = _projected_return_tensor([0.25, 1.0], row_count=2, like=torch.zeros(2))
    torch.testing.assert_close(tensor, torch.tensor([0.25, 1.0]))
    with pytest.raises(ValueError, match="must be finite"):
        _projected_return_tensor([float("nan")], row_count=1, like=torch.zeros(1))


def test_marft_config_rejects_unsupported_multi_head_and_missing_per_step_callable():
    config = _marft_config(independent_critic="lora")
    config["marft"]["independent_critic"] = "multi_head"
    with pytest.raises(ValueError, match="multi_head is not supported"):
        resolve_marft_settings(config)

    with pytest.raises(ValueError, match="per_step credit requires"):
        resolve_marft_settings({"marft": {"credit_strategy": "per_step"}})


def test_marft_verl_rejects_nonunit_gamma_and_kl_without_actor_lora():
    gamma_config = _marft_config()
    gamma_config["marft"]["return_gamma"] = 0.5
    with pytest.raises(ValueError, match="requires return_gamma=1.0"):
        build_marft_launch_overrides(gamma_config, config_path="marft.yaml")

    no_lora_config = _marft_config()
    no_lora_config["marft"]["use_multi_lora"] = False
    with pytest.raises(ValueError, match="kl_coef>0 requires use_multi_lora=true"):
        resolve_marft_settings(no_lora_config)


@pytest.mark.parametrize("independent_critic", [None, "lora"])
def test_marft_verl_rejects_value_critic_hf_export(independent_critic):
    config = _marft_config(independent_critic=independent_critic)
    config["verl"]["overrides"].append("critic.checkpoint.save_contents=[model,optimizer,extra,hf_model]")

    with pytest.raises(ValueError, match="value critic checkpoints.*hf_model"):
        build_marft_launch_overrides(config, config_path="marft.yaml")
