from __future__ import annotations

import uuid
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from trajweave.backends.policy import PolicyRequest, PolicyResponse, StableByteTokenizer
from trajweave.backends.verl.emitters.mrlx import MrlXEmitterMixin
from trajweave.backends.verl.extensions.mrlx import MrlXMGRPOHooks
from trajweave.backends.verl.extensions.registry import _extension_names
from trajweave.backends.verl.workflow_runtime import build_hf_workflow_outputs
from trajweave.credit.mrlx import apply_mrlx_trajectory_rewards
from trajweave.envs.search import SearchAnswerEnvironment, SearchDocument, SearchTask
from trajweave.orchestration.mrlx import MrlXResearchOrchestra
from trajweave.recipes.mrlx import build_mrlx_launch_overrides, default_mrlx_team, run_mrlx_smoke
from trajweave.recipes.registry import resolve_recipe
from trajweave.runner import run_from_config


def test_mrlx_registry_and_smoke_expose_two_independent_training_roles():
    recipe = resolve_recipe("m-grpo")
    summary, result = run_mrlx_smoke(rollouts_per_task=2)

    assert recipe.name == "mrlx.mgrpo_research_qa"
    assert recipe.runtime_recipe == "mrlx_research_qa"
    assert summary.trajectories == 4
    assert summary.success_rate == 0.5
    assert summary.explorer_samples == summary.adapter_samples == 8
    assert {sample.policy_group for sample in result.samples} == {"explorer_policy", "adapter_policy"}
    assert {sample.metadata["mrlx_training_mode"] for sample in result.samples} == {
        "on_policy_explorer",
        "off_policy_adapter",
    }

    explorer = [sample for sample in result.samples if sample.agent_name == "main_explorer"]
    adapter = [sample for sample in result.samples if sample.agent_name == "sub_adapter"]
    assert {sample.reward for sample in explorer} == {0.1, 1.0}
    assert {sample.reward for sample in adapter} == {0.1, 1.0}
    assert all(sample.metadata["advantage_group"].endswith(":main_explorer") for sample in explorer)
    assert all(sample.metadata["advantage_group"].endswith(":sub_adapter") for sample in adapter)


def test_mrlx_format_gate_zeroes_only_the_invalid_role():
    tokenizer = StableByteTokenizer()
    responses = {
        "mrlx_delegate": "CALL sub_adapter: capital France",
        "mrlx_adapter_tool": "not a valid tool call",
        "mrlx_adapter_result": "Research result: no evidence",
        "mrlx_final": "Final answer: Paris",
    }

    class ScriptedBackend:
        def generate(self, request: PolicyRequest) -> PolicyResponse:
            text = responses[str(request.metadata["stage"])]
            token_ids = tokenizer.encode(text)
            return PolicyResponse(text=text, token_ids=token_ids, logprobs=[0.0] * len(token_ids))

    task = _search_task()
    team = default_mrlx_team()
    trajectory = MrlXResearchOrchestra().run(
        episode_id="mrlx-invalid-adapter",
        rollout_group=task.task_id,
        task=task,
        team=team,
        observation=task.question,
        policy_backend=ScriptedBackend(),
        environment=SearchAnswerEnvironment(),
    )
    trajectory.global_reward = 1.0
    rewards = apply_mrlx_trajectory_rewards(
        trajectory,
        explorer_agent="main_explorer",
        adapter_agent="sub_adapter",
    )

    assert rewards == {"main_explorer": 1.0, "sub_adapter": 0.0}
    assert trajectory.metadata["mrlx_explorer_format_valid"] is True
    assert trajectory.metadata["mrlx_adapter_format_valid"] is False


def test_mrlx_invalid_explorer_final_gates_correct_outcome_before_adapter_inheritance():
    tokenizer = StableByteTokenizer()
    responses = {
        "mrlx_delegate": "CALL sub_adapter: capital France",
        "mrlx_adapter_tool": "CALL search_and_browse: capital France",
        "mrlx_adapter_result": "Research result: France's capital is Paris.",
        "mrlx_final": "Paris",
    }

    class ScriptedBackend:
        def generate(self, request: PolicyRequest) -> PolicyResponse:
            text = responses[str(request.metadata["stage"])]
            token_ids = tokenizer.encode(text)
            return PolicyResponse(text=text, token_ids=token_ids, logprobs=[0.0] * len(token_ids))

    task = _search_task()
    trajectory = MrlXResearchOrchestra().run(
        episode_id="mrlx-invalid-explorer",
        rollout_group=task.task_id,
        task=task,
        team=default_mrlx_team(),
        observation=task.question,
        policy_backend=ScriptedBackend(),
        environment=SearchAnswerEnvironment(),
    )
    trajectory.global_reward = 1.0

    rewards = apply_mrlx_trajectory_rewards(
        trajectory,
        explorer_agent="main_explorer",
        adapter_agent="sub_adapter",
    )

    assert rewards == {"main_explorer": 0.0, "sub_adapter": 0.1}
    assert trajectory.metadata["mrlx_explorer_format_valid"] is False
    assert trajectory.metadata["mrlx_adapter_format_valid"] is True


def test_mrlx_invalid_delegation_stops_without_creating_adapter_samples():
    tokenizer = StableByteTokenizer()
    responses = {
        "mrlx_delegate": "not a valid delegation",
    }
    stages: list[str] = []

    class ScriptedBackend:
        def generate(self, request: PolicyRequest) -> PolicyResponse:
            stage = str(request.metadata["stage"])
            stages.append(stage)
            text = responses[stage]
            token_ids = tokenizer.encode(text)
            return PolicyResponse(text=text, token_ids=token_ids, logprobs=[0.0] * len(token_ids))

    task = _search_task()
    trajectory = MrlXResearchOrchestra().run(
        episode_id="mrlx-invalid-delegation",
        rollout_group=task.task_id,
        task=task,
        team=default_mrlx_team(),
        observation=task.question,
        policy_backend=ScriptedBackend(),
        environment=SearchAnswerEnvironment(),
    )
    trajectory.global_reward = 1.0

    rewards = apply_mrlx_trajectory_rewards(
        trajectory,
        explorer_agent="main_explorer",
        adapter_agent="sub_adapter",
    )

    assert stages == ["mrlx_delegate"]
    assert rewards == {"main_explorer": 0.0}
    assert all(turn.agent_name != "sub_adapter" for turn in trajectory.turns)
    assert trajectory.metadata["mrlx_stop_reason"] == "invalid_explorer_delegation"
    assert trajectory.metadata["mrlx_adapter_present"] is False


def test_mrlx_main_history_contains_summaries_without_adapter_internal_evidence():
    tokenizer = StableByteTokenizer()
    responses = iter(
        (
            "CALL sub_adapter: first delegated query",
            "CALL search_and_browse: internal tool query one",
            "Research result: first public summary.",
            "CALL sub_adapter: second delegated query",
            "CALL search_and_browse: internal tool query two",
            "Research result: second public summary.",
            "Final answer: Paris",
        )
    )
    requests: list[PolicyRequest] = []

    class CapturingBackend:
        def generate(self, request: PolicyRequest) -> PolicyResponse:
            requests.append(request)
            text = next(responses)
            token_ids = tokenizer.encode(text)
            return PolicyResponse(text=text, token_ids=token_ids, logprobs=[0.0] * len(token_ids))

    task = SearchTask(
        task_id="mrlx-history-visibility",
        question="Which city is the capital of France?",
        answer="Paris",
        search_query="capital France",
        documents=(
            SearchDocument(title="private-source", text="RAW-EVIDENCE-SENTINEL: Paris is the capital."),
        ),
    )
    trajectory = MrlXResearchOrchestra(research_rounds=2).run(
        episode_id="mrlx-history-visibility",
        rollout_group=task.task_id,
        task=task,
        team=default_mrlx_team(research_rounds=2),
        observation=task.question,
        policy_backend=CapturingBackend(),
        environment=SearchAnswerEnvironment(),
    )

    main_requests = [request for request in requests if request.agent.name == "main_explorer"]
    main_visible_text = "\n".join(
        text for request in main_requests for text in (request.prompt, request.team_context)
    )
    assert "first public summary" in main_requests[1].prompt
    assert "first public summary" in main_requests[1].team_context
    assert "first public summary" in main_requests[-1].prompt
    assert "second public summary" in main_requests[-1].prompt
    assert "RAW-EVIDENCE-SENTINEL" not in main_visible_text
    assert "internal tool query" not in main_visible_text
    assert "Sub adapter tool call:" not in main_visible_text
    assert "Tool observation:" not in main_visible_text
    assert trajectory.metadata["research_history"] == (
        "Main explorer: CALL sub_adapter: first delegated query",
        "Sub adapter result: Research result: first public summary.",
        "Main explorer: CALL sub_adapter: second delegated query",
        "Sub adapter result: Research result: second public summary.",
    )

    adapter_turns = [turn for turn in trajectory.turns if turn.agent_name == "sub_adapter"]
    assert any("RAW-EVIDENCE-SENTINEL" in turn.metadata["tool_observation"] for turn in adapter_turns)
    assert any("RAW-EVIDENCE-SENTINEL" in turn.prompt for turn in adapter_turns)


def test_mrlx_credit_normalizes_trajectory_rewards_per_role_not_per_turn():
    summary, result = run_mrlx_smoke(rollouts_per_task=2)
    task_samples = [sample for sample in result.samples if sample.task_id == "mrlx_capital_france"]
    by_episode_role: dict[tuple[str, str], set[float]] = {}
    for sample in task_samples:
        by_episode_role.setdefault((sample.episode_id, sample.agent_name), set()).add(float(sample.advantage))

    assert summary.success_rate == 0.5
    assert all(len(values) == 1 for values in by_episode_role.values())
    explorer_advantages = {
        next(iter(values)) for (episode, role), values in by_episode_role.items() if role == "main_explorer"
    }
    adapter_advantages = {
        next(iter(values)) for (episode, role), values in by_episode_role.items() if role == "sub_adapter"
    }
    assert len(explorer_advantages) == 2
    assert len(adapter_advantages) == 2


def test_mrlx_extension_and_trainer_are_registered():
    pytest.importorskip("transfer_queue")
    from trajweave.backends.verl.trainers import register_trajweave_trainers
    from verl.trainer.ppo.v1 import get_trainer_cls

    register_trajweave_trainers()

    assert _extension_names({"trajweave": {"recipe": "mrlx_research_qa"}}) == ("trajweave_mrlx_mgrpo",)
    assert get_trainer_cls("trajweave_mrlx_async").__name__ == "TrajWeaveMrlXAsyncTrainer"
    assert MrlXMGRPOHooks().name == "mrlx_mgrpo"


def test_mrlx_hook_collapses_repeated_role_turns_before_grpo_normalization():
    rewards = torch.tensor([[1.0], [1.0], [0.0], [0.0]])
    mask = torch.ones_like(rewards)
    advantages, _ = MrlXMGRPOHooks().compute_grpo_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=mask,
        index=np.array(["task:main"] * 4, dtype=object),
        traj_index=np.array(["success", "success", "failure", "failure"], dtype=object),
        group_by_agent_id=True,
    )

    expected = 1.0 / (2.0**0.5)
    torch.testing.assert_close(advantages[:, 0], torch.tensor([expected, expected, -expected, -expected]))


def test_mrlx_config_wires_role_grpo_multi_actor_and_one_step_delay():
    config = {
        "recipe": "mrlx.mgrpo_research_qa",
        "mode": "verl_plan",
        "prepare": {"tiny_verl_assets": {"enabled": False}},
        "mrlx": {
            "explorer_model_id": "main_model",
            "adapter_model_id": "sub_model",
            "worker_groups": {
                "main_model": {"model_path": "/models/main", "tokenizer_path": "/tokenizer", "gpus": 1},
                "sub_model": {"model_path": "/models/sub", "tokenizer_path": "/tokenizer", "gpus": 1},
            },
        },
        "verl": {
            "enabled": True,
            "execute": False,
            "overrides": ["trainer.use_v1=true", "trainer.critic_warmup=99"],
        },
    }
    result = run_from_config(config, config_path="configs/mrlx/mgrpo_research_qa_2gpu.yaml")
    command = result["verl_launch"]["command"]

    assert "trainer.v1.trainer_mode=trajweave_mrlx_async" in command
    assert "trainer.critic_warmup=0" in command
    assert "trainer.critic_warmup=99" not in command
    assert "++algorithm.group_by_agent_id=true" in command
    assert "++algorithm.extension_hooks_class=trajweave.backends.verl.extensions.mrlx.MrlXMGRPOHooks" in command
    assert '+trajweave.mrlx.explorer_group="main_model"' in command
    assert '+trajweave.mrlx.adapter_group="sub_model"' in command
    assert "+trajweave.mrlx.adapter_delay_steps=1" in command
    assert "+agent.orchestra.mrlx.explorer_format_bonus=0.1" in command
    assert "actor_rollout_ref.actor.clip_ratio_high=0.28" in command
    assert result["mrlx"]["adapter_update"] == "off_policy_one_step_lag"


@pytest.mark.parametrize(
    ("mrlx", "message"),
    [
        ({"explorer_model_id": "same", "adapter_model_id": "same"}, "distinct"),
        ({"adapter_delay_steps": 2}, "adapter_delay_steps=1"),
        ({"research_rounds": 2}, "research_rounds=1"),
        ({"research_rounds": "1.5"}, "positive integer"),
        ({"tokenizer_mode": "compatible"}, "tokenizer_mode=shared"),
        ({"tool_name": "shell"}, "tool_name"),
    ],
)
def test_mrlx_config_rejects_invalid_algorithm_contract(mrlx, message):
    with pytest.raises(ValueError, match=message):
        build_mrlx_launch_overrides({"mrlx": mrlx}, config_path=None)


class FakeMrlXWorkflowWorker(MrlXEmitterMixin):
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.tokenizer = StableByteTokenizer()
        self.model_config = SimpleNamespace(local_path="/models/default")
        self.config = {
            "agent": {
                "agent_ids": ["main_explorer", "sub_adapter"],
                "model_ids": ["explorer_policy", "adapter_policy"],
                "orchestra": {
                    "mrlx": {
                        "research_rounds": 1,
                        "tool_name": "search_and_browse",
                        "adapter_format_bonus": 0.1,
                    }
                },
            }
        }

    def _encode_prompt_text(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def _decode_response_ids(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def _generate_local_response_ids(self, prompt_ids: list[int], **kwargs) -> list[int]:
        del prompt_ids, kwargs
        return self.tokenizer.encode(self.responses.pop(0))

    def _local_policy_version(self) -> int:
        return 3

    @staticmethod
    def _worker_group_model_path(group_id: str) -> str:
        return f"/models/{group_id}"


def test_hf_mrlx_workflow_routes_both_models_and_applies_role_rewards():
    outputs = build_hf_workflow_outputs(
        FakeMrlXWorkflowWorker(
            [
                "CALL sub_adapter: capital France",
                "CALL search_and_browse: capital France",
                "Research result: France's capital is Paris.",
                "Final answer: Paris",
            ]
        ),
        recipe="mrlx_research_qa",
        prompt=_search_prompt(),
        session_id=0,
    )

    assert [output.extra_fields["worker_group"] for output in outputs] == [
        "explorer_policy",
        "adapter_policy",
        "adapter_policy",
        "explorer_policy",
    ]
    assert [output.reward_score for output in outputs] == [1.0, 1.0, 1.0, 1.0]
    assert [output.extra_fields["mrlx_policy_lag"] for output in outputs] == [0, 1, 1, 0]
    assert {output.extra_fields["policy_version"] for output in outputs} == {3}
    assert outputs[1].extra_fields["worker_group_model_path"] == "/models/adapter_policy"


def test_mrlx_adapter_replay_allows_a_later_batch_without_adapter_samples():
    tq = pytest.importorskip("transfer_queue")
    from tensordict import TensorDict
    from transfer_queue import KVBatchMeta

    from trajweave.backends.verl.trainers.mrlx_async import TrajWeaveMrlXAsyncTrainer

    class FakeActorGroup:
        world_size = 1

        def __init__(self):
            self.phases: list[str] = []

        def update_actor(self, route):
            self.phases.append(str(route.extra_info["mrlx_update_phase"]))
            return {"metrics": {"loss": [float(len(route.keys))]}}

    trainer = object.__new__(TrajWeaveMrlXAsyncTrainer)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "calculate_entropy": False,
                    "entropy_coeff": 0.0,
                    "ppo_mini_batch_size": 1,
                    "ppo_epochs": 1,
                    "data_loader_seed": 0,
                    "shuffle": False,
                },
                "rollout": {"n": 1, "temperature": 1.0},
            },
            "trajweave": {
                "recipe": "mrlx_research_qa",
                "multi_actor": {"metric_namespace": "mrlx"},
                "mrlx": {"drain_adapter_replay": True},
            },
        }
    )
    trainer.mrlx_explorer_group = "explorer_policy"
    trainer.mrlx_adapter_group = "adapter_policy"
    trainer.multi_actor_trainable_group_ids = ["explorer_policy", "adapter_policy"]
    trainer.actor_rollout_wgs = {"explorer_policy": FakeActorGroup(), "adapter_policy": FakeActorGroup()}
    trainer._mrlx_adapter_replay = None
    trainer._mrlx_replay_partition = f"mrlx-test-replay-{uuid.uuid4().hex}"
    trainer.total_training_steps = 3
    partition = f"mrlx-test-current-{uuid.uuid4().hex}"

    def put_step(step: int, *, include_adapter: bool = True) -> KVBatchMeta:
        groups = ["explorer_policy"]
        if include_adapter:
            groups.append("adapter_policy")
        keys = [f"step-{step}-{group}" for group in groups]
        fields = TensorDict(
            {
                "worker_group": groups,
                "payload": torch.tensor([step] * len(groups)),
            },
            batch_size=[len(groups)],
        )
        tags = [{"seq_len": 1} for _ in groups]
        tq.kv_batch_put(keys=keys, partition_id=partition, fields=fields, tags=tags)
        return KVBatchMeta(keys=keys, tags=tags, partition_id=partition, fields=["worker_group", "payload"])

    tq.init()
    batches = []
    try:
        trainer.global_steps = 1
        first = put_step(1)
        batches.append(first)
        first_metrics: dict = {}
        trainer._update_actor(first, first_metrics)
        assert trainer.actor_rollout_wgs["explorer_policy"].phases == ["current"]
        assert trainer.actor_rollout_wgs["adapter_policy"].phases == []
        assert trainer._mrlx_adapter_replay is not None
        assert first_metrics["trajweave/mrlx/adapter/replay_pending"] == 1

        trainer.global_steps = 2
        second = put_step(2, include_adapter=False)
        batches.append(second)
        second_metrics: dict = {}
        trainer._update_actor(second, second_metrics)
        assert trainer.actor_rollout_wgs["explorer_policy"].phases == ["current", "current"]
        assert trainer.actor_rollout_wgs["adapter_policy"].phases == ["delayed"]
        assert second_metrics["trajweave/mrlx/adapter/update_lag_steps"] == 1
        assert second_metrics["trajweave/mrlx/adapter/collected_samples"] == 0
        assert second_metrics["trajweave/mrlx/adapter/replay_pending"] == 0
        assert trainer._mrlx_adapter_replay is None

        trainer.global_steps = 3
        third = put_step(3, include_adapter=False)
        batches.append(third)
        third_metrics: dict = {}
        trainer._update_actor(third, third_metrics)
        assert trainer.actor_rollout_wgs["explorer_policy"].phases == ["current", "current", "current"]
        assert trainer.actor_rollout_wgs["adapter_policy"].phases == ["delayed"]
        assert third_metrics["trajweave/mrlx/adapter/drained_samples"] == 0
        assert third_metrics["trajweave/mrlx/adapter/updated_samples_total"] == 1
        assert trainer._mrlx_adapter_replay is None
    finally:
        for item in batches:
            tq.kv_clear(keys=item.keys, partition_id=item.partition_id)
        if trainer._mrlx_adapter_replay is not None:
            tq.kv_clear(
                keys=trainer._mrlx_adapter_replay.keys,
                partition_id=trainer._mrlx_adapter_replay.partition_id,
            )
        tq.close()


def test_mrlx_final_adapter_batch_drains_only_after_an_older_replay_update(monkeypatch):
    from trajweave.backends.verl.trainers import mrlx_async

    trainer, batch, phases = _routing_only_mrlx_trainer(mrlx_async, include_adapter=True, include_replay=True)
    monkeypatch.setattr(mrlx_async.tq, "kv_clear", lambda **kwargs: None)
    metrics: dict = {}

    trainer._update_actor(batch, metrics)

    assert phases == [
        ("explorer_policy", "current"),
        ("adapter_policy", "delayed"),
        ("adapter_policy", "drain"),
    ]
    assert metrics["trajweave/mrlx/adapter/updated_samples_total"] == 2
    assert trainer._mrlx_adapter_replay is None


def test_mrlx_final_adapter_batch_rejects_zero_step_lag():
    from trajweave.backends.verl.trainers import mrlx_async

    trainer, batch, phases = _routing_only_mrlx_trainer(mrlx_async, include_adapter=True)

    with pytest.raises(RuntimeError, match="violate adapter_delay_steps=1"):
        trainer._update_actor(batch, {})

    assert phases == []


def test_mrlx_training_rejects_a_run_without_any_adapter_update():
    from trajweave.backends.verl.trainers import mrlx_async

    trainer, batch, phases = _routing_only_mrlx_trainer(mrlx_async)

    with pytest.raises(RuntimeError, match="did not exercise two-policy co-training"):
        trainer._update_actor(batch, {})

    assert phases == []


@pytest.mark.parametrize(
    ("total_steps", "loader_steps", "epochs", "message"),
    [
        (1, 2, 1, "at least two training steps"),
        (3, 2, 1, "exceeds the steps reachable"),
    ],
)
def test_mrlx_dataloader_rejects_unreachable_adapter_schedule(
    monkeypatch, total_steps, loader_steps, epochs, message
):
    from trajweave.backends.verl.trainers.mrlx_async import TrajWeaveMrlXAsyncTrainer
    from trajweave.backends.verl.trainers.multi_actor_sync import TrajWeaveMultiActorSyncTrainer

    monkeypatch.setattr(TrajWeaveMultiActorSyncTrainer, "_init_dataloader", lambda self: None)
    trainer = object.__new__(TrajWeaveMrlXAsyncTrainer)
    trainer.config = OmegaConf.create({"trainer": {"total_epochs": epochs}})
    trainer.train_dataloader = [object()] * loader_steps
    trainer.total_training_steps = total_steps

    with pytest.raises(ValueError, match=message):
        trainer._init_dataloader()


def test_mrlx_trainer_rejects_actor_warmup_that_skips_replay_staging(monkeypatch):
    from trajweave.backends.verl.trainers.mrlx_async import TrajWeaveMrlXAsyncTrainer
    from trajweave.backends.verl.trainers.multi_actor_sync import TrajWeaveMultiActorSyncTrainer

    monkeypatch.setattr(TrajWeaveMultiActorSyncTrainer, "_validate_multi_actor_specs", lambda self: None)
    trainer = object.__new__(TrajWeaveMrlXAsyncTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {"critic_warmup": 2},
            "trajweave": {
                "recipe": "mrlx_research_qa",
                "mrlx": {"adapter_delay_steps": 1, "drain_adapter_replay": True},
            },
        }
    )
    trainer.mrlx_explorer_group = "explorer_policy"
    trainer.mrlx_adapter_group = "adapter_policy"
    trainer.multi_actor_trainable_group_ids = ["explorer_policy", "adapter_policy"]

    with pytest.raises(ValueError, match="critic_warmup=0"):
        trainer._validate_multi_actor_specs()


def test_mrlx_train_end_rejects_unconsumed_adapter_replay(monkeypatch):
    from trajweave.backends.verl.trainers import mrlx_async

    trainer = object.__new__(mrlx_async.TrajWeaveMrlXAsyncTrainer)
    trainer._mrlx_adapter_replay = SimpleNamespace(keys=["adapter-row"], partition_id="adapter-replay")
    cleared: list[tuple[list[str], str]] = []
    monkeypatch.setattr(
        mrlx_async.tq,
        "kv_clear",
        lambda *, keys, partition_id: cleared.append((keys, partition_id)),
    )

    with pytest.raises(RuntimeError, match="unconsumed Adapter replay"):
        trainer.on_train_end()

    assert cleared == [(["adapter-row"], "adapter-replay")]
    assert trainer._mrlx_adapter_replay is None


def _routing_only_mrlx_trainer(mrlx_async, *, include_adapter=False, include_replay=False):
    trainer = object.__new__(mrlx_async.TrajWeaveMrlXAsyncTrainer)
    trainer.config = OmegaConf.create(
        {
            "trajweave": {
                "recipe": "mrlx_research_qa",
                "multi_actor": {"metric_namespace": "mrlx"},
                "mrlx": {"drain_adapter_replay": True},
            }
        }
    )
    trainer.mrlx_explorer_group = "explorer_policy"
    trainer.mrlx_adapter_group = "adapter_policy"
    trainer.actor_rollout_wgs = {"explorer_policy": object(), "adapter_policy": object()}
    trainer.global_steps = 2
    trainer.total_training_steps = 2
    trainer._mrlx_adapter_collected_samples = 0
    trainer._mrlx_adapter_updated_samples = 0
    trainer._mrlx_adapter_replay = (
        SimpleNamespace(
            keys=["old-adapter"],
            partition_id="old-adapter-replay",
            extra_info={"mrlx_collected_step": 1},
        )
        if include_replay
        else None
    )

    explorer = SimpleNamespace(keys=["current-explorer"], extra_info={})
    routes = [SimpleNamespace(group_id="explorer_policy", batch=explorer)]
    if include_adapter:
        adapter = SimpleNamespace(keys=["current-adapter"], extra_info={})
        routes.append(SimpleNamespace(group_id="adapter_policy", batch=adapter))
    batch = SimpleNamespace(extra_info={})
    phases: list[tuple[str, str]] = []
    trainer._prepare_actor_update = lambda value: None
    trainer._route_batch = lambda value: routes
    trainer._update_group = lambda group_id, route, *, phase: phases.append((group_id, phase)) or {}
    return trainer, batch, phases


def _search_task() -> SearchTask:
    return SearchTask(
        task_id="mrlx-test",
        question="Which city is the capital of France?",
        answer="Paris",
        search_query="capital France",
        documents=(SearchDocument(title="France", text="The capital of France is Paris."),),
    )


def _search_prompt() -> dict:
    return {
        "uid": "mrlx-search-1",
        "raw_prompt": [{"role": "user", "content": "Which city is the capital of France?"}],
        "reward_model": {"ground_truth": "Paris"},
        "extra_info": {
            "search_query": "capital France",
            "documents": [{"title": "France", "text": "The capital of France is Paris."}],
        },
    }
