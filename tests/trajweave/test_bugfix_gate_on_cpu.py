from __future__ import annotations

import asyncio
import builtins
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from trajweave.backends.local import TinyTorchPolicyBackend
from trajweave.backends.verl.agent_loop import _schedule_background_task
from trajweave.backends.verl.extensions.common import nested_compat
from trajweave.backends.verl.local_generation import HFLocalGenerationMixin
from trajweave.backends.verl.routing import RoutedBatch
from trajweave.backends.verl.schema import required_ground_truth
from trajweave.backends.verl.trainers.multi_actor_sync import TrajWeaveMultiActorSyncTrainer
from trajweave.pipeline.config import load_yaml_config
from trajweave.pipeline.launch import _integer_override, _validate_training_progress
from trajweave.recipes.registry import RECIPES, resolve_recipe, validate_recipe_mode
from trajweave.runner import _output_failure_reason, run_from_config
from trajweave.storage.run_store import git_worktree_fingerprint


@pytest.mark.parametrize(
    "config, missing_key",
    [
        ({"mode": "smoke"}, "recipe"),
        ({"recipe": "doctor_mas_math"}, "mode"),
        ({"recipe": " ", "mode": "smoke"}, "recipe"),
        ({"recipe": "doctor_mas_math", "mode": " "}, "mode"),
    ],
)
def test_required_recipe_and_mode_never_fall_back(tmp_path: Path, config: dict, missing_key: str):
    config["run"] = {"root_dir": str(tmp_path)}
    with pytest.raises(ValueError, match=missing_key):
        run_from_config(config)


@pytest.mark.parametrize("recipe", RECIPES, ids=lambda item: item.name)
def test_every_recipe_rejects_unknown_mode(recipe):
    with pytest.raises(ValueError, match="Unsupported mode"):
        validate_recipe_mode(recipe, "verl_trian")


def test_verl_train_never_accepts_missing_disabled_or_dry_run_launch():
    assert _output_failure_reason({"mode": "verl_train"})
    assert _output_failure_reason({"mode": "verl_train", "verl_launch": {"status": "disabled"}})
    assert _output_failure_reason({"mode": "verl_train", "verl_launch": {"status": "dry_run"}})


def test_verl_plan_generates_command_and_finalizes_as_planned(tmp_path: Path):
    result = run_from_config(
        {
            "recipe": "agentflow.flow_grpo.planner_tool",
            "mode": "verl_plan",
            "run": {"root_dir": str(tmp_path)},
            "logging": {"console": False},
            "prepare": {"tiny_verl_assets": {"enabled": False}},
            "agentflow": {"agent_loop_backend": "synthetic_tq", "max_steps": 2},
            "verl": {
                "enabled": True,
                "execute": False,
                "overrides": ["trainer.total_training_steps=1"],
            },
        }
    )

    status = json.loads((Path(result["run_dir"]) / "status.json").read_text(encoding="utf-8"))
    assert result["verl_launch"]["status"] == "dry_run"
    assert status["status"] == "planned"


@pytest.mark.parametrize("execute", [False, True])
def test_smoke_mode_cannot_enable_verl(tmp_path: Path, execute: bool):
    with pytest.raises(ValueError, match="cannot enable VERL"):
        run_from_config(
            {
                "recipe": "doctor_mas_math",
                "mode": "smoke",
                "run": {"root_dir": str(tmp_path)},
                "logging": {"console": False},
                "backend": {"type": "rule"},
                "rollout": {"rollouts_per_task": 1},
                "verl": {"enabled": True, "execute": execute},
            }
        )


@pytest.mark.parametrize("execute", [False, None])
def test_verl_train_requires_real_execution(tmp_path: Path, execute: bool | None):
    verl = {"enabled": True, "overrides": ["trainer.total_training_steps=1"]}
    if execute is not None:
        verl["execute"] = execute
    with pytest.raises(ValueError, match="verl.execute=true"):
        run_from_config(
            {
                "recipe": "agentflow.flow_grpo.planner_tool",
                "mode": "verl_train",
                "run": {"root_dir": str(tmp_path)},
                "logging": {"console": False},
                "prepare": {"tiny_verl_assets": {"enabled": False}},
                "agentflow": {"agent_loop_backend": "synthetic_tq", "max_steps": 2},
                "verl": verl,
            }
        )


@pytest.mark.parametrize(
    "overrides",
    [(), ("trainer.total_training_steps=0",), ("trainer.total_training_steps=invalid",)],
)
def test_training_progress_requires_positive_explicit_step_count(overrides: tuple[str, ...]):
    result = {"status": "ok", "returncode": 0}

    _validate_training_progress(result, overrides, require_explicit_steps=True)

    assert result["status"] == "failed"
    assert "positive integer" in result["validation_error"]


def test_agent_loop_background_dispatch_returns_before_rollout_finishes():
    async def scenario() -> None:
        release = asyncio.Event()
        started = asyncio.Event()
        tasks: set[asyncio.Task] = set()

        async def rollout() -> None:
            started.set()
            await release.wait()

        task = _schedule_background_task(tasks, rollout())
        await started.wait()
        assert task in tasks
        assert not task.done()
        release.set()
        await task
        await asyncio.sleep(0)
        assert task not in tasks

    asyncio.run(scenario())


def test_chat_template_failure_is_strict_unless_fallback_is_explicit():
    class Tokenizer:
        eos_token_id = 1
        pad_token_id = 0

        def apply_chat_template(self, *_args, **_kwargs):
            raise ValueError("missing template")

        def encode(self, _text, **_kwargs):
            return [7, 8]

    class Worker(HFLocalGenerationMixin):
        pass

    worker = Worker()
    worker.tokenizer = Tokenizer()
    worker.rollout_config = SimpleNamespace(prompt_length=8, response_length=8)
    worker.config = {"trajweave": {"allow_plain_text_prompt_fallback": False}}
    with pytest.raises(RuntimeError, match="no usable chat template"):
        worker._encode_chat_prompt(worker.tokenizer, "hello")

    worker.config["trajweave"]["allow_plain_text_prompt_fallback"] = True
    assert worker._encode_chat_prompt(worker.tokenizer, "hello") == [7, 8]


def test_tiny_torch_does_not_silently_fall_back_when_torch_import_fails(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("torch unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(RuntimeError, match="requires a working PyTorch"):
        TinyTorchPolicyBackend()


def test_tiny_torch_does_not_silently_move_cuda_requests_to_cpu(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        TinyTorchPolicyBackend(device="cuda:0")


def test_nested_compat_installation_rolls_back_every_worker_patch(monkeypatch):
    before = nested_compat._capture_compat_patch_state(include_trainer=False)

    def fail_after_earlier_patches() -> None:
        raise RuntimeError("injected late patch failure")

    monkeypatch.setattr(nested_compat, "_patch_fsdp_response_outputs", fail_after_earlier_patches)
    with pytest.raises(RuntimeError, match="injected late patch failure"):
        nested_compat.install_worker_nested_tensor_compat()

    after = nested_compat._capture_compat_patch_state(include_trainer=False)
    assert after["tensor_sum"] is before["tensor_sum"]
    assert after["tensor_sum_marker_exists"] is before["tensor_sum_marker_exists"]
    assert after["to_padded_tensor"] is before["to_padded_tensor"]
    assert after["losses_ppo_loss"] is before["losses_ppo_loss"]
    assert after["engine_workers_ppo_loss"] is before["engine_workers_ppo_loss"]
    assert after["fsdp_prepare_model_outputs"] is before["fsdp_prepare_model_outputs"]


@pytest.mark.parametrize("prompt", [{}, {"reward_model": {}}, {"reward_model": {"ground_truth": " "}}])
def test_ground_truth_is_required_for_training_emitters(prompt: dict):
    with pytest.raises(ValueError, match="ground_truth"):
        required_ground_truth(prompt)


def test_missing_trainable_worker_group_aborts_before_any_actor_update():
    class Batch:
        def __init__(self, keys):
            self.keys = list(keys)
            self.extra_info = {}

        def __len__(self):
            return len(self.keys)

    updates: list[str] = []
    trainer = object.__new__(TrajWeaveMultiActorSyncTrainer)
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            actor=SimpleNamespace(
                ppo_mini_batch_size=1,
                calculate_entropy=False,
                entropy_coeff=0.0,
                ppo_epochs=1,
                data_loader_seed=1,
                shuffle=False,
            ),
            rollout=SimpleNamespace(n=1, temperature=1.0),
        )
    )
    trainer.multi_actor_trainable_group_ids = ["group_a", "group_b"]
    trainer.actor_rollout_wgs = {"group_a": object(), "group_b": object()}
    trainer._metric_namespace = lambda: "maporl"
    trainer._route_batch = lambda _batch: [RoutedBatch("group_a", Batch(["a0"]))]
    trainer._actor_wg = lambda group_id: updates.append(group_id)

    with pytest.raises(RuntimeError, match="missing samples"):
        trainer._update_actor(Batch(["a0"]), {})
    assert updates == []


def test_all_checked_in_configs_follow_strict_run_contract():
    config_root = Path(__file__).resolve().parents[2] / "configs"
    paths = sorted(config_root.rglob("*.yaml"))
    assert paths
    for path in paths:
        config = load_yaml_config(path)
        recipe = resolve_recipe(str(config.get("recipe", "")))
        mode = str(config.get("mode", ""))
        validate_recipe_mode(recipe, mode)
        verl = config.get("verl", {}) or {}
        if mode == "verl_train":
            assert verl.get("enabled") is True, path
            assert verl.get("execute") is True, path
            steps = _integer_override(
                tuple(str(item) for item in verl.get("overrides", [])),
                "trainer.total_training_steps",
            )
            assert steps is not None and steps > 0, path
        elif mode == "verl_plan":
            assert verl.get("enabled") is True, path
            assert verl.get("execute") is False, path
        elif mode == "smoke":
            assert verl.get("enabled", False) is False, path


def test_worktree_fingerprint_includes_untracked_file_contents(tmp_path: Path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("tracked\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=TrajWeave Test",
            "-c",
            "user.email=trajweave@example.invalid",
            "commit",
            "-qm",
            "test fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    baseline = git_worktree_fingerprint(tmp_path)
    untracked = tmp_path / "rubric.py"
    untracked.write_text("first\n", encoding="utf-8")
    first = git_worktree_fingerprint(tmp_path)
    untracked.write_text("second\n", encoding="utf-8")
    second = git_worktree_fingerprint(tmp_path)

    assert baseline and first and second
    assert len({baseline, first, second}) == 3


def _load_bugfix_rubric_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "run_bugfix_rubric.py"
    spec = importlib.util.spec_from_file_location("trajweave_bugfix_rubric", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_r13_rejects_single_actor_only_evidence(monkeypatch, tmp_path: Path):
    rubric = _load_bugfix_rubric_module()
    monkeypatch.setattr(
        rubric,
        "_audit_one_run",
        lambda *_args, **_kwargs: ([], {"native_multi_actor_training": False}),
    )

    result = rubric._audit_e2e_runs([tmp_path])

    assert result.status == "FAIL"
    assert "No native multi-actor" in result.evidence["failures"][-1]


def test_r13_rejects_duplicate_worker_group_steps(tmp_path: Path):
    rubric = _load_bugfix_rubric_module()
    failures: list[str] = []
    evidence: dict = {}
    rows = []
    for group, steps in (("group_a", (1, 1)), ("group_b", (1, 2))):
        for step in steps:
            rows.extend(
                [
                    {"name": f"recipe/actor_groups/{group}/samples", "value": 2, "step": step},
                    {"name": f"recipe/actor_groups/{group}/updated", "value": 1, "step": step},
                ]
            )
    rows.extend([{"name": "recipe/actor_groups/missing_trainable", "value": 0, "step": step} for step in (1, 2)])
    trajectories = [
        {
            "recipe": "maporl",
            "uid": group,
            "turn_id": 0,
            "agent_id": group,
            "worker_group": group,
            "reward_score": 1.0,
        }
        for group in ("group_a", "group_b")
    ]

    rubric._audit_multi_actor_training(
        run_dir=tmp_path,
        metadata={"trainable_worker_groups": ["group_a", "group_b"]},
        metric_rows=rows,
        latest_metrics={
            "recipe/actor_groups/updated": 2,
            "recipe/actor_groups/total": 2,
        },
        expected_steps=2,
        trajectory_rows=trajectories,
        failures=failures,
        evidence=evidence,
    )

    assert any("group_a" in failure and "sample steps" in failure for failure in failures)
    assert any("group_a" in failure and "update steps" in failure for failure in failures)


def test_hf_role_prompt_honors_chat_template_options():
    calls = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.append((messages, kwargs))
            return [3, 4]

    class Worker(HFLocalGenerationMixin):
        pass

    worker = Worker()
    worker.config = {"data": {"apply_chat_template_kwargs": {"enable_thinking": False}}}
    worker.rollout_config = SimpleNamespace(prompt_length=16)
    assert worker._encode_chat_prompt(Tokenizer(), "规划下一步") == [3, 4]
    assert calls == [([{"role": "user", "content": "规划下一步"}], {
        "add_generation_prompt": True, "tokenize": True, "enable_thinking": False,
    })]
