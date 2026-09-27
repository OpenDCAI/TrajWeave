from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from trajweave.backends.policy import PolicyRequest, PolicyResponse, StableByteTokenizer
from trajweave.backends.verl.emitters.marshal import MARSHALSelfPlayEmitterMixin
from trajweave.backends.verl.extensions.common.hooks import extension_hooks_for_config
from trajweave.backends.verl.extensions.marshal import MARSHALHooks
from trajweave.backends.verl.extensions.registry import _extension_names
from trajweave.backends.verl.tiny_assets import prepare_verl_dataset
from trajweave.backends.verl.workflow_runtime import build_hf_workflow_outputs
from trajweave.credit.marshal import compute_marshal_scalar_advantages
from trajweave.envs.strategic import TicTacToeEnvironment, TicTacToeTask
from trajweave.orchestration.marshal import MARSHALSelfPlayOrchestra
from trajweave.recipes.marshal import default_marshal_team, run_marshal_smoke
from trajweave.recipes.marshal.config import build_marshal_launch_overrides, resolve_marshal_settings
from trajweave.recipes.registry import resolve_recipe
from trajweave.runner import run_from_config
from verl import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator


def test_marshal_registry_and_smoke_run_balanced_shared_policy_selfplay():
    recipe = resolve_recipe("marshal")
    summary, result = run_marshal_smoke(rollouts_per_task=1)

    assert recipe.name == "marshal.tictactoe_selfplay"
    assert recipe.runtime_recipe == "marshal_tictactoe_selfplay"
    assert summary.trajectories == 2
    assert summary.samples == 11
    assert summary.player_0_wins == 1
    assert summary.player_1_wins == 1
    assert summary.draws == 0
    assert {sample.policy_group for sample in result.samples} == {"shared_policy"}
    assert {sample.agent_name for sample in result.samples} == {"player_0", "player_1"}
    assert any(abs(float(sample.advantage)) > 1e-6 for sample in result.samples)

    for trajectory in result.trajectories:
        assert trajectory.metadata["marshal_shared_policy"] is True
        assert set(trajectory.metadata["marshal_per_player_subtrajectories"]) == {"player_0", "player_1"}
        assert [turn.metadata["marshal_player_id"] for turn in trajectory.turns] == [
            index % 2 for index in range(len(trajectory.turns))
        ]


def test_marshal_terminal_payoff_is_attached_to_each_players_latest_action():
    _summary, result = run_marshal_smoke(
        rollouts_per_task=1,
        format_reward=0.05,
        whiten_rewards=False,
        whiten_advantages=False,
    )
    trajectory = next(item for item in result.trajectories if item.metadata["marshal_winner"] == 0)
    by_player = {
        player: [turn for turn in trajectory.turns if turn.metadata["marshal_player_id"] == player]
        for player in (0, 1)
    }

    assert by_player[0][-1].step_reward == pytest.approx(1.05)
    assert by_player[1][-1].step_reward == pytest.approx(-0.95)
    assert all(turn.step_reward == pytest.approx(0.05) for turns in by_player.values() for turn in turns[:-1])
    assert by_player[0][-1].metadata["marshal_terminal_payoff"] == 1.0
    assert by_player[1][-1].metadata["marshal_terminal_payoff"] == -1.0


class _InvalidPolicy:
    tokenizer = StableByteTokenizer()

    def generate(self, request: PolicyRequest) -> PolicyResponse:
        del request
        text = "move zero"
        return PolicyResponse(text=text, token_ids=self.tokenizer.encode(text))


def test_marshal_invalid_action_ends_game_and_assigns_loss():
    team = default_marshal_team()
    task = TicTacToeTask(task_id="invalid")
    trajectory = MARSHALSelfPlayOrchestra().run(
        episode_id="invalid-0",
        rollout_group="invalid",
        task=task,
        team=team,
        observation="",
        policy_backend=_InvalidPolicy(),
        environment=TicTacToeEnvironment(),
    )

    assert len(trajectory.turns) == 1
    assert trajectory.turns[0].reward == -1.0
    assert trajectory.turns[0].metadata["marshal_format_valid"] is False
    assert trajectory.metadata["marshal_invalid_action"] is True
    assert trajectory.metadata["marshal_winner"] == 1


def test_marshal_prompt_lists_complete_legal_responses_without_placeholder():
    prompt = MARSHALSelfPlayOrchestra._prompt(
        board=". . .\n. X .\n. . .",
        player_id=1,
        legal_actions=(0, 8),
        history=[],
    )

    assert "<answer>0</answer>, <answer>8</answer>" in prompt
    assert "<answer>N</answer>" not in prompt
    assert "Copy exactly one complete legal response" in prompt


def test_marshal_optional_bare_action_is_legal_without_format_reward():
    game = TicTacToeEnvironment()
    state = game.initial_state(TicTacToeTask(task_id="bare-action"))
    parsed = game.parse_response("4", allow_bare_action=True)
    transition = game.step(state, parsed.action, format_valid=parsed.format_valid)

    assert parsed.action == 4
    assert parsed.format_valid is False
    assert transition.format_valid is False
    assert transition.state.board[4] == 0
    assert transition.state.done is False


class _BareMovePolicy:
    tokenizer = StableByteTokenizer()
    moves = iter(("4", "0"))

    def generate(self, request: PolicyRequest) -> PolicyResponse:
        del request
        text = next(self.moves)
        return PolicyResponse(text=text, token_ids=self.tokenizer.encode(text))


def test_marshal_bare_action_history_records_valid_move_not_valid_format():
    trajectory = MARSHALSelfPlayOrchestra(max_actions=2, allow_bare_actions=True).run(
        episode_id="bare-history",
        rollout_group="bare-history",
        task=TicTacToeTask(task_id="bare-history"),
        team=default_marshal_team(max_actions=2),
        observation="",
        policy_backend=_BareMovePolicy(),
        environment=TicTacToeEnvironment(),
    )

    assert trajectory.turns[0].metadata["marshal_action_valid"] is True
    assert trajectory.turns[0].metadata["marshal_format_valid"] is False
    assert "action=4" in trajectory.metadata["marshal_action_history"][0]


def test_marshal_turn_returns_are_discounted_within_each_player_subtrajectory():
    result = compute_marshal_scalar_advantages(
        turn_rewards=torch.tensor([0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0, 1.0]),
        episode_ids=["a", "a", "a", "a", "b", "b", "b", "b"],
        player_ids=[0, 1, 0, 1, 0, 1, 0, 1],
        player_turn_ids=[0, 0, 1, 1, 0, 0, 1, 1],
        gamma=0.5,
        reward_normalization="identity",
        advantage_normalization="mean",
        whiten_rewards=False,
        whiten_advantages=False,
    )

    torch.testing.assert_close(
        result.turn_returns,
        torch.tensor([0.5, -0.5, 1.0, -1.0, -0.5, 0.5, -1.0, 1.0]),
    )
    torch.testing.assert_close(result.advantages, result.turn_returns)


def test_marshal_agent_normalization_uses_unique_values_not_frequency_weighting():
    result = compute_marshal_scalar_advantages(
        turn_rewards=torch.tensor([1.0, 1.0, 3.0, 10.0, 14.0]),
        episode_ids=["a", "b", "c", "d", "e"],
        player_ids=[0, 0, 0, 1, 1],
        player_turn_ids=[0, 0, 0, 0, 0],
        reward_normalization="identity",
        advantage_normalization="mean",
        whiten_rewards=False,
        whiten_advantages=False,
    )

    torch.testing.assert_close(result.advantages, torch.tensor([-1.0, -1.0, 1.0, -2.0, 2.0]))


def test_marshal_verl_hook_ignores_padding_and_emits_player_metrics():
    response_mask = torch.tensor([[1, 1], [1, 1], [1, 1], [1, 1], [0, 0]], dtype=torch.int64)
    data = DataProto.from_dict(
        tensors={
            "response_mask": response_mask,
            "token_level_rewards": torch.zeros_like(response_mask, dtype=torch.float32),
        },
        non_tensors={
            "uid": np.array(["game", "game", "game", "game", "padding"], dtype=object),
            "agent_id": np.array(["player_0", "player_1", "player_0", "player_1", "__padding__"], dtype=object),
            "traj_uid": np.array(["a", "a", "a", "a", "padding"], dtype=object),
            "turn_id": np.array([0, 1, 2, 3, 0], dtype=object),
            "marshal_episode_id": np.array(["a", "a", "a", "a", "padding"], dtype=object),
            "marshal_player_id": np.array([0, 1, 0, 1, -1], dtype=object),
            "marshal_player_turn": np.array([0, 0, 1, 1, 0], dtype=object),
            "marshal_turn_reward": np.array([0.0, 0.0, 1.0, -1.0, 0.0], dtype=object),
            "active_mask": np.array([1.0, 1.0, 1.0, 1.0, 0.0], dtype=object),
        },
    )
    hooks = MARSHALHooks()
    result = hooks.compute_advantage(
        data,
        adv_estimator=AdvantageEstimator.REINFORCE_PLUS_PLUS,
        gamma=0.5,
        config={
            "marshal": {
                "reward_normalization": "identity",
                "advantage_normalization": "mean",
                "whiten_rewards": False,
                "whiten_advantages": False,
            }
        },
    )

    torch.testing.assert_close(result.batch["advantages"][:4, 0], torch.tensor([-0.25, 0.25, 0.25, -0.25]))
    torch.testing.assert_close(result.batch["advantages"][4], torch.zeros(2))
    metrics = hooks.compute_extra_metrics(result, {}, "advantage")
    assert metrics["trajweave/marshal/active_players"] == 2.0
    assert metrics["trajweave/marshal/nonzero_advantage_ratio"] == 1.0


def test_marshal_extension_hook_and_schema_are_registered():
    config = {
        "trajweave": {
            "recipe": "marshal_tictactoe_selfplay",
            "credit_allocator": "marshal_turn_level_reinforce",
        }
    }

    assert _extension_names(config) == ("trajweave_marshal_turn_advantage",)
    assert isinstance(extension_hooks_for_config(config), MARSHALHooks)
    assert {
        "marshal_episode_id",
        "marshal_player_id",
        "marshal_player_turn",
        "marshal_turn_reward",
    } <= set(MARSHALHooks().batch_schema_fields("advantage"))


def test_marshal_config_wires_shared_actor_and_turn_advantage_contract():
    config = {
        "marshal": {"player_ids": ["first", "second"], "shared_model_id": "one_policy"},
        "verl": {"overrides": ["trainer.use_v1=false", "critic.enable=true"]},
    }
    overrides = build_marshal_launch_overrides(config, config_path="configs/marshal/example.yaml")

    assert "trainer.use_v1=true" in overrides
    assert "trainer.use_v1=false" not in overrides
    assert "critic.enable=false" in overrides
    assert "critic.enable=true" not in overrides
    assert "algorithm.adv_estimator=reinforce_plus_plus" in overrides
    assert "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1" in overrides
    assert "actor_rollout_ref.model.lora_rank=8" in overrides
    assert "actor_rollout_ref.model.lora_alpha=16" in overrides
    assert 'actor_rollout_ref.model.target_modules="all-linear"' in overrides
    assert "+agent.orchestra.marshal.allow_bare_actions=false" in overrides
    assert '+agent.agent_ids=["first","second"]' in overrides
    assert '+agent.model_ids=["one_policy","one_policy"]' in overrides
    assert "+agent.model_sharing=true" in overrides
    assert any("MARSHALHooks" in item for item in overrides)


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"player_ids": ["only_one"]}, "two distinct"),
        ({"max_actions": 10}, "must not exceed 9"),
        ({"gamma": 1.1}, "must be in"),
        ({"lora_rank": 0}, "positive integer"),
        ({"game": "connect_four"}, "supports game=tictactoe"),
        ({"agent_loop_backend": "verl_tq"}, "synthetic_tq or hf_local_tq"),
    ],
)
def test_marshal_config_rejects_unsupported_contract(settings, message):
    with pytest.raises(ValueError, match=message):
        resolve_marshal_settings({"marshal": settings})


def test_marshal_strategic_dataset_and_verl_plan_are_executable(tmp_path):
    dataset = prepare_verl_dataset(
        tmp_path / "assets",
        train_size=2,
        val_size=1,
        task_family="strategic_game",
        recipe_name="marshal_tictactoe_selfplay",
    )
    with open(dataset["train_file"], encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    assert {row["extra_info"]["strategy"] for row in rows} == {"player0_win", "player1_win"}

    plan = run_from_config(
        {
            "recipe": "marshal.tictactoe_selfplay",
            "mode": "verl_plan",
            "output_dir": str(tmp_path / "runs"),
            "prepare": {"tiny_verl_assets": {"enabled": False}},
            "marshal": {"agent_loop_backend": "synthetic_tq"},
            "verl": {"enabled": True, "execute": False, "overrides": []},
        },
        config_path="marshal-plan.yaml",
    )
    assert plan["verl_launch"]["status"] == "dry_run"
    assert "+trajweave.recipe=marshal_tictactoe_selfplay" in plan["verl_launch"]["command"]


class _FakeMARSHALWorker(MARSHALSelfPlayEmitterMixin):
    def __init__(self, responses: list[str] | None = None):
        self.tokenizer = StableByteTokenizer()
        self.responses = list(responses or [])
        self.model_config = SimpleNamespace(local_path="/models/shared")
        self.config = {
            "agent": {
                "agent_ids": ["player_0", "player_1"],
                "orchestra": {
                    "marshal": {
                        "player_ids": ["player_0", "player_1"],
                        "shared_model_id": "shared_policy",
                        "max_actions": 9,
                        "format_reward": 0.05,
                    }
                },
            }
        }

    def _encode_prompt(self, prompt):
        return self.tokenizer.encode(str(prompt))

    def _encode_text(self, text):
        return self.tokenizer.encode(text)

    def _encode_prompt_text(self, text):
        return self.tokenizer.encode(text)

    def _decode_response_ids(self, token_ids):
        return self.tokenizer.decode(token_ids)

    def _generate_local_response_ids(self, prompt_ids, **kwargs):
        del prompt_ids, kwargs
        return self.tokenizer.encode(self.responses.pop(0))

    def _local_policy_version(self):
        return 3

    @staticmethod
    def _worker_group_model_path(group_id):
        return f"/models/{group_id}"


def test_marshal_synthetic_and_hf_emitters_preserve_player_turn_rewards():
    synthetic = _FakeMARSHALWorker()._build_marshal_tictactoe_outputs(
        {"uid": "game", "extra_info": {"strategy": "player0_win"}},
        session_id=0,
    )
    hf = build_hf_workflow_outputs(
        _FakeMARSHALWorker(
            [
                "<answer>0</answer>",
                "<answer>8</answer>",
                "<answer>1</answer>",
                "<answer>7</answer>",
                "<answer>2</answer>",
            ]
        ),
        recipe="marshal_tictactoe_selfplay",
        prompt={
            "uid": "game",
            "raw_prompt": [{"role": "user", "content": "Play Tic-Tac-Toe."}],
            "reward_model": {"ground_truth": "self_play"},
            "extra_info": {"strategy": "player0_win"},
        },
        session_id=0,
    )

    for outputs in (synthetic, hf):
        assert [item.extra_fields["marshal_player_id"] for item in outputs] == [0, 1, 0, 1, 0]
        assert [item.extra_fields["marshal_player_turn"] for item in outputs] == [0, 0, 1, 1, 2]
        assert [item.reward_score for item in outputs] == pytest.approx([0.05, 0.05, 0.05, -0.95, 1.05])
        assert {item.extra_fields["policy_group"] for item in outputs} == {"shared_policy"}


def test_marshal_checked_in_configs_parse():
    for path in (
        Path("configs/marshal/tictactoe_selfplay_smoke.yaml"),
        Path("configs/marshal/tictactoe_selfplay_verl_tiny.yaml"),
        Path("configs/marshal/tictactoe_selfplay_qwen05b_1gpu.yaml"),
    ):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert resolve_marshal_settings(config)["game"] == "tictactoe"

    real_config = yaml.safe_load(
        Path("configs/marshal/tictactoe_selfplay_qwen05b_1gpu.yaml").read_text(encoding="utf-8")
    )
    assert resolve_marshal_settings(real_config)["allow_bare_actions"] is True
