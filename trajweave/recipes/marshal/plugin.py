from __future__ import annotations

from typing import Any

from trajweave.pipeline.context import RunContext
from trajweave.pipeline.launch import maybe_run_verl_launch
from trajweave.recipes.marshal.config import build_marshal_launch_overrides, resolve_marshal_settings
from trajweave.recipes.marshal.self_play import run_marshal_smoke


class MARSHALRecipePlugin:
    name = "marshal"

    def supports(self, context: RunContext) -> bool:
        return context.recipe_definition.family == "marshal"

    def run(self, context: RunContext) -> dict[str, Any]:
        if context.mode == "smoke":
            return self._run_smoke(context)
        if context.mode in {"verl_train", "verl_plan"}:
            return self._run_verl(context)
        raise ValueError(f"Unsupported MARSHAL mode: {context.mode!r}.")

    def _run_smoke(self, context: RunContext) -> dict[str, Any]:
        settings = resolve_marshal_settings(context.config)
        summary, result = run_marshal_smoke(
            rollouts_per_task=int((context.config.get("rollout", {}) or {}).get("rollouts_per_task", 1)),
            player_ids=settings["player_ids"],
            shared_model_id=settings["shared_model_id"],
            max_actions=settings["max_actions"],
            format_reward=settings["format_reward"],
            allow_bare_actions=settings["allow_bare_actions"],
            gamma=settings["gamma"],
            reward_normalization=settings["reward_normalization"],
            advantage_normalization=settings["advantage_normalization"],
            whiten_rewards=settings["whiten_rewards"],
            whiten_advantages=settings["whiten_advantages"],
        )
        context.tracker.log_rollout_result(result, source="marshal_tictactoe_selfplay")
        return {
            **context.base_output(),
            "trajectories": summary.trajectories,
            "samples": summary.samples,
            "success_rate": summary.success_rate,
            "player_0_wins": summary.player_0_wins,
            "player_1_wins": summary.player_1_wins,
            "draws": summary.draws,
            "marshal": marshal_summary(context.config),
        }

    def _run_verl(self, context: RunContext) -> dict[str, Any]:
        output = context.base_output()
        output["marshal"] = marshal_summary(context.config)
        maybe_run_verl_launch(
            context.config,
            output,
            overrides=build_marshal_launch_overrides(context.config, config_path=context.config_path),
            default_enabled=True,
            default_module="trajweave.backends.verl.main_ppo",
            tracker=context.tracker,
            mode=context.mode,
        )
        return output


def marshal_summary(config: dict[str, Any]) -> dict[str, Any]:
    settings = resolve_marshal_settings(config)
    return {
        "task": "strategic_self_play",
        "game": settings["game"],
        "runtime_recipe": "marshal_tictactoe_selfplay",
        "coordination_protocol": "marshal_alternating_self_play",
        "player_ids": list(settings["player_ids"]),
        "shared_model_id": settings["shared_model_id"],
        "shared_policy": True,
        "per_player_subtrajectories": True,
        "allow_bare_actions": settings["allow_bare_actions"],
        "credit_allocator": "marshal_turn_level_reinforce",
        "credit_levels": ["turn_return", "agent_specific_normalization"],
        "training_backend": "verl_v1_shared_actor",
    }
