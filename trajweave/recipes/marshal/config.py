from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Any

TRAJWEAVE_AGENT_LOOP_MANAGER_FQN = "trajweave.backends.verl.agent_loop.TrajWeaveAgentLoopManager"
MARSHAL_HOOKS_FQN = "trajweave.backends.verl.extensions.marshal.MARSHALHooks"


def resolve_marshal_settings(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("marshal", {}) or {}
    if not isinstance(raw, dict):
        raise ValueError("MARSHAL settings must be a mapping.")
    backend = str(raw.get("agent_loop_backend", "hf_local_tq")).strip()
    if backend not in {"synthetic_tq", "hf_local_tq"}:
        raise ValueError("MARSHAL agent_loop_backend must be synthetic_tq or hf_local_tq.")
    game = str(raw.get("game", "tictactoe")).strip().lower()
    if game != "tictactoe":
        raise ValueError("The current MARSHAL slice supports game=tictactoe.")
    player_ids = tuple(str(value).strip() for value in raw.get("player_ids", ["player_0", "player_1"]))
    if len(player_ids) != 2 or len(set(player_ids)) != 2 or any(not value for value in player_ids):
        raise ValueError("MARSHAL player_ids must contain two distinct non-empty IDs.")
    reward_normalization = str(raw.get("reward_normalization", "mean"))
    if reward_normalization not in {"identity", "mean", "mean_std"}:
        raise ValueError("MARSHAL reward_normalization must be identity, mean, or mean_std.")
    advantage_normalization = str(raw.get("advantage_normalization", "mean"))
    if advantage_normalization not in {"mean", "mean_std"}:
        raise ValueError("MARSHAL advantage_normalization must be mean or mean_std.")
    gamma = _finite_float(raw.get("gamma", 1.0), "gamma")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("MARSHAL gamma must be in [0, 1].")
    max_actions = _positive_int(raw.get("max_actions", 9), "max_actions")
    if max_actions > 9:
        raise ValueError("MARSHAL max_actions must not exceed 9 for Tic-Tac-Toe.")
    format_reward = _finite_float(raw.get("format_reward", 0.05), "format_reward")
    if format_reward < 0:
        raise ValueError("MARSHAL format_reward must be non-negative.")
    shared_model_id = str(raw.get("shared_model_id", "shared_policy")).strip()
    if not shared_model_id:
        raise ValueError("MARSHAL shared_model_id must be non-empty.")
    lora_rank = _positive_int(raw.get("lora_rank", 8), "lora_rank")
    lora_alpha = _positive_int(raw.get("lora_alpha", 16), "lora_alpha")
    lora_target_modules = str(raw.get("lora_target_modules", "all-linear")).strip()
    if not lora_target_modules:
        raise ValueError("MARSHAL lora_target_modules must be non-empty.")
    return {
        "agent_loop_backend": backend,
        "game": game,
        "player_ids": player_ids,
        "shared_model_id": shared_model_id,
        "max_actions": max_actions,
        "format_reward": format_reward,
        "allow_bare_actions": bool(raw.get("allow_bare_actions", False)),
        "gamma": gamma,
        "reward_normalization": reward_normalization,
        "advantage_normalization": advantage_normalization,
        "whiten_rewards": bool(raw.get("whiten_rewards", True)),
        "whiten_advantages": bool(raw.get("whiten_advantages", True)),
        "kl_loss_coef": _nonnegative(raw.get("kl_loss_coef", 0.2), "kl_loss_coef"),
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_target_modules": lora_target_modules,
        "ref_log_prob_micro_batch_size_per_gpu": _positive_int(
            raw.get("ref_log_prob_micro_batch_size_per_gpu", 1),
            "ref_log_prob_micro_batch_size_per_gpu",
        ),
    }


def build_marshal_launch_overrides(config: dict[str, Any], *, config_path: str | None) -> tuple[str, ...]:
    settings = resolve_marshal_settings(config)
    verl = config.get("verl", {}) or {}
    if not isinstance(verl, dict):
        raise ValueError("MARSHAL verl settings must be a mapping.")
    source_config = config_path or str(Path.cwd())
    player_ids = list(settings["player_ids"])
    model_ids = [settings["shared_model_id"]] * 2
    required = (
        "trainer.use_v1=true",
        "algorithm.adv_estimator=reinforce_plus_plus",
        f"algorithm.gamma={settings['gamma']}",
        "algorithm.use_kl_in_reward=false",
        "critic.enable=false",
        "actor_rollout_ref.actor.policy_loss.loss_mode=vanilla",
        "actor_rollout_ref.actor.loss_agg_mode=token-mean",
        "actor_rollout_ref.actor.ppo_epochs=1",
        "actor_rollout_ref.actor.use_kl_loss=true",
        f"actor_rollout_ref.actor.kl_loss_coef={settings['kl_loss_coef']}",
        f"actor_rollout_ref.model.lora_rank={settings['lora_rank']}",
        f"actor_rollout_ref.model.lora_alpha={settings['lora_alpha']}",
        f"actor_rollout_ref.model.target_modules={_quote(settings['lora_target_modules'])}",
        (
            "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="
            f"{settings['ref_log_prob_micro_batch_size_per_gpu']}"
        ),
        f"++algorithm.extension_hooks_class={MARSHAL_HOOKS_FQN}",
        f"++algorithm.marshal.reward_normalization={settings['reward_normalization']}",
        f"++algorithm.marshal.advantage_normalization={settings['advantage_normalization']}",
        f"++algorithm.marshal.whiten_rewards={str(settings['whiten_rewards']).lower()}",
        f"++algorithm.marshal.whiten_advantages={str(settings['whiten_advantages']).lower()}",
        f"+agent.agent_ids={_hydra_list(player_ids)}",
        f"+agent.model_ids={_hydra_list(model_ids)}",
        "+agent.model_sharing=true",
        "+agent.orchestra_type=marshal_self_play",
        f"+agent.orchestra.marshal.game={settings['game']}",
        f"+agent.orchestra.marshal.player_ids={_hydra_list(player_ids)}",
        f"+agent.orchestra.marshal.shared_model_id={_quote(settings['shared_model_id'])}",
        f"+agent.orchestra.marshal.max_actions={settings['max_actions']}",
        f"+agent.orchestra.marshal.format_reward={settings['format_reward']}",
        f"+agent.orchestra.marshal.allow_bare_actions={str(settings['allow_bare_actions']).lower()}",
        "+trajweave.recipe=marshal_tictactoe_selfplay",
        f"+trajweave.config={_quote(source_config)}",
        "+trajweave.coordination_protocol=marshal_alternating_self_play",
        "+trajweave.trajectory_schema=marshal_per_player_turn_v1",
        "+trajweave.credit_allocator=marshal_turn_level_reinforce",
        "+trajweave.verl_extensions=[trajweave_marshal_turn_advantage]",
        f"+trajweave.agent_loop_backend={settings['agent_loop_backend']}",
        "+trajweave.turn_padding_multiple=2",
        f"+actor_rollout_ref.rollout.agent.agent_loop_manager_class={TRAJWEAVE_AGENT_LOOP_MANAGER_FQN}",
    )
    configured = tuple(str(item) for item in verl.get("overrides", []))
    required_keys = {_override_key(item) for item in required}
    retained = tuple(item for item in configured if _override_key(item) not in required_keys)
    return retained + required


def _hydra_list(values: list[str]) -> str:
    return "[" + ",".join(_quote(value) for value in values) + "]"


def _quote(value: Any) -> str:
    return '"' + str(value).replace('"', '\\"') + '"'


def _override_key(value: str) -> str:
    return value.split("=", 1)[0].lstrip("+")


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"MARSHAL {field} must be a positive integer.")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"MARSHAL {field} must be a positive integer.") from exc
    if result < 1 or float(value) != result:
        raise ValueError(f"MARSHAL {field} must be a positive integer.")
    return result


def _finite_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"MARSHAL {field} must be finite.") from exc
    if not isfinite(result):
        raise ValueError(f"MARSHAL {field} must be finite.")
    return result


def _nonnegative(value: Any, field: str) -> float:
    result = _finite_float(value, field)
    if result < 0:
        raise ValueError(f"MARSHAL {field} must be non-negative.")
    return result
