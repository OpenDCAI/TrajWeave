from __future__ import annotations

from typing import Any

from trajweave.pipeline.context import RunContext
from trajweave.pipeline.launch import maybe_run_verl_launch
from trajweave.recipes.maporl import build_maporl_launch_overrides, run_debate_math_smoke
from trajweave.recipes.maporl.config import resolve_maporl_credit_settings, validate_maporl_multi_actor_config


class MAPoRLRecipePlugin:
    name = "maporl"

    def supports(self, context: RunContext) -> bool:
        return context.recipe_definition.family == "maporl"

    def run(self, context: RunContext) -> dict:
        if context.mode in {"verl_train", "verl_plan"}:
            return self._run_verl_train(context)
        if context.mode == "smoke":
            return self._run_smoke(context)
        raise ValueError(f"Unsupported MAPoRL mode: {context.mode!r}.")

    def _run_smoke(self, context: RunContext) -> dict:
        config = context.config
        rollout_cfg = config.get("rollout", {})
        team_cfg = config.get("team", {})
        backend_cfg = config.get("backend", {})
        maporl_cfg = config.get("maporl", {})
        protocol_cfg = config.get("protocol", {})
        credit_settings = resolve_maporl_credit_settings(config)
        agent_count = int(maporl_cfg.get("agent_count", team_cfg.get("agent_count", 2)))
        agent_ids = tuple(maporl_cfg.get("agent_ids", [f"agent_{idx}" for idx in range(agent_count)]))
        model_ids = tuple(maporl_cfg.get("model_ids", ["shared"] * len(agent_ids)))
        consensus_threshold = int(
            protocol_cfg.get(
                "consensus_threshold",
                maporl_cfg.get("consensus_threshold", agent_count),
            )
        )
        summary, result = run_debate_math_smoke(
            backend=str(backend_cfg.get("type", "rule")),
            device=str(backend_cfg.get("device", "cpu")),
            agent_count=agent_count,
            agent_ids=agent_ids,
            model_ids=model_ids,
            rollouts_per_task=int(rollout_cfg.get("rollouts_per_task", 2)),
            max_turns=int(team_cfg.get("max_turns", 2)),
            consensus_threshold=consensus_threshold,
            early_stop=bool(protocol_cfg.get("early_stop", maporl_cfg.get("early_stop", True))),
            reward_feedback=bool(maporl_cfg.get("reward_feedback", protocol_cfg.get("reward_feedback", False))),
            criteria_for_consensus_percentage=(
                float(maporl_cfg["criteria_for_consensus_percentage"])
                if "criteria_for_consensus_percentage" in maporl_cfg
                else (
                    float(protocol_cfg["criteria_for_consensus_percentage"])
                    if "criteria_for_consensus_percentage" in protocol_cfg
                    else None
                )
            ),
            criteria_for_consensus_reward_threshold=float(
                maporl_cfg.get(
                    "criteria_for_consensus_reward_threshold",
                    protocol_cfg.get("criteria_for_consensus_reward_threshold", 0.7),
                )
            ),
            rule_horizon=credit_settings["rule_horizon"],
            rule_agent_share=credit_settings["rule_agent_share"],
            rule_discount=credit_settings["rule_discount"],
            alpha=credit_settings["alpha"],
            policy_separation=bool(maporl_cfg.get("policy_separation", True)),
            collaboration_separation=bool(maporl_cfg.get("collaboration_separation", True)),
            task_training=bool(maporl_cfg.get("task_training", False)),
        )
        context.tracker.log_rollout_result(result, source="maporl_debate_math")
        output = {
            "run_id": context.run_id,
            "run_dir": str(context.run_dir),
            "config_path": context.config_path,
            "recipe": context.recipe,
            "mode": context.mode,
            "trajectories": summary.trajectories,
            "samples": summary.samples,
            "success_rate": summary.success_rate,
            "dataproto_status": summary.dataproto_status,
            "dataproto_rows": summary.dataproto_rows,
        }
        if context.recipe != context.recipe_definition.name:
            output["canonical_recipe"] = context.recipe_definition.name
        if context.prepared_assets:
            output["prepared_assets"] = context.prepared_assets
        return output

    def _run_verl_train(self, context: RunContext) -> dict:
        output: dict[str, Any] = context.base_output()
        output["maporl"] = maporl_summary(context.config)
        overrides = build_maporl_launch_overrides(context.config, config_path=context.config_path)
        maybe_run_verl_launch(
            context.config,
            output,
            overrides=overrides,
            default_enabled=True,
            default_module="trajweave.backends.verl.main_ppo",
            tracker=context.tracker,
            mode=context.mode,
        )
        return output


def maporl_summary(config: dict[str, Any]) -> dict[str, Any]:
    maporl_cfg = config.get("maporl", {})
    team_cfg = config.get("team", {})
    protocol_cfg = config.get("protocol", {})
    agent_count = int(maporl_cfg.get("agent_count", team_cfg.get("agent_count", 2)))
    model_ids = list(maporl_cfg.get("model_ids", ["shared"] * agent_count))
    worker_groups = maporl_worker_groups_summary(maporl_cfg, model_ids=model_ids)
    trainable_groups = [group_id for group_id, group in worker_groups.items() if group.get("trainable", True)]
    native_multi_actor_training = bool(maporl_cfg.get("multi_actor_training", len(trainable_groups) > 1))
    validation = validate_maporl_multi_actor_config(
        maporl_cfg,
        worker_groups=worker_groups,
        model_ids=tuple(str(model_id) for model_id in model_ids),
        multi_actor_training=native_multi_actor_training,
    )
    return {
        "task": "math",
        "runtime_recipe": "maporl_debate_math",
        "agent_ids": list(maporl_cfg.get("agent_ids", [f"agent_{idx}" for idx in range(agent_count)])),
        "model_ids": model_ids,
        "worker_groups": worker_groups,
        "coordination_protocol": "debate_consensus",
        "communication_graph": protocol_cfg.get("communication_graph", "fully_connected"),
        "aggregation": protocol_cfg.get("aggregation", "consensus"),
        "credit_allocator": "maporl_ppo_score_rule",
        "single_model_only": len(set(model_ids)) == 1,
        "trainable_worker_groups": trainable_groups,
        "native_multi_actor_training": native_multi_actor_training,
        "training_backend": "verl_v1_multi_actor_wg" if native_multi_actor_training else "verl_v1_single_actor_wg",
        "tokenizer_mode": validation["tokenizer_mode"],
        "multi_actor_validation_status": validation["status"],
        "multi_actor_validation": validation,
        "routing_field": "worker_group",
        "policy_separation": bool(maporl_cfg.get("policy_separation", True)),
        "collaboration_separation": bool(maporl_cfg.get("collaboration_separation", True)),
        "reward_feedback": bool(maporl_cfg.get("reward_feedback", protocol_cfg.get("reward_feedback", False))),
    }


def maporl_worker_groups_summary(maporl_cfg: dict[str, Any], *, model_ids: list[str]) -> dict[str, dict[str, Any]]:
    raw_groups = maporl_cfg.get("worker_groups", {}) or {}
    groups: dict[str, dict[str, Any]] = {}
    for model_id in dict.fromkeys(model_ids):
        raw_group = raw_groups.get(model_id, {}) if isinstance(raw_groups, dict) else {}
        group = dict(raw_group) if isinstance(raw_group, dict) else {}
        group.setdefault("trainable", True)
        groups[str(model_id)] = group
    if isinstance(raw_groups, dict):
        for group_id, raw_group in raw_groups.items():
            if group_id in groups:
                continue
            group = dict(raw_group) if isinstance(raw_group, dict) else {}
            group.setdefault("trainable", False)
            groups[str(group_id)] = group
    return groups
