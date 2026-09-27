from __future__ import annotations

from typing import Any

from trajweave.backends.verl.async_buffer import run_asymmetric_three_step_fixture
from trajweave.pipeline.context import RunContext
from trajweave.pipeline.launch import maybe_run_verl_launch
from trajweave.recipes.marti_mars2 import build_marti_mars2_launch_overrides, run_single_mcts_smoke
from trajweave.recipes.marti_mars2.acceptance import audit_fidelity_training_run, checkpoint_dir_from_overrides


class MARTIMARS2RecipePlugin:
    name = "marti_mars2"

    def supports(self, context: RunContext) -> bool:
        return context.recipe_definition.family == "marti_mars2"

    def run(self, context: RunContext) -> dict:
        if context.mode == "marti_eval":
            return self._run_marti_eval(context)
        if context.mode in {"verl_train", "verl_plan"}:
            return self._run_verl_train(context)
        return self._run_smoke(context)

    def _run_smoke(self, context: RunContext) -> dict:
        config = context.config
        rollout_cfg = config.get("rollout", {})
        mars2_cfg = config.get("marti_mars2", {})
        credit_mode = _credit_mode(config)
        summary, result = run_single_mcts_smoke(
            max_num_nodes=int(mars2_cfg.get("max_num_nodes", rollout_cfg.get("max_num_nodes", 2))),
            rollouts_per_task=int(rollout_cfg.get("rollouts_per_task", 1)),
            credit_mode=credit_mode,
        )
        context.tracker.log_rollout_result(result, source="marti_mars2_single_mcts")
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
            "marti_mars2": marti_mars2_summary(config),
        }
        if bool(mars2_cfg.get("async_fixture", False)):
            async_result = run_asymmetric_three_step_fixture()
            output["async_buffer_fixture"] = async_result
            if async_result["acceptance"]["status"] != "passed":
                raise RuntimeError(f"MARTI async-buffer fixture failed: {async_result['acceptance']}")
        if context.recipe != context.recipe_definition.name:
            output["canonical_recipe"] = context.recipe_definition.name
        if context.prepared_assets:
            output["prepared_assets"] = context.prepared_assets
        return output

    def _run_marti_eval(self, context: RunContext) -> dict[str, Any]:
        """Return/execute the optimizer-free standalone evaluation contract."""
        from trajweave.recipes.marti_mars2.eval import StandalonePolicyGroupEndpoints

        cfg = context.config.get("eval", {}) or {}
        model_paths = cfg.get("model_paths", {}) or {}
        if not model_paths:
            # model_ids alone are not enough: silently loading the training
            # actor would violate the standalone-eval boundary.
            raise ValueError("mode=marti_eval requires eval.model_paths keyed by policy group")
        endpoints = StandalonePolicyGroupEndpoints(model_paths)
        output = context.base_output()
        output["marti_eval"] = {
            "mode": "standalone_native_vllm",
            "optimizer": False,
            "policy_groups": list(model_paths),
            "dataset_revision": str(cfg.get("dataset_revision", "livecodebench-v5")),
            "task_count": int(cfg.get("task_count", 12)),
            "strategies": ["greedy", "best_of_n", "mcts"],
        }
        if bool(cfg.get("execute", False)):
            # An endpoint factory is supplied by the production launcher; the
            # local runner intentionally fails fast instead of falling back to
            # the optimizer's actor or an unverified reward model.
            endpoints.start()
        return output

    def _run_verl_train(self, context: RunContext) -> dict:
        output: dict[str, Any] = context.base_output()
        output["marti_mars2"] = marti_mars2_summary(context.config)
        overrides = build_marti_mars2_launch_overrides(context.config, config_path=context.config_path)
        maybe_run_verl_launch(
            context.config,
            output,
            overrides=overrides,
            default_enabled=True,
            default_module="trajweave.backends.verl.main_ppo",
            tracker=context.tracker,
            mode=context.mode,
        )
        if context.mode == "verl_train" and context.config.get("acceptance", {}).get("enabled", False):
            acceptance = audit_fidelity_training_run(
                context.run_dir,
                metric_summary=context.tracker.metric_aggregator.summarize(),
                checkpoint_dir=checkpoint_dir_from_overrides(context.config, run_dir=context.run_dir),
                require_correction=bool(context.config.get("marti_mars2", {}).get("enable_vllm_is_correction", False)),
                require_learning_signal=bool(context.config.get("acceptance", {}).get("require_learning_signal", True)),
                require_multi_agent_routing=bool(
                    context.config.get("acceptance", {}).get("require_multi_agent_routing", False)
                ),
                require_multi_actor_weight_sync=bool(
                    context.config.get("acceptance", {}).get("require_multi_actor_weight_sync", False)
                ),
                allow_local_verifier_fallback=bool(
                    context.config.get("acceptance", {}).get("allow_local_verifier_fallback", False)
                ),
            )
            acceptance.setdefault("name", "marti_mars2_fidelity")
            acceptance.setdefault(
                "message",
                "all configured fidelity checks passed"
                if acceptance.get("status") == "passed"
                else "one or more configured fidelity checks failed",
            )
            output["acceptance"] = acceptance
            output["marti_mars2_acceptance"] = acceptance
        return output


def marti_mars2_summary(config: dict[str, Any]) -> dict[str, Any]:
    mars2_cfg = config.get("marti_mars2", {})
    rollout_cfg = config.get("rollout", {})
    max_num_nodes = int(mars2_cfg.get("max_num_nodes", rollout_cfg.get("max_num_nodes", 2)))
    agent_ids = list(mars2_cfg.get("agent_ids", ["generator"]))
    model_ids = list(mars2_cfg.get("model_ids", ["shared"] * len(agent_ids)))
    search_mode = str(mars2_cfg.get("search_mode", "mcts"))
    return {
        "task": "code",
        "runtime_recipe": "marti_mars2_single_mcts",
        "agent_ids": agent_ids,
        "model_ids": model_ids,
        "max_num_nodes": max_num_nodes,
        "search_mode": search_mode,
        "coordination_protocol": (
            "independent_group_rollouts"
            if search_mode == "vanilla_grpo"
            else "mcts_selection_expansion_refinement_termination"
        ),
        "communication_graph": "single_agent_tree" if len(agent_ids) == 1 else "multi_agent_shared_tree",
        "aggregation": "group_normalized_rewards" if search_mode == "vanilla_grpo" else "best_path_or_mcts_eval",
        "credit_allocator": (
            "vanilla_group_grpo"
            if search_mode == "vanilla_grpo"
            else (
                "marti_mars2_tree_path_grpo"
                if _credit_mode(config) == "experimental"
                else "marti_mars2_fidelity_group_grpo"
            )
        ),
        "credit_mode": _credit_mode(config),
        "tree_identity_required": True,
        "tis_hook": bool(mars2_cfg.get("enable_vllm_is_correction", False)),
        "training_backend": (
            "trajweave_multi_actor_sync"
            if bool(mars2_cfg.get("multi_actor_training", len(agent_ids) > 1)) and len(agent_ids) > 1
            else "verl_v1_single_actor_wg"
        ),
    }


def _credit_mode(config: dict[str, Any]) -> str:
    credit_cfg = config.get("credit", {}) or {}
    configured = credit_cfg.get("mode")
    if configured is not None:
        mode = str(configured)
    else:
        recipe = str(config.get("recipe", ""))
        mode = "experimental" if "tree_credit_experimental" in recipe else "fidelity"
    if mode not in {"fidelity", "experimental"}:
        raise ValueError(f"Unsupported MARTI-MARS² credit mode: {mode!r}.")
    return mode
