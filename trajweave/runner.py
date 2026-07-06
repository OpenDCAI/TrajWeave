from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from trajweave.backends.hf import HFTransformersPolicyBackend
from trajweave.backends.local import RuleBasedMathPolicyBackend, TinyTorchPolicyBackend
from trajweave.backends.search import RuleBasedSearchPolicyBackend
from trajweave.core.specs import TeamSpec
from trajweave.credit.doctor_mas import DoctorMASCreditAssigner
from trajweave.envs.math import SolverVerifierMathEnvironment
from trajweave.envs.search import SearchAnswerEnvironment
from trajweave.orchestration.search_answer import SearchAnswerOrchestra
from trajweave.orchestration.solver_verifier import SolverVerifierOrchestra
from trajweave.recipes.doctor_mas.math_smoke import default_math_tasks, default_team, run_smoke
from trajweave.recipes.doctor_mas.search_smoke import (
    default_search_tasks,
    default_search_team,
    run_search_smoke,
)
from trajweave.recipes.doctor_mas.train_tiny import TrainConfig, run_training
from trajweave.recipes.drmas_native import build_drmas_native_launch_overrides, recipe_spec
from trajweave.recipes.maporl import build_maporl_launch_overrides, run_debate_math_smoke
from trajweave.recipes.registry import resolve_recipe
from trajweave.rollout.engine import RolloutEngine, RolloutResult


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("YAML config loading requires PyYAML.") from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def run_from_config_path(path: str | Path) -> dict[str, Any]:
    return run_from_config(load_yaml_config(path), config_path=str(path))


def run_from_config(config: dict[str, Any], config_path: str | None = None) -> dict[str, Any]:
    recipe = _recipe_name(config)
    recipe_definition = resolve_recipe(recipe)
    mode = _mode(config)
    prepared_assets = _maybe_prepare_assets(config)
    if recipe_definition.family == "drmas_native":
        output: dict[str, Any] = {
            "config_path": config_path,
            "recipe": recipe,
            "canonical_recipe": recipe_definition.name,
            "mode": mode,
            "drmas_native": _drmas_native_summary(config, recipe),
        }
        if prepared_assets:
            output["prepared_assets"] = prepared_assets
        _maybe_prepare_drmas_native_verl_launch(config, output, config_path=config_path)
        return output
    if recipe_definition.family == "maporl" and mode == "verl_train":
        output = {
            "config_path": config_path,
            "recipe": recipe,
            "canonical_recipe": recipe_definition.name,
            "mode": mode,
            "maporl": _maporl_summary(config),
        }
        if prepared_assets:
            output["prepared_assets"] = prepared_assets
        _maybe_prepare_maporl_verl_launch(config, output, config_path=config_path)
        return output
    if mode == "train_tiny":
        result = _run_tiny_training(config, config_path=config_path)
        if prepared_assets:
            result["prepared_assets"] = prepared_assets
        return result

    summary, result = _run_rollout_recipe(config, recipe=recipe)
    output: dict[str, Any] = {
        "config_path": config_path,
        "recipe": recipe,
        "mode": mode,
        "trajectories": summary.trajectories,
        "samples": summary.samples,
        "success_rate": summary.success_rate,
        "dataproto_status": summary.dataproto_status,
        "dataproto_rows": summary.dataproto_rows,
    }
    if prepared_assets:
        output["prepared_assets"] = prepared_assets
    _maybe_export_dataproto(config, result, output)
    _maybe_prepare_verl_launch(config, output)
    return output


def _recipe_name(config: dict[str, Any]) -> str:
    return str(config.get("recipe", config.get("run", {}).get("recipe", "doctor_mas_math")))


def _mode(config: dict[str, Any]) -> str:
    return str(config.get("mode", config.get("run", {}).get("mode", "smoke")))


def _run_rollout_recipe(config: dict[str, Any], recipe: str):
    recipe_definition = resolve_recipe(recipe)
    runtime_recipe = recipe_definition.runtime_recipe
    rollout_cfg = config.get("rollout", {})
    team_cfg = config.get("team", {})
    backend_cfg = config.get("backend", {})
    rollouts_per_task = int(rollout_cfg.get("rollouts_per_task", 2))
    max_turns = int(team_cfg.get("max_turns", 2))
    backend_type = str(backend_cfg.get("type", "rule"))
    device = str(backend_cfg.get("device", "cpu"))

    if runtime_recipe == "doctor_mas_math":
        if backend_type in {"rule", "tiny-torch"}:
            return run_smoke(
                backend=backend_type,
                device=device,
                rollouts_per_task=rollouts_per_task,
                max_turns=max_turns,
            )
        engine = _build_custom_engine(
            recipe=runtime_recipe,
            team=default_team(max_turns=max_turns),
            backend_cfg=backend_cfg,
        )
        result = engine.run(default_math_tasks(), rollouts_per_task=rollouts_per_task)
        return _summary_from_result(result), result
    if runtime_recipe == "doctor_mas_search":
        if backend_type == "rule":
            return run_search_smoke(rollouts_per_task=rollouts_per_task, max_turns=max_turns)
        engine = _build_custom_engine(
            recipe=runtime_recipe,
            team=default_search_team(max_turns=max_turns),
            backend_cfg=backend_cfg,
        )
        result = engine.run(default_search_tasks(), rollouts_per_task=rollouts_per_task)
        return _summary_from_result(result), result
    if runtime_recipe == "maporl_debate_math":
        maporl_cfg = config.get("maporl", {})
        protocol_cfg = config.get("protocol", {})
        credit_cfg = config.get("credit", {})
        agent_count = int(config.get("maporl", {}).get("agent_count", team_cfg.get("agent_count", 2)))
        agent_ids = tuple(maporl_cfg.get("agent_ids", [f"agent_{idx}" for idx in range(agent_count)]))
        model_ids = tuple(maporl_cfg.get("model_ids", ["shared"] * len(agent_ids)))
        consensus_threshold = int(
            protocol_cfg.get(
                "consensus_threshold",
                maporl_cfg.get("consensus_threshold", agent_count),
            )
        )
        return run_debate_math_smoke(
            backend=backend_type,
            device=device,
            agent_count=agent_count,
            agent_ids=agent_ids,
            model_ids=model_ids,
            rollouts_per_task=rollouts_per_task,
            max_turns=max_turns,
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
            rule_horizon=str(credit_cfg.get("rule_horizon", maporl_cfg.get("rule_horizon", "discounted_sum"))),
            rule_agent_share=str(credit_cfg.get("rule_agent_share", maporl_cfg.get("rule_agent_share", "all"))),
            rule_discount=float(credit_cfg.get("rule_discount", maporl_cfg.get("rule_discount", 0.3))),
            alpha=tuple(float(value) for value in credit_cfg.get("alpha", maporl_cfg.get("alpha", [0, 0, 0, 0]))),
            policy_separation=bool(maporl_cfg.get("policy_separation", True)),
            collaboration_separation=bool(maporl_cfg.get("collaboration_separation", True)),
            task_training=bool(maporl_cfg.get("task_training", False)),
        )
    raise ValueError(f"Unknown recipe: {recipe}")


def _build_custom_engine(recipe: str, team: TeamSpec, backend_cfg: dict[str, Any]) -> RolloutEngine:
    backend_type = str(backend_cfg.get("type"))
    if backend_type == "hf-transformers":
        policy_backend = HFTransformersPolicyBackend(
            model_path=str(backend_cfg["model_path"]),
            device=backend_cfg.get("device"),
            device_map=backend_cfg.get("device_map"),
            torch_dtype=backend_cfg.get("torch_dtype"),
            max_new_tokens=int(backend_cfg.get("max_new_tokens", 128)),
            generation_config=backend_cfg.get("generation_config", {}),
        )
    elif backend_type == "tiny-torch":
        policy_backend = TinyTorchPolicyBackend(device=str(backend_cfg.get("device", "cpu")))
    elif backend_type == "rule":
        policy_backend = RuleBasedSearchPolicyBackend() if recipe == "doctor_mas_search" else RuleBasedMathPolicyBackend()
    else:
        raise ValueError(f"Unknown backend.type: {backend_type}")

    if recipe == "doctor_mas_math":
        return RolloutEngine(
            team=team,
            orchestra=SolverVerifierOrchestra(),
            environment=SolverVerifierMathEnvironment(),
            policy_backend=policy_backend,
            credit_assigner=DoctorMASCreditAssigner(),
        )
    if recipe == "doctor_mas_search":
        return RolloutEngine(
            team=team,
            orchestra=SearchAnswerOrchestra(),
            environment=SearchAnswerEnvironment(),
            policy_backend=policy_backend,
            credit_assigner=DoctorMASCreditAssigner(),
        )
    raise ValueError(f"Unknown recipe: {recipe}")


def _run_tiny_training(config: dict[str, Any], config_path: str | None) -> dict[str, Any]:
    training_cfg = config.get("training", {})
    backend_cfg = config.get("backend", {})
    output_dir = config.get("output", {}).get("dir", training_cfg.get("output_dir", "outputs/doctor_mas_tiny_train"))
    train_config = TrainConfig(
        steps=int(training_cfg.get("steps", 120)),
        batch_tasks=int(training_cfg.get("batch_tasks", 8)),
        rollouts_per_task=int(config.get("rollout", {}).get("rollouts_per_task", training_cfg.get("rollouts_per_task", 8))),
        max_turns=int(config.get("team", {}).get("max_turns", training_cfg.get("max_turns", 2))),
        eval_interval=int(training_cfg.get("eval_interval", 10)),
        lr=float(training_cfg.get("lr", 0.03)),
        entropy_coef=float(training_cfg.get("entropy_coef", 0.02)),
        seed=int(training_cfg.get("seed", 7)),
        device=str(backend_cfg.get("device", training_cfg.get("device", "cpu"))),
        output_dir=str(output_dir),
        task_mode=str(training_cfg.get("task_mode", "fixed")),
        num_solver_candidates=int(training_cfg.get("num_solver_candidates", 3)),
        train_verifier=bool(training_cfg.get("train_verifier", False)),
    )
    history = run_training(train_config)
    return {
        "config_path": config_path,
        "recipe": _recipe_name(config),
        "mode": "train_tiny",
        "steps": train_config.steps,
        "output_dir": train_config.output_dir,
        "final_metrics": history[-1] if history else None,
    }


def _maybe_prepare_assets(config: dict[str, Any]) -> dict[str, str] | None:
    tiny_cfg = config.get("prepare", {}).get("tiny_verl_assets", {})
    if not tiny_cfg.get("enabled", False):
        return None
    from trajweave.backends.verl.tiny_assets import prepare_tiny_verl_assets

    return prepare_tiny_verl_assets(
        output_dir=tiny_cfg.get("output_dir", "outputs/doctor_mas_verl_tiny_assets"),
        train_size=int(tiny_cfg.get("train_size", 2)),
        val_size=int(tiny_cfg.get("val_size", 2)),
        task_family=str(tiny_cfg.get("task_family", "math")),
        overwrite=bool(tiny_cfg.get("overwrite", True)),
    )


def _summary_from_result(result: RolloutResult):
    class Summary:
        trajectories = len(result.trajectories)
        samples = len(result.samples)
        success_rate = result.success_rate
        dataproto_rows = None
        dataproto_status = "skipped"

    return Summary()


def _maybe_export_dataproto(config: dict[str, Any], result: RolloutResult, output: dict[str, Any]) -> None:
    export_cfg = config.get("export", {})
    dataproto_path = export_cfg.get("dataproto_path")
    if not dataproto_path:
        return
    try:
        from trajweave.backends.verl.export import export_dataproto

        path = export_dataproto(result.samples, dataproto_path)
        output["dataproto_path"] = str(path)
        output["dataproto_export_status"] = "ok"
    except (ModuleNotFoundError, RuntimeError) as exc:
        output["dataproto_export_status"] = f"unavailable: {exc}"
        if export_cfg.get("required", False):
            raise


def _maybe_prepare_verl_launch(config: dict[str, Any], output: dict[str, Any]) -> None:
    verl_cfg = config.get("verl", {})
    if not verl_cfg.get("enabled", False):
        return
    from trajweave.backends.verl.launcher import VerlTrainerLaunchConfig, VerlTrainerLauncher

    launch_config = VerlTrainerLaunchConfig(
        python=str(verl_cfg.get("python", VerlTrainerLaunchConfig.python)),
        module=str(verl_cfg.get("module", "verl.trainer.main_ppo")),
        overrides=tuple(str(item) for item in verl_cfg.get("overrides", [])),
        env=dict(verl_cfg.get("env", {})),
        cwd=verl_cfg.get("cwd"),
        execute=bool(verl_cfg.get("execute", False)),
        stdout_path=verl_cfg.get("stdout_path"),
        stderr_path=verl_cfg.get("stderr_path"),
    )
    launcher = VerlTrainerLauncher(launch_config)
    command_file = verl_cfg.get("command_file")
    if command_file:
        output["verl_command_file"] = str(launcher.write_command_file(command_file))
    output["verl_launch"] = launcher.run()


def _maybe_prepare_drmas_native_verl_launch(
    config: dict[str, Any],
    output: dict[str, Any],
    *,
    config_path: str | None,
) -> None:
    verl_cfg = config.get("verl", {})
    if not verl_cfg.get("enabled", True):
        output["verl_launch"] = {"status": "disabled"}
        return
    from trajweave.backends.verl.launcher import VerlTrainerLaunchConfig, VerlTrainerLauncher

    overrides = build_drmas_native_launch_overrides(config, config_path=config_path)
    launch_config = VerlTrainerLaunchConfig(
        python=str(verl_cfg.get("python", VerlTrainerLaunchConfig.python)),
        module=str(verl_cfg.get("module", "trajweave.backends.verl.main_ppo")),
        overrides=overrides,
        env=dict(verl_cfg.get("env", {})),
        cwd=verl_cfg.get("cwd"),
        execute=bool(verl_cfg.get("execute", False)),
        stdout_path=verl_cfg.get("stdout_path"),
        stderr_path=verl_cfg.get("stderr_path"),
    )
    launcher = VerlTrainerLauncher(launch_config)
    command_file = verl_cfg.get("command_file")
    if command_file:
        output["verl_command_file"] = str(launcher.write_command_file(command_file))
    output["verl_launch"] = launcher.run()


def _maybe_prepare_maporl_verl_launch(
    config: dict[str, Any],
    output: dict[str, Any],
    *,
    config_path: str | None,
) -> None:
    verl_cfg = config.get("verl", {})
    if not verl_cfg.get("enabled", True):
        output["verl_launch"] = {"status": "disabled"}
        return
    from trajweave.backends.verl.launcher import VerlTrainerLaunchConfig, VerlTrainerLauncher

    overrides = build_maporl_launch_overrides(config, config_path=config_path)
    launch_config = VerlTrainerLaunchConfig(
        python=str(verl_cfg.get("python", VerlTrainerLaunchConfig.python)),
        module=str(verl_cfg.get("module", "trajweave.backends.verl.main_ppo")),
        overrides=overrides,
        env=dict(verl_cfg.get("env", {})),
        cwd=verl_cfg.get("cwd"),
        execute=bool(verl_cfg.get("execute", False)),
        stdout_path=verl_cfg.get("stdout_path"),
        stderr_path=verl_cfg.get("stderr_path"),
    )
    launcher = VerlTrainerLauncher(launch_config)
    command_file = verl_cfg.get("command_file")
    if command_file:
        output["verl_command_file"] = str(launcher.write_command_file(command_file))
    output["verl_launch"] = launcher.run()


def _is_drmas_native_recipe(recipe: str) -> bool:
    return resolve_recipe(recipe).family == "drmas_native"


def _drmas_native_summary(config: dict[str, Any], recipe: str) -> dict[str, Any]:
    spec = recipe_spec(recipe)
    native_cfg = config.get("drmas_native", {})
    return {
        "task": spec.task,
        "runtime_recipe": spec.runtime_recipe,
        "agent_ids": list(native_cfg.get("agent_ids", spec.agent_ids)),
        "orchestra_type": spec.orchestra_type,
        "coordination_protocol": spec.coordination_protocol,
        "group_by_agent_id": True,
    }


def _maporl_summary(config: dict[str, Any]) -> dict[str, Any]:
    maporl_cfg = config.get("maporl", {})
    team_cfg = config.get("team", {})
    protocol_cfg = config.get("protocol", {})
    agent_count = int(maporl_cfg.get("agent_count", team_cfg.get("agent_count", 2)))
    return {
        "task": "math",
        "runtime_recipe": "maporl_debate_math",
        "agent_ids": list(maporl_cfg.get("agent_ids", [f"agent_{idx}" for idx in range(agent_count)])),
        "coordination_protocol": "debate_consensus",
        "communication_graph": protocol_cfg.get("communication_graph", "fully_connected"),
        "aggregation": protocol_cfg.get("aggregation", "consensus"),
        "credit_allocator": "maporl_ppo_score_rule",
        "single_model_only": len(set(maporl_cfg.get("model_ids", ["shared"] * agent_count))) == 1,
        "policy_separation": bool(maporl_cfg.get("policy_separation", True)),
        "collaboration_separation": bool(maporl_cfg.get("collaboration_separation", True)),
        "reward_feedback": bool(maporl_cfg.get("reward_feedback", protocol_cfg.get("reward_feedback", False))),
    }


def dumps_result(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, indent=2)
