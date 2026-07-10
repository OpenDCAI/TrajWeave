from __future__ import annotations

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
from trajweave.pipeline.context import RunContext
from trajweave.pipeline.export import maybe_export_dataproto
from trajweave.pipeline.launch import maybe_run_verl_launch
from trajweave.recipes.doctor_mas.math_smoke import default_math_tasks, default_team, run_smoke
from trajweave.recipes.doctor_mas.search_smoke import default_search_tasks, default_search_team, run_search_smoke
from trajweave.recipes.doctor_mas.train_tiny import TrainConfig, run_training
from trajweave.rollout.engine import RolloutEngine, RolloutResult


class DoctorMASRecipePlugin:
    name = "doctor_mas"

    def supports(self, context: RunContext) -> bool:
        return context.recipe_definition.family == "doctor_mas" or context.mode == "train_tiny"

    def run(self, context: RunContext) -> dict:
        if context.mode == "train_tiny":
            return self._run_tiny_training(context)

        summary, result = self._run_rollout_recipe(context)
        context.tracker.log_rollout_result(result, source=context.recipe_definition.runtime_recipe)
        output: dict[str, Any] = {
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
        if context.prepared_assets:
            output["prepared_assets"] = context.prepared_assets
        maybe_export_dataproto(context.config, result, output, tracker=context.tracker)
        maybe_run_verl_launch(context.config, output, default_enabled=False, tracker=context.tracker)
        return output

    def _run_rollout_recipe(self, context: RunContext):
        runtime_recipe = context.recipe_definition.runtime_recipe
        config = context.config
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
            engine = self._build_custom_engine(
                recipe=runtime_recipe,
                team=default_team(max_turns=max_turns),
                backend_cfg=backend_cfg,
            )
            result = engine.run(default_math_tasks(), rollouts_per_task=rollouts_per_task)
            return _summary_from_result(result), result

        if runtime_recipe == "doctor_mas_search":
            if backend_type == "rule":
                return run_search_smoke(rollouts_per_task=rollouts_per_task, max_turns=max_turns)
            engine = self._build_custom_engine(
                recipe=runtime_recipe,
                team=default_search_team(max_turns=max_turns),
                backend_cfg=backend_cfg,
            )
            result = engine.run(default_search_tasks(), rollouts_per_task=rollouts_per_task)
            return _summary_from_result(result), result

        raise ValueError(f"Unknown DoctorMAS runtime recipe: {runtime_recipe}")

    def _build_custom_engine(self, recipe: str, team: TeamSpec, backend_cfg: dict[str, Any]) -> RolloutEngine:
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
            policy_backend = (
                RuleBasedSearchPolicyBackend() if recipe == "doctor_mas_search" else RuleBasedMathPolicyBackend()
            )
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

    def _run_tiny_training(self, context: RunContext) -> dict[str, Any]:
        config = context.config
        training_cfg = config.get("training", {})
        backend_cfg = config.get("backend", {})
        output_dir = config.get("output", {}).get(
            "dir", training_cfg.get("output_dir", "outputs/doctor_mas_tiny_train")
        )
        train_config = TrainConfig(
            steps=int(training_cfg.get("steps", 120)),
            batch_tasks=int(training_cfg.get("batch_tasks", 8)),
            rollouts_per_task=int(
                config.get("rollout", {}).get("rollouts_per_task", training_cfg.get("rollouts_per_task", 8))
            ),
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
        history = run_training(train_config, tracker=context.tracker)
        output = {
            "run_id": context.run_id,
            "run_dir": str(context.run_dir),
            "config_path": context.config_path,
            "recipe": context.recipe,
            "mode": "train_tiny",
            "steps": train_config.steps,
            "output_dir": train_config.output_dir,
            "final_metrics": history[-1] if history else None,
        }
        context.tracker.log_metrics(output["final_metrics"] or {}, source="tiny_training")
        context.tracker.log_artifact(
            name="legacy_tiny_metrics.jsonl",
            path=f"{train_config.output_dir}/metrics.jsonl",
            kind="metrics",
        )
        context.tracker.log_artifact(
            name="legacy_tiny_policy.pt",
            path=f"{train_config.output_dir}/tiny_policy.pt",
            kind="checkpoint",
        )
        if context.prepared_assets:
            output["prepared_assets"] = context.prepared_assets
        return output


def _summary_from_result(result: RolloutResult):
    class Summary:
        trajectories = len(result.trajectories)
        samples = len(result.samples)
        success_rate = result.success_rate
        dataproto_rows = None
        dataproto_status = "skipped"

    return Summary()
