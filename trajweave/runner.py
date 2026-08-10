from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from trajweave.pipeline.assets import maybe_prepare_assets
from trajweave.pipeline.config import load_yaml_config, mode_name, recipe_name
from trajweave.pipeline.context import RunContext
from trajweave.pipeline.registry import run_recipe
from trajweave.recipes.registry import resolve_recipe, validate_recipe_mode
from trajweave.runtime import ExperimentTracker
from trajweave.storage import RunStore, RunStoreConfig
from trajweave.storage.serialization import redact_secrets


def run_from_config_path(path: str | Path) -> dict[str, Any]:
    return run_from_config(load_yaml_config(path), config_path=str(path))


def run_from_config(config: dict[str, Any], config_path: str | None = None) -> dict[str, Any]:
    recipe = recipe_name(config)
    recipe_definition = resolve_recipe(recipe)
    mode = mode_name(config)
    validate_recipe_mode(recipe_definition, mode)
    run_store = RunStore(
        RunStoreConfig.from_config(config, recipe=recipe),
        recipe=recipe,
        canonical_recipe=recipe_definition.name,
        mode=mode,
        config_path=config_path,
        raw_config=config,
    )
    run_store.initialize()
    try:
        tracker = ExperimentTracker(run_store, logging_config=config.get("logging", {}))
        tracker.log_event(
            "run_started",
            "TrajWeave run started.",
            {"recipe": recipe, "canonical_recipe": recipe_definition.name, "mode": mode},
        )
        try:
            prepared_assets = maybe_prepare_assets(config)
            context = RunContext(
                config=config,
                config_path=config_path,
                recipe=recipe,
                mode=mode,
                recipe_definition=recipe_definition,
                run_id=run_store.run_id,
                run_dir=run_store.run_dir,
                tracker=tracker,
                prepared_assets=prepared_assets,
            )
            if prepared_assets:
                for name, path in prepared_assets.items():
                    tracker.log_artifact(name=name, path=path, kind="prepared_asset")
            output = run_recipe(context)
            output = _enforce_runtime_identity(output, run_store=run_store)
            output = tracker.write_summary(output)
            failure_reason = _output_failure_reason(output)
            if failure_reason:
                tracker.finalize("failed", error=failure_reason)
            elif _output_is_plan(output):
                tracker.finalize("planned")
            else:
                tracker.finalize("completed")
            return output
        except Exception as exc:
            tracker.log_event("run_failed", "TrajWeave run failed.", {"error": str(exc)}, level="ERROR")
            tracker.finalize("failed", error=str(exc))
            raise
    finally:
        run_store.release_lock()


def dumps_result(result: dict[str, Any]) -> str:
    return json.dumps(redact_secrets(result), ensure_ascii=False, indent=2)


def result_exit_code(result: dict[str, Any]) -> int:
    return 1 if _output_failure_reason(result) else 0


def _output_failure_reason(output: dict[str, Any]) -> str | None:
    verl_launch = output.get("verl_launch")
    if not isinstance(verl_launch, dict):
        if output.get("mode") == "verl_train":
            return "mode='verl_train' completed without a VERL launch result."
        return None
    status = verl_launch.get("status")
    returncode = verl_launch.get("returncode")
    if output.get("mode") == "verl_train" and status in {"disabled", "dry_run"}:
        return f"VERL training did not execute: launch status={status!r}."
    if status == "failed" or (isinstance(returncode, int) and returncode != 0):
        validation_error = verl_launch.get("validation_error")
        if validation_error:
            return f"VERL training validation failed: {validation_error}"
        return f"VERL launch failed with status={status!r}, returncode={returncode!r}."
    acceptance = output.get("marti_mars2_acceptance")
    if isinstance(acceptance, dict) and acceptance.get("status") == "failed":
        failed_checks = [name for name, passed in acceptance.get("checks", {}).items() if not passed]
        return f"MARTI-MARS2 fidelity acceptance failed: {', '.join(failed_checks) or 'unknown check'}."
    return None


def _output_is_plan(output: dict[str, Any]) -> bool:
    verl_launch = output.get("verl_launch")
    return (
        output.get("mode") == "verl_plan" and isinstance(verl_launch, dict) and verl_launch.get("status") == "dry_run"
    )


def _enforce_runtime_identity(output: dict[str, Any], *, run_store: RunStore) -> dict[str, Any]:
    if not isinstance(output, dict):
        raise TypeError("Recipe output must be a mapping.")
    output_run_id = output.get("run_id")
    if output_run_id is not None and output_run_id != run_store.run_id:
        raise RuntimeError(f"Recipe output run_id {output_run_id!r} does not match parent run_id {run_store.run_id!r}.")
    output_run_dir = output.get("run_dir")
    if output_run_dir is not None and Path(str(output_run_dir)).expanduser().resolve() != run_store.run_dir:
        raise RuntimeError(
            f"Recipe output run_dir {output_run_dir!r} does not match parent run_dir {str(run_store.run_dir)!r}."
        )
    output["run_id"] = run_store.run_id
    output["run_dir"] = str(run_store.run_dir)
    return output


__all__ = ["dumps_result", "load_yaml_config", "result_exit_code", "run_from_config", "run_from_config_path"]
