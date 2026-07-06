from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from trajweave.pipeline.assets import maybe_prepare_assets
from trajweave.pipeline.config import load_yaml_config, mode_name, recipe_name
from trajweave.pipeline.context import RunContext
from trajweave.pipeline.registry import run_recipe
from trajweave.recipes.registry import resolve_recipe
from trajweave.runtime import ExperimentTracker
from trajweave.storage import RunStore, RunStoreConfig


def run_from_config_path(path: str | Path) -> dict[str, Any]:
    return run_from_config(load_yaml_config(path), config_path=str(path))


def run_from_config(config: dict[str, Any], config_path: str | None = None) -> dict[str, Any]:
    recipe = recipe_name(config)
    recipe_definition = resolve_recipe(recipe)
    mode = mode_name(config)
    run_store = RunStore(
        RunStoreConfig.from_config(config, recipe=recipe),
        recipe=recipe,
        canonical_recipe=recipe_definition.name,
        mode=mode,
        config_path=config_path,
        raw_config=config,
    )
    run_store.initialize()
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
        output.setdefault("run_id", run_store.run_id)
        output.setdefault("run_dir", str(run_store.run_dir))
        output = tracker.write_summary(output)
        tracker.finalize("completed")
        return output
    except Exception as exc:
        tracker.log_event("run_failed", "TrajWeave run failed.", {"error": str(exc)}, level="ERROR")
        tracker.finalize("failed", error=str(exc))
        raise


def dumps_result(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, indent=2)


__all__ = ["dumps_result", "load_yaml_config", "run_from_config", "run_from_config_path"]
