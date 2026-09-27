from __future__ import annotations

from typing import Any

from trajweave.pipeline.config import recipe_name


def maybe_prepare_assets(config: dict[str, Any]) -> dict[str, str] | None:
    prepare_cfg = config.get("prepare", {}) or {}
    tiny_cfg = prepare_cfg.get("tiny_verl_assets", {}) or {}
    dataset_cfg = prepare_cfg.get("verl_dataset", {}) or {}
    if tiny_cfg.get("enabled", False) and dataset_cfg.get("enabled", False):
        raise ValueError("prepare.tiny_verl_assets and prepare.verl_dataset cannot both be enabled.")
    if dataset_cfg.get("enabled", False):
        from trajweave.backends.verl.tiny_assets import prepare_verl_dataset

        return prepare_verl_dataset(
            output_dir=dataset_cfg.get("output_dir", f"outputs/{recipe_name(config)}_dataset"),
            train_size=int(dataset_cfg.get("train_size", 2)),
            val_size=int(dataset_cfg.get("val_size", 2)),
            task_family=str(dataset_cfg.get("task_family", "math")),
            recipe_name=str(dataset_cfg.get("recipe_name", recipe_name(config))),
            overwrite=bool(dataset_cfg.get("overwrite", True)),
        )
    if not tiny_cfg.get("enabled", False):
        return None
    from trajweave.backends.verl.tiny_assets import prepare_tiny_verl_assets

    return prepare_tiny_verl_assets(
        output_dir=tiny_cfg.get("output_dir", f"outputs/{recipe_name(config)}_tiny_assets"),
        train_size=int(tiny_cfg.get("train_size", 2)),
        val_size=int(tiny_cfg.get("val_size", 2)),
        task_family=str(tiny_cfg.get("task_family", "math")),
        recipe_name=str(tiny_cfg.get("recipe_name", recipe_name(config))),
        overwrite=bool(tiny_cfg.get("overwrite", True)),
    )
