from __future__ import annotations

from typing import Any

from trajweave.pipeline.config import recipe_name


def maybe_prepare_assets(config: dict[str, Any]) -> dict[str, str] | None:
    tiny_cfg = config.get("prepare", {}).get("tiny_verl_assets", {})
    if not tiny_cfg.get("enabled", False):
        return None
    from trajweave.backends.verl.tiny_assets import prepare_tiny_verl_assets

    return prepare_tiny_verl_assets(
        output_dir=tiny_cfg.get("output_dir", "outputs/doctor_mas_verl_tiny_assets"),
        train_size=int(tiny_cfg.get("train_size", 2)),
        val_size=int(tiny_cfg.get("val_size", 2)),
        task_family=str(tiny_cfg.get("task_family", "math")),
        recipe_name=str(tiny_cfg.get("recipe_name", recipe_name(config))),
        overwrite=bool(tiny_cfg.get("overwrite", True)),
    )
