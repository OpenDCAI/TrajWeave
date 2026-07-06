from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trajweave.recipes.registry import RecipeDefinition
from trajweave.runtime import ExperimentTracker


@dataclass(frozen=True)
class RunContext:
    config: dict[str, Any]
    config_path: str | None
    recipe: str
    mode: str
    recipe_definition: RecipeDefinition
    run_id: str
    run_dir: Path
    tracker: ExperimentTracker
    prepared_assets: dict[str, str] | None = None

    def base_output(self) -> dict[str, Any]:
        output: dict[str, Any] = {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "config_path": self.config_path,
            "recipe": self.recipe,
            "canonical_recipe": self.recipe_definition.name,
            "mode": self.mode,
        }
        if self.prepared_assets:
            output["prepared_assets"] = self.prepared_assets
        return output
