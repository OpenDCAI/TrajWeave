from trajweave.pipeline.config import load_yaml_config, mode_name, recipe_name
from trajweave.pipeline.context import RunContext

__all__ = ["RunContext", "load_yaml_config", "mode_name", "recipe_name", "run_recipe"]


def __getattr__(name: str):
    # The registry imports recipe plugins, while recipe plugins import
    # ``pipeline.context``. Resolve it lazily to keep direct plugin imports
    # usable without a package-initialization cycle.
    if name == "run_recipe":
        from trajweave.pipeline.registry import run_recipe

        return run_recipe
    raise AttributeError(name)
