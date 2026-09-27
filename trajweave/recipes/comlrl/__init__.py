from trajweave.recipes.comlrl.config import build_comlrl_launch_overrides, resolve_comlrl_topology

__all__ = [
    "CoMLRLRecipePlugin",
    "build_comlrl_launch_overrides",
    "comlrl_summary",
    "resolve_comlrl_topology",
]


def __getattr__(name: str):
    if name in {"CoMLRLRecipePlugin", "comlrl_summary"}:
        from trajweave.recipes.comlrl.plugin import CoMLRLRecipePlugin, comlrl_summary

        return {"CoMLRLRecipePlugin": CoMLRLRecipePlugin, "comlrl_summary": comlrl_summary}[name]
    raise AttributeError(name)
