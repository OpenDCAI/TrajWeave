from __future__ import annotations

from typing import Any

from trajweave.rollout.engine import RolloutResult
from trajweave.runtime import ExperimentTracker


def maybe_export_dataproto(
    config: dict[str, Any],
    result: RolloutResult,
    output: dict[str, Any],
    *,
    tracker: ExperimentTracker | None = None,
) -> None:
    export_cfg = config.get("export", {})
    dataproto_path = export_cfg.get("dataproto_path")
    if not dataproto_path:
        return
    try:
        from trajweave.backends.verl.export import export_dataproto

        path = export_dataproto(result.samples, dataproto_path)
        output["dataproto_path"] = str(path)
        output["dataproto_export_status"] = "ok"
        if tracker is not None:
            tracker.log_artifact(name="dataproto.pt", path=path, kind="dataproto")
    except (ModuleNotFoundError, RuntimeError) as exc:
        output["dataproto_export_status"] = f"unavailable: {exc}"
        if export_cfg.get("required", False):
            raise
