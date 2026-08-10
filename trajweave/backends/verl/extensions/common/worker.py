from __future__ import annotations

from verl.workers.engine_workers import ActorRolloutRefWorker


class TrajWeaveActorRolloutRefWorker(ActorRolloutRefWorker):
    """Shared actor worker that installs TrajWeave compatibility shims."""

    def __init__(self, *args, **kwargs):
        from trajweave.backends.verl import _ensure_torch_dtensor_import_compat
        from trajweave.backends.verl.extensions.common.nested_compat import install_worker_nested_tensor_compat

        _ensure_torch_dtensor_import_compat()
        install_worker_nested_tensor_compat()
        super().__init__(*args, **kwargs)
