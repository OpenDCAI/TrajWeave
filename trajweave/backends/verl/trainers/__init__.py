from __future__ import annotations


def register_trajweave_trainers() -> None:
    from trajweave.backends.verl.trainers import multi_actor_sync  # noqa: F401


__all__ = ["register_trajweave_trainers"]
