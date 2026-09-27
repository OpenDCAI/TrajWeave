from __future__ import annotations


def _ensure_torch_dtensor_import_compat() -> None:
    """测试环境下先补齐 VERL 当前快照依赖的 Torch 分布式符号。"""

    try:
        import builtins
        import torch.distributed.tensor as tensor_mod

        from torch.distributed._tensor import DTensor, Shard
        from torch.distributed._tensor.placement_types import DTensorSpec

        if not hasattr(tensor_mod, "DTensor"):
            tensor_mod.DTensor = DTensor
        if not hasattr(tensor_mod, "Shard"):
            tensor_mod.Shard = Shard
        for name, value in {
            "DTensor": DTensor,
            "Shard": Shard,
            "DTensorSpec": DTensorSpec,
        }.items():
            if not hasattr(builtins, name):
                setattr(builtins, name, value)
    except Exception:
        return


_ensure_torch_dtensor_import_compat()
