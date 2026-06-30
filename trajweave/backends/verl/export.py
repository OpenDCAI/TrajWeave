from __future__ import annotations

from pathlib import Path

from trajweave.core.trajectory import TrainingSample


def export_dataproto(samples: list[TrainingSample], path: str | Path) -> Path:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("Exporting VERL DataProto requires torch.") from exc

    from trajweave.backends.verl.dataproto import VerlDataProtoAdapter

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataproto = VerlDataProtoAdapter().build(samples)
    torch.save(dataproto, output_path)
    return output_path
