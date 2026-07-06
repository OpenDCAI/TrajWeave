from types import SimpleNamespace

import torch

from verl.utils.model import load_valuehead_model


def test_load_valuehead_model_respects_configured_attention(monkeypatch):
    captured = {}
    model_config = SimpleNamespace(_attn_implementation_internal="eager")

    class FakeTokenClassification:
        @staticmethod
        def from_pretrained(**kwargs):
            captured.update(kwargs)
            return object()

    monkeypatch.setattr(
        "transformers.AutoModelForTokenClassification",
        FakeTokenClassification,
    )

    load_valuehead_model(
        local_path="/tmp/model",
        torch_dtype=torch.float32,
        model_config=model_config,
        trust_remote_code=False,
    )

    assert captured["attn_implementation"] == "eager"
