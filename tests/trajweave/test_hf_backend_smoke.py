import pytest

from trajweave.backends.hf import HFTransformersPolicyBackend
from trajweave.backends.policy import PolicyRequest
from trajweave.core.specs import AgentSpec


def test_tiny_random_hf_backend_generates_response_on_available_device():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    backend = HFTransformersPolicyBackend(
        model_path="__tiny_random_gpt2__",
        device=device,
        max_new_tokens=3,
        generation_config={"do_sample": False, "pad_token_id": 0},
    )
    response = backend.generate(
        PolicyRequest(
            agent=AgentSpec(name="solver", role="solver", policy_group="tiny"),
            task_id="hf_smoke",
            observation="What is 1 + 1?",
            prompt="Task:\nWhat is 1 + 1?\n\nFinal answer:",
        )
    )

    assert response.token_ids
    assert response.metadata["policy_backend"] == "hf-transformers"
