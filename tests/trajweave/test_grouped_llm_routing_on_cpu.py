import asyncio

import pytest

from trajweave.backends.verl.llm_routing import GroupedLLMServerClient


def test_grouped_llm_client_routes_and_strips_policy_marker():
    calls = []

    class Client:
        async def generate(self, **kwargs):
            calls.append(kwargs)
            return "ok"

    client = GroupedLLMServerClient({"policy_a": Client(), "policy_b": Client()})
    result = asyncio.run(
        client.generate(
            request_id="req",
            prompt_ids=[1],
            sampling_params={"temperature": 0.1, "trajweave_policy_group": "policy_b"},
        )
    )

    assert result == "ok"
    assert calls == [{"request_id": "req", "prompt_ids": [1], "sampling_params": {"temperature": 0.1}}]


def test_grouped_llm_client_rejects_missing_or_unknown_group():
    client = GroupedLLMServerClient({"policy_a": object()})
    with pytest.raises(ValueError, match="requires sampling_params"):
        asyncio.run(client.generate(request_id="req", prompt_ids=[1], sampling_params={}))
    with pytest.raises(KeyError, match="No vLLM client"):
        asyncio.run(
            client.generate(
                request_id="req",
                prompt_ids=[1],
                sampling_params={"trajweave_policy_group": "policy_b"},
            )
        )
