from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class GroupedLLMServerClient:
    """Route native VERL generation requests to an explicit policy-group client."""

    def __init__(self, clients: Mapping[str, Any]):
        normalized = {str(group_id): client for group_id, client in clients.items()}
        if not normalized or any(not group_id for group_id in normalized):
            raise ValueError("Grouped LLM client requires non-empty policy-group bindings.")
        self.clients = normalized

    async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        params = dict(sampling_params)
        group_id = params.pop("trajweave_policy_group", None)
        if not group_id:
            raise ValueError("Grouped vLLM generation requires sampling_params.trajweave_policy_group.")
        client = self.clients.get(str(group_id))
        if client is None:
            raise KeyError(
                f"No vLLM client is bound to policy group {group_id!r}; known groups: {tuple(self.clients)}."
            )
        return await client.generate(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=params,
            **kwargs,
        )
