from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

from trajweave.storage.serialization import REDACTED, redact_secrets, redact_text

_POLICY_ALIASES = {
    "current": "current",
    "self": "current",
    "online": "current",
    "current_copy": "current_copy",
    "copy": "current_copy",
    "online_copy": "current_copy",
    "current-offload": "current_copy",
    "current_offload": "current_copy",
    "model": "model",
    "external": "model",
    "comparator": "model",
    "reference": "model",
    "ref": "model",
    "history": "history",
    "checkpoint": "history",
    "previous": "history",
    "previous_iteration": "history",
    "api": "api",
    "http": "api",
    "endpoint": "api",
}
_GENERATION_MODE_ALIASES = {
    "decentralized": "decentralized",
    "decentralised": "decentralized",
    "multi_agent": "decentralized",
    "multi-agent": "decentralized",
    "centralized": "centralized",
    "centralised": "centralized",
    "single_agent": "centralized",
    "single-agent": "centralized",
}


def canonical_comparator_policy(policy: str | None) -> str:
    value = str(policy or "current").strip().lower()
    try:
        return _POLICY_ALIASES[value]
    except KeyError as exc:
        raise ValueError("comparator policy must be one of: current, current_copy, model, history, api") from exc


def canonical_generation_mode(mode: str | None) -> str:
    value = str(mode or "decentralized").strip().lower()
    try:
        return _GENERATION_MODE_ALIASES[value]
    except KeyError as exc:
        raise ValueError("generation mode must be one of: decentralized, centralized") from exc


@runtime_checkable
class PolicyProvider(Protocol):
    def generate(
        self,
        prompt: str,
        *,
        agent_index: int,
        num_candidates: int,
        context: Mapping[str, Any] | None = None,
    ) -> Sequence[str]: ...


@dataclass(frozen=True)
class FrozenPolicySnapshot:
    """A lifecycle-owned frozen policy and its provenance.

    This wrapper is deliberately required for current-copy, model, and history
    policies so callers cannot silently substitute the mutable live policy.
    """

    provider: PolicyProvider | Sequence[PolicyProvider] = field(repr=False)
    snapshot_id: str
    path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.snapshot_id:
            raise ValueError("snapshot_id must be non-empty")


@dataclass(frozen=True)
class ComparatorResult:
    prompts: tuple[str, ...]
    candidates_by_agent: tuple[tuple[str, ...], ...]
    policy: str
    generation_mode: str
    iteration: int | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def num_agents(self) -> int:
        return len(self.candidates_by_agent)

    @property
    def num_candidates(self) -> int:
        return len(self.candidates_by_agent[0]) if self.candidates_by_agent else 0


class Comparator(Protocol):
    def generate(
        self,
        prompts: Sequence[str],
        *,
        num_candidates: int,
        iteration: int | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> ComparatorResult: ...


CentralizedPromptAdapter = Callable[[Sequence[str], Mapping[str, Any] | None], str]
CentralizedResponseParser = Callable[[str], Sequence[str]]
HistoryPolicyResolver = Callable[[int], FrozenPolicySnapshot]


@dataclass(frozen=True)
class TaggedCentralizedComparatorAdapter:
    """Domain-neutral centralized protocol used by current CoMLRL main."""

    num_agents: int

    def __post_init__(self) -> None:
        _positive_int(self.num_agents, "num_agents")

    def build_prompt(
        self,
        prompts: Sequence[str],
        context: Mapping[str, Any] | None = None,
    ) -> str:
        del context
        normalized_prompts = _validate_prompts(prompts, self.num_agents)
        prompt_sections = "\n\n".join(
            f"Agent {agent_index} original prompt:\n{prompt}" for agent_index, prompt in enumerate(normalized_prompts)
        )
        output_sections = "\n".join(
            f"<agent_{agent_index}>\n...\n</agent_{agent_index}>" for agent_index in range(self.num_agents)
        )
        return (
            "You are a centralized coordinator producing the separate outputs of "
            f"{self.num_agents} agents.\n\n"
            "Each output must satisfy its corresponding original prompt. Keep the outputs separate;\n"
            "do not merge them or add commentary outside the required tags.\n\n"
            f"{prompt_sections}\n\n"
            "Return exactly one section for each agent in this format:\n"
            f"{output_sections}\n"
        )

    def parse_completion(self, completion: str) -> tuple[str, ...]:
        outputs: list[str] = []
        found = False
        for agent_index in range(self.num_agents):
            pattern = rf"<\s*agent_{agent_index}\s*>(.*?)<\s*/\s*agent_{agent_index}\s*>"
            match = re.search(pattern, str(completion), flags=re.IGNORECASE | re.DOTALL)
            found = found or match is not None
            outputs.append(match.group(1).strip() if match is not None else "")
        if not found:
            raise ValueError("centralized completion did not contain any indexed agent sections")
        return tuple(outputs)


class PolicyComparator:
    def __init__(
        self,
        *,
        policy: str,
        current_policy: PolicyProvider | Sequence[PolicyProvider],
        num_agents: int,
        generation_mode: str = "decentralized",
        current_copy: FrozenPolicySnapshot | None = None,
        model: FrozenPolicySnapshot | None = None,
        history_resolver: HistoryPolicyResolver | None = None,
        centralized_prompt_adapter: CentralizedPromptAdapter | None = None,
        centralized_response_parser: CentralizedResponseParser | None = None,
        centralized_agent_index: int = 0,
    ) -> None:
        self.policy = canonical_comparator_policy(policy)
        if self.policy == "api":
            raise ValueError("use RemoteComparator for comparator policy 'api'")
        self.generation_mode = canonical_generation_mode(generation_mode)
        self.num_agents = _positive_int(num_agents, "num_agents")
        self.current_policy = current_policy
        self.current_copy = current_copy
        self.model = model
        self.history_resolver = history_resolver
        self.centralized_agent_index = _non_negative_int(centralized_agent_index, "centralized_agent_index")

        if self.centralized_agent_index >= self.num_agents:
            raise ValueError("centralized_agent_index must be smaller than num_agents")
        if self.generation_mode == "centralized":
            centralized_prompt_adapter, centralized_response_parser = _resolve_centralized_callbacks(
                self.num_agents,
                centralized_prompt_adapter,
                centralized_response_parser,
            )
        self.centralized_prompt_adapter = centralized_prompt_adapter
        self.centralized_response_parser = centralized_response_parser
        if self.policy == "current_copy":
            if current_copy is None:
                raise ValueError("current_copy requires a FrozenPolicySnapshot")
            _assert_snapshot_is_not_live(current_copy, current_policy)
        elif self.policy == "model":
            if model is None:
                raise ValueError("model comparator requires a FrozenPolicySnapshot")
            _assert_snapshot_is_not_live(model, current_policy)
        elif self.policy == "history" and history_resolver is None:
            raise ValueError("history comparator requires a history_resolver")

    def generate(
        self,
        prompts: Sequence[str],
        *,
        num_candidates: int,
        iteration: int | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> ComparatorResult:
        normalized_prompts = _validate_prompts(prompts, self.num_agents)
        candidate_count = _positive_int(num_candidates, "num_candidates")
        snapshot = self._resolve_snapshot(iteration)
        provider = self.current_policy if snapshot is None else snapshot.provider

        if self.generation_mode == "decentralized":
            candidates = tuple(
                _generate_candidates(
                    _provider_for_agent(provider, agent_index, self.num_agents),
                    prompt,
                    agent_index=agent_index,
                    num_candidates=candidate_count,
                    context=context,
                )
                for agent_index, prompt in enumerate(normalized_prompts)
            )
        else:
            assert self.centralized_prompt_adapter is not None
            assert self.centralized_response_parser is not None
            centralized_prompt = self.centralized_prompt_adapter(normalized_prompts, context)
            if not isinstance(centralized_prompt, str) or not centralized_prompt.strip():
                raise ValueError("centralized prompt adapter returned an empty prompt")
            centralized_provider = _provider_for_agent(provider, self.centralized_agent_index, self.num_agents)
            raw_candidates = _generate_candidates(
                centralized_provider,
                centralized_prompt,
                agent_index=self.centralized_agent_index,
                num_candidates=candidate_count,
                context=context,
            )
            split: list[list[str]] = [[] for _ in range(self.num_agents)]
            for candidate_index, raw_candidate in enumerate(raw_candidates):
                parsed = tuple(self.centralized_response_parser(raw_candidate))
                if len(parsed) != self.num_agents or any(
                    not isinstance(item, str) or not item.strip() for item in parsed
                ):
                    raise ValueError(
                        f"centralized response parser must return {self.num_agents} non-empty agent outputs "
                        f"for candidate {candidate_index}"
                    )
                for agent_index, item in enumerate(parsed):
                    split[agent_index].append(item)
            candidates = tuple(tuple(items) for items in split)

        provenance: dict[str, Any] = {"policy": self.policy}
        if snapshot is not None:
            provenance.update(
                {
                    "snapshot_id": snapshot.snapshot_id,
                    "snapshot_path": snapshot.path,
                    "snapshot_metadata": dict(snapshot.metadata),
                }
            )
        return ComparatorResult(
            prompts=normalized_prompts,
            candidates_by_agent=candidates,
            policy=self.policy,
            generation_mode=self.generation_mode,
            iteration=iteration,
            provenance=provenance,
        )

    def _resolve_snapshot(self, iteration: int | None) -> FrozenPolicySnapshot | None:
        if self.policy == "current":
            return None
        if self.policy == "current_copy":
            assert self.current_copy is not None
            return self.current_copy
        if self.policy == "model":
            assert self.model is not None
            return self.model
        if iteration is None or isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
            raise ValueError("history comparator requires a non-negative iteration")
        assert self.history_resolver is not None
        snapshot = self.history_resolver(iteration)
        if not isinstance(snapshot, FrozenPolicySnapshot):
            raise TypeError("history_resolver must return FrozenPolicySnapshot")
        _assert_snapshot_is_not_live(snapshot, self.current_policy)
        return snapshot


class RemoteComparator:
    def __init__(
        self,
        *,
        url: str,
        num_agents: int,
        api_format: str = "generic",
        generation_mode: str = "decentralized",
        model: str | None = None,
        timeout: float = 120.0,
        headers: Mapping[str, str] | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
        api_key_header: str = "Authorization",
        api_key_prefix: str = "Bearer",
        response_field: str = "completions",
        extra_body: Mapping[str, Any] | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        max_candidates_per_request: int | None = None,
        centralized_prompt_adapter: CentralizedPromptAdapter | None = None,
        centralized_response_parser: CentralizedResponseParser | None = None,
        centralized_agent_index: int = 0,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        self.url = _validate_explicit_url(url)
        self.num_agents = _positive_int(num_agents, "num_agents")
        self.api_format = _canonical_api_format(api_format)
        self.generation_mode = canonical_generation_mode(generation_mode)
        if isinstance(timeout, bool) or not isinstance(timeout, int | float) or timeout <= 0:
            raise ValueError("timeout must be positive")
        self.timeout = float(timeout)
        self.model = model
        self.headers = _validated_public_headers(headers)
        if api_key is not None:
            raise ValueError("Comparator credentials must be configured through api_key_env, not api_key")
        self._api_key = None
        self.api_key_env = str(api_key_env) if api_key_env is not None else None
        if self.api_key_env is not None and not self.api_key_env.strip():
            raise ValueError("api_key_env must be non-empty when configured")
        self.api_key_header = str(api_key_header)
        self.api_key_prefix = str(api_key_prefix)
        self.response_field = str(response_field)
        self.extra_body = dict(extra_body or {})
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        if max_candidates_per_request is None and "deepseek" in str(model or "").lower():
            max_candidates_per_request = 1
        self.max_candidates_per_request = (
            _positive_int(max_candidates_per_request, "max_candidates_per_request")
            if max_candidates_per_request is not None
            else None
        )
        self.centralized_agent_index = _non_negative_int(centralized_agent_index, "centralized_agent_index")
        self._opener = opener or urlrequest.urlopen

        if self.centralized_agent_index >= self.num_agents:
            raise ValueError("centralized_agent_index must be smaller than num_agents")
        if not self.api_key_header:
            raise ValueError("api_key_header must be non-empty")
        if not self.response_field and self.api_format == "generic":
            raise ValueError("response_field must be non-empty for generic API responses")
        if self.generation_mode == "centralized":
            centralized_prompt_adapter, centralized_response_parser = _resolve_centralized_callbacks(
                self.num_agents,
                centralized_prompt_adapter,
                centralized_response_parser,
            )
        self.centralized_prompt_adapter = centralized_prompt_adapter
        self.centralized_response_parser = centralized_response_parser

    def __repr__(self) -> str:
        return (
            f"RemoteComparator(url={_safe_url(self.url)!r}, num_agents={self.num_agents!r}, "
            f"api_format={self.api_format!r}, generation_mode={self.generation_mode!r}, "
            f"model={self.model!r}, timeout={self.timeout!r}, api_key={REDACTED!r})"
        )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_api_key"] = None
        state["url"] = _safe_url(state["url"])
        state["headers"] = redact_secrets(state["headers"])
        state["_opener"] = None
        return state

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self.__dict__.update(state)
        self._opener = urlrequest.urlopen

    def to_dict(self) -> dict[str, Any]:
        return redact_secrets(
            {
                "url": _safe_url(self.url),
                "num_agents": self.num_agents,
                "api_format": self.api_format,
                "generation_mode": self.generation_mode,
                "centralized_agent_index": self.centralized_agent_index,
                "model": self.model,
                "timeout": self.timeout,
                "headers": self.headers,
                "api_key": self._api_key,
                "api_key_env": self.api_key_env,
                "api_key_header": self.api_key_header,
                "api_key_prefix": self.api_key_prefix,
                "response_field": self.response_field,
                "extra_body": self.extra_body,
            }
        )

    def generate(
        self,
        prompts: Sequence[str],
        *,
        num_candidates: int,
        iteration: int | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> ComparatorResult:
        normalized_prompts = _validate_prompts(prompts, self.num_agents)
        candidate_count = _positive_int(num_candidates, "num_candidates")
        if self.generation_mode == "decentralized":
            candidates = tuple(
                self._request_candidates(
                    prompt,
                    agent_index=agent_index,
                    num_candidates=candidate_count,
                    context=context,
                )
                for agent_index, prompt in enumerate(normalized_prompts)
            )
        else:
            assert self.centralized_prompt_adapter is not None
            assert self.centralized_response_parser is not None
            centralized_prompt = self.centralized_prompt_adapter(normalized_prompts, context)
            if not isinstance(centralized_prompt, str) or not centralized_prompt.strip():
                raise ValueError("centralized prompt adapter returned an empty prompt")
            raw_candidates = self._request_candidates(
                centralized_prompt,
                agent_index=self.centralized_agent_index,
                num_candidates=candidate_count,
                context=context,
            )
            split: list[list[str]] = [[] for _ in range(self.num_agents)]
            for candidate_index, raw_candidate in enumerate(raw_candidates):
                parsed = tuple(self.centralized_response_parser(raw_candidate))
                if len(parsed) != self.num_agents or any(
                    not isinstance(item, str) or not item.strip() for item in parsed
                ):
                    raise ValueError(
                        f"centralized response parser must return {self.num_agents} non-empty agent outputs "
                        f"for candidate {candidate_index}"
                    )
                for agent_index, item in enumerate(parsed):
                    split[agent_index].append(item)
            candidates = tuple(tuple(items) for items in split)
        return ComparatorResult(
            prompts=normalized_prompts,
            candidates_by_agent=candidates,
            policy="api",
            generation_mode=self.generation_mode,
            iteration=iteration,
            provenance={"api_format": self.api_format, "model": self.model, "url": _safe_url(self.url)},
        )

    def _request_candidates(
        self,
        prompt: str,
        *,
        agent_index: int,
        num_candidates: int,
        context: Mapping[str, Any] | None,
    ) -> tuple[str, ...]:
        if self.api_format == "generic":
            payload = self._generic_payload(prompt, agent_index, num_candidates, context)
            candidates = self._extract_candidates(self._send(payload))
            return _strict_candidates(candidates, num_candidates, source="Comparator API")

        if self.api_format in {"anthropic", "openai_responses"}:
            output: list[str] = []
            while len(output) < num_candidates:
                if self.api_format == "anthropic":
                    payload = self._anthropic_payload(prompt)
                    candidates = self._extract_anthropic_candidates(self._send(payload))
                    source = "Anthropic comparator API"
                else:
                    payload = self._openai_responses_payload(prompt)
                    candidates = self._extract_openai_responses_candidates(self._send(payload))
                    source = "OpenAI Responses comparator API"
                output.extend(_strict_candidates(candidates[:1], 1, source=source))
            return tuple(output)

        max_per_request = self.max_candidates_per_request or num_candidates
        output: list[str] = []
        while len(output) < num_candidates:
            request_count = min(max_per_request, num_candidates - len(output))
            payload = self._openai_payload(prompt, request_count)
            candidates = self._extract_openai_candidates(self._send(payload))
            output.extend(_strict_candidates(candidates, request_count, source="OpenAI comparator API"))
        return tuple(output)

    def _generic_payload(
        self,
        prompt: str,
        agent_index: int,
        num_candidates: int,
        context: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "prompt": prompt,
            "agent_idx": agent_index,
            "num_return_sequences": num_candidates,
        }
        optional = {
            "model": self.model,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "batch_item": dict(context) if context is not None else None,
        }
        payload.update({key: value for key, value in optional.items() if value is not None})
        payload.update(self.extra_body)
        return payload

    def _openai_payload(self, prompt: str, num_candidates: int) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "n": num_candidates,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        payload = {key: value for key, value in payload.items() if value is not None}
        payload.update(self.extra_body)
        return payload

    def _anthropic_payload(self, prompt: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_new_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        payload = {key: value for key, value in payload.items() if value is not None}
        payload.update(self.extra_body)
        return payload

    def _openai_responses_payload(self, prompt: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": prompt,
            "max_output_tokens": self.max_new_tokens,
        }
        payload = {key: value for key, value in payload.items() if value is not None}
        payload.update(self.extra_body)
        return payload

    def _request_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.headers}
        if self.api_format == "anthropic":
            headers.setdefault("anthropic-version", "2023-06-01")
        api_key = os.environ.get(self.api_key_env) if self.api_key_env else None
        if api_key:
            key_header = self.api_key_header
            prefix = self.api_key_prefix.strip()
            if self.api_format == "anthropic":
                if key_header.lower() == "authorization":
                    key_header = "x-api-key"
                if prefix.lower() == "bearer":
                    prefix = ""
            headers[key_header] = f"{prefix} {api_key}" if prefix else api_key
        return headers

    def _send(self, payload: Mapping[str, Any]) -> Any:
        request = urlrequest.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._request_headers(),
            method="POST",
        )
        try:
            with self._opener(request, timeout=self.timeout) as response:
                raw = response.read()
        except urlerror.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            message = redact_text(detail or str(exc.reason))
            raise RuntimeError(f"Comparator API HTTP {exc.code}: {message}") from exc
        except (urlerror.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"Comparator API request failed: {redact_text(str(exc))}") from exc
        if not raw:
            raise ValueError("Comparator API returned an empty response body")
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            preview = redact_text(raw[:200].decode("utf-8", errors="replace"))
            raise ValueError(f"Comparator API returned invalid JSON: {preview}") from exc

    def _extract_candidates(self, response: Any) -> list[str]:
        value = _get_dotted(response, self.response_field)
        return _normalize_candidate_items(value)

    @staticmethod
    def _extract_openai_candidates(response: Any) -> list[str]:
        if not isinstance(response, Mapping) or not isinstance(response.get("choices"), list):
            return []
        output: list[str] = []
        for choice in response["choices"]:
            if not isinstance(choice, Mapping):
                continue
            message = choice.get("message")
            if isinstance(message, Mapping) and message.get("content") is not None:
                output.append(str(message["content"]))
            elif choice.get("text") is not None:
                output.append(str(choice["text"]))
        return output

    @staticmethod
    def _extract_anthropic_candidates(response: Any) -> list[str]:
        if not isinstance(response, Mapping) or not isinstance(response.get("content"), list):
            return []
        return [
            str(block["text"])
            for block in response["content"]
            if isinstance(block, Mapping) and block.get("type") == "text" and block.get("text") is not None
        ]

    @staticmethod
    def _extract_openai_responses_candidates(response: Any) -> list[str]:
        if not isinstance(response, Mapping) or not isinstance(response.get("output"), list):
            return []
        output: list[str] = []
        for item in response["output"]:
            if not isinstance(item, Mapping) or not isinstance(item.get("content"), list):
                continue
            for block in item["content"]:
                if (
                    isinstance(block, Mapping)
                    and block.get("type") in {"output_text", "text"}
                    and block.get("text") is not None
                ):
                    output.append(str(block["text"]))
        return output


def _canonical_api_format(value: str) -> str:
    normalized = str(value or "generic").strip().lower()
    aliases = {
        "generic": "generic",
        "openai": "openai",
        "openai_chat": "openai",
        "chat": "openai",
        "anthropic": "anthropic",
        "anthropic_messages": "anthropic",
        "messages": "anthropic",
        "openai_responses": "openai_responses",
        "responses": "openai_responses",
        "codex": "openai_responses",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError("api_format must be one of: generic, openai, anthropic, openai_responses") from exc


def _validated_public_headers(headers: Mapping[str, str] | None) -> dict[str, str]:
    normalized = {str(key): str(value) for key, value in dict(headers or {}).items()}
    secret_headers = {
        "authorization",
        "proxy-authorization",
        "x-api-key",
        "api-key",
    }
    unsafe = sorted(key for key in normalized if key.strip().lower() in secret_headers)
    if unsafe:
        raise ValueError(f"Comparator credential headers must be populated from api_key_env; remove headers {unsafe}")
    return normalized


def _resolve_centralized_callbacks(
    num_agents: int,
    prompt_adapter: CentralizedPromptAdapter | None,
    response_parser: CentralizedResponseParser | None,
) -> tuple[CentralizedPromptAdapter, CentralizedResponseParser]:
    if prompt_adapter is None and response_parser is None:
        adapter = TaggedCentralizedComparatorAdapter(num_agents=num_agents)
        return adapter.build_prompt, adapter.parse_completion
    if prompt_adapter is None or response_parser is None:
        raise ValueError(
            "centralized generation requires both centralized_prompt_adapter and "
            "centralized_response_parser, or neither to use the tagged adapter"
        )
    return prompt_adapter, response_parser


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _validate_prompts(prompts: Sequence[str], num_agents: int) -> tuple[str, ...]:
    if isinstance(prompts, (str, bytes)):
        raise TypeError("prompts must be a sequence of strings")
    normalized = tuple(prompts)
    if len(normalized) != num_agents:
        raise ValueError(f"expected {num_agents} prompts, got {len(normalized)}")
    if any(not isinstance(prompt, str) or not prompt for prompt in normalized):
        raise ValueError("prompts must contain non-empty strings")
    return normalized


def _provider_for_agent(
    provider: PolicyProvider | Sequence[PolicyProvider],
    agent_index: int,
    num_agents: int,
) -> PolicyProvider:
    if isinstance(provider, Sequence) and not isinstance(provider, (str, bytes)):
        if len(provider) != num_agents:
            raise ValueError(f"policy provider sequence must contain {num_agents} agents")
        selected = provider[agent_index]
    else:
        selected = provider
    if not isinstance(selected, PolicyProvider):
        raise TypeError("policy providers must implement generate()")
    return selected


def _generate_candidates(
    provider: PolicyProvider,
    prompt: str,
    *,
    agent_index: int,
    num_candidates: int,
    context: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    candidates = provider.generate(
        prompt,
        agent_index=agent_index,
        num_candidates=num_candidates,
        context=context,
    )
    return _strict_candidates(candidates, num_candidates, source="policy provider")


def _strict_candidates(candidates: Sequence[str], expected: int, *, source: str) -> tuple[str, ...]:
    if isinstance(candidates, (str, bytes)):
        candidates = [str(candidates)]
    normalized = tuple(candidates)
    if len(normalized) != expected:
        raise ValueError(f"{source} returned {len(normalized)} candidates; expected exactly {expected}")
    if any(not isinstance(candidate, str) or not candidate.strip() for candidate in normalized):
        raise ValueError(f"{source} returned an empty candidate")
    return normalized


def _provider_identities(provider: PolicyProvider | Sequence[PolicyProvider]) -> set[int]:
    if isinstance(provider, Sequence) and not isinstance(provider, (str, bytes)):
        return {id(item) for item in provider}
    return {id(provider)}


def _assert_snapshot_is_not_live(
    snapshot: FrozenPolicySnapshot,
    current_policy: PolicyProvider | Sequence[PolicyProvider],
) -> None:
    if _provider_identities(snapshot.provider).intersection(_provider_identities(current_policy)):
        raise ValueError("frozen comparator snapshot must not reuse a live current-policy object")


def _validate_explicit_url(url: str) -> str:
    if not isinstance(url, str) or not url.strip():
        raise ValueError("an explicit comparator API URL is required")
    normalized = url.strip()
    parsed = urlparse.urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("comparator API URL must be an explicit http(s) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("comparator API URL must not contain userinfo credentials")
    return normalized


def _safe_url(url: str) -> str:
    parsed = urlparse.urlsplit(url)
    return urlparse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _get_dotted(value: Any, field: str) -> Any:
    current = value
    for segment in field.split("."):
        if isinstance(current, Mapping):
            if segment not in current:
                return None
            current = current[segment]
        elif isinstance(current, list) and segment.isdigit():
            index = int(segment)
            if index >= len(current):
                return None
            current = current[index]
        else:
            return None
    return current


def _normalize_candidate_items(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return [str(value)]
    output: list[str] = []
    for item in value:
        if isinstance(item, str):
            output.append(item)
        elif isinstance(item, Mapping):
            text = item.get("text") if item.get("text") is not None else item.get("content")
            if text is not None:
                output.append(str(text))
        elif item is not None:
            output.append(str(item))
    return output


__all__ = [
    "Comparator",
    "ComparatorResult",
    "FrozenPolicySnapshot",
    "PolicyComparator",
    "PolicyProvider",
    "RemoteComparator",
    "TaggedCentralizedComparatorAdapter",
    "canonical_comparator_policy",
    "canonical_generation_mode",
]
