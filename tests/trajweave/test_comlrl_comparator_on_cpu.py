import io
import json
import pickle
from urllib import error as urlerror

import pytest

from trajweave.orchestration.comlrl.comparator import (
    FrozenPolicySnapshot,
    PolicyComparator,
    RemoteComparator,
    TaggedCentralizedComparatorAdapter,
    canonical_comparator_policy,
    canonical_generation_mode,
)


class FakePolicy:
    def __init__(self, label: str) -> None:
        self.label = label
        self.calls = []

    def generate(self, prompt, *, agent_index, num_candidates, context=None):
        call = len(self.calls)
        self.calls.append((prompt, agent_index, num_candidates, context))
        return [f"{self.label}:{agent_index}:{call}:{index}" for index in range(num_candidates)]


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self.body = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.body


class ScriptedPolicy(FakePolicy):
    def __init__(self, outputs: list[str]) -> None:
        super().__init__("scripted")
        self.outputs = outputs

    def generate(self, prompt, *, agent_index, num_candidates, context=None):
        self.calls.append((prompt, agent_index, num_candidates, context))
        return self.outputs[:num_candidates]


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("self", "current"),
        ("online_copy", "current_copy"),
        ("reference", "model"),
        ("checkpoint", "history"),
        ("endpoint", "api"),
    ],
)
def test_comparator_policy_aliases_match_upstream(alias, canonical):
    assert canonical_comparator_policy(alias) == canonical


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [("multi-agent", "decentralized"), ("decentralised", "decentralized"), ("single_agent", "centralized")],
)
def test_generation_mode_aliases_match_upstream(alias, canonical):
    assert canonical_generation_mode(alias) == canonical


def test_current_resamples_independently_and_decentralized_calls_each_agent():
    policies = [FakePolicy("agent-0"), FakePolicy("agent-1")]
    comparator = PolicyComparator(policy="current", current_policy=policies, num_agents=2)

    first = comparator.generate(["p0", "p1"], num_candidates=2)
    second = comparator.generate(["p0", "p1"], num_candidates=2)

    assert first.candidates_by_agent != second.candidates_by_agent
    assert [len(policy.calls) for policy in policies] == [2, 2]
    assert first.candidates_by_agent[0] == ("agent-0:0:0:0", "agent-0:0:0:1")


def test_current_copy_requires_distinct_frozen_policy_object():
    live = FakePolicy("live")
    with pytest.raises(ValueError, match="must not reuse a live"):
        PolicyComparator(
            policy="copy",
            current_policy=live,
            current_copy=FrozenPolicySnapshot(live, "copy-0"),
            num_agents=1,
        )

    frozen = FakePolicy("frozen")
    comparator = PolicyComparator(
        policy="current_copy",
        current_policy=live,
        current_copy=FrozenPolicySnapshot(frozen, "copy-0", path="snapshots/copy-0"),
        num_agents=1,
    )
    result = comparator.generate(["prompt"], num_candidates=1)
    assert result.candidates_by_agent == (("frozen:0:0:0",),)
    assert not live.calls
    assert result.provenance["snapshot_path"] == "snapshots/copy-0"


def test_history_resolves_snapshot_for_requested_iteration_and_reports_path():
    live = FakePolicy("live")
    requested = []

    def resolve(iteration):
        requested.append(iteration)
        return FrozenPolicySnapshot(
            FakePolicy(f"history-{iteration}"), f"iteration-{iteration}", f"iteration_{iteration:04d}"
        )

    comparator = PolicyComparator(
        policy="previous_iteration",
        current_policy=live,
        history_resolver=resolve,
        num_agents=1,
    )
    result = comparator.generate(["prompt"], num_candidates=1, iteration=3)

    assert requested == [3]
    assert result.candidates_by_agent == (("history-3:0:0:0",),)
    assert result.provenance["snapshot_path"] == "iteration_0003"


def test_model_uses_frozen_provider_and_centralized_adapter_parser_are_generic():
    live = [FakePolicy("live-0"), FakePolicy("live-1")]
    with pytest.raises(ValueError, match="must not reuse a live"):
        PolicyComparator(
            policy="model",
            current_policy=live,
            model=FrozenPolicySnapshot(live, "invalid-model"),
            num_agents=2,
        )
    frozen = [FakePolicy("coordinator"), FakePolicy("unused")]

    def adapter(prompts, context):
        return f"JOIN::{prompts[0]}::{prompts[1]}::{context['task']}"

    def parser(text):
        return f"left::{text}", f"right::{text}"

    comparator = PolicyComparator(
        policy="model",
        current_policy=live,
        model=FrozenPolicySnapshot(frozen, "model-v1"),
        num_agents=2,
        generation_mode="centralized",
        centralized_prompt_adapter=adapter,
        centralized_response_parser=parser,
    )
    result = comparator.generate(["alpha", "beta"], num_candidates=2, context={"task": "math"})

    assert frozen[0].calls[0][0] == "JOIN::alpha::beta::math"
    assert not frozen[1].calls
    assert result.candidates_by_agent[0][0].startswith("left::coordinator")
    assert result.candidates_by_agent[1][1].startswith("right::coordinator")


def test_centralized_supports_any_agent_count_with_default_tagged_adapter():
    policy = ScriptedPolicy(
        [
            "<agent_0>alpha-0</agent_0><agent_1>beta-0</agent_1><agent_2>gamma-0</agent_2>",
            "<AGENT_0>alpha-1</AGENT_0><AGENT_1>beta-1</AGENT_1><AGENT_2>gamma-1</AGENT_2>",
        ]
    )
    comparator = PolicyComparator(
        policy="current",
        current_policy=policy,
        num_agents=3,
        generation_mode="centralized",
        centralized_agent_index=1,
    )

    result = comparator.generate(["p0", "p1", "p2"], num_candidates=2)

    assert policy.calls[0][1] == 1
    assert "Agent 2 original prompt:\np2" in policy.calls[0][0]
    assert "<agent_2>" in policy.calls[0][0]
    assert result.candidates_by_agent == (
        ("alpha-0", "alpha-1"),
        ("beta-0", "beta-1"),
        ("gamma-0", "gamma-1"),
    )


def test_tagged_adapter_and_centralized_validation_reject_missing_agent_output():
    adapter = TaggedCentralizedComparatorAdapter(num_agents=3)
    assert adapter.parse_completion("<agent_0>a</agent_0><agent_1>b</agent_1>") == ("a", "b", "")
    with pytest.raises(ValueError, match="indexed agent sections"):
        adapter.parse_completion("no tags")

    comparator = PolicyComparator(
        policy="current",
        current_policy=FakePolicy("live"),
        num_agents=3,
        generation_mode="centralized",
        centralized_prompt_adapter=lambda prompts, context: "joined",
        centralized_response_parser=lambda text: ("valid", "", "third"),
    )
    with pytest.raises(ValueError, match="3 non-empty"):
        comparator.generate(["a", "b", "c"], num_candidates=1)


def test_remote_generic_payload_env_key_timeout_and_dotted_response(monkeypatch):
    calls = []

    def opener(request, *, timeout):
        calls.append((request, timeout))
        return FakeResponse({"data": {"outputs": [{"text": "one"}, {"content": "two"}]}})

    monkeypatch.setenv("COMPARATOR_TEST_KEY", "super-secret")
    comparator = RemoteComparator(
        url="https://comparator.example/v1/generate",
        num_agents=1,
        timeout=3.5,
        api_key_env="COMPARATOR_TEST_KEY",
        api_key_header="X-API-Key",
        api_key_prefix="Token",
        response_field="data.outputs",
        max_new_tokens=64,
        temperature=0.7,
        extra_body={"custom": True},
        opener=opener,
    )
    result = comparator.generate(["prompt"], num_candidates=2, context={"task_id": "t1"})

    request, timeout = calls[0]
    payload = json.loads(request.data)
    assert timeout == 3.5
    assert request.get_header("X-api-key") == "Token super-secret"
    assert payload == {
        "prompt": "prompt",
        "agent_idx": 0,
        "num_return_sequences": 2,
        "max_new_tokens": 64,
        "temperature": 0.7,
        "batch_item": {"task_id": "t1"},
        "custom": True,
    }
    assert result.candidates_by_agent == (("one", "two"),)
    assert "super-secret" not in repr(comparator)
    assert "super-secret" not in json.dumps(comparator.to_dict())


def test_remote_pickle_restores_default_opener_without_serializing_secret(monkeypatch):
    monkeypatch.setenv("PICKLE_COMPARATOR_KEY", "secret")
    comparator = RemoteComparator(
        url="https://api.example/generate",
        num_agents=1,
        api_key_env="PICKLE_COMPARATOR_KEY",
    )

    restored = pickle.loads(pickle.dumps(comparator))

    assert callable(restored._opener)
    assert restored._api_key is None
    assert "secret" not in json.dumps(restored.to_dict())


def test_remote_rejects_inline_credentials_and_secret_headers():
    with pytest.raises(ValueError, match="api_key_env"):
        RemoteComparator(url="https://api.example/generate", num_agents=1, api_key="inline-secret")
    with pytest.raises(ValueError, match="credential headers"):
        RemoteComparator(
            url="https://api.example/generate",
            num_agents=1,
            headers={"Authorization": "inline-secret"},
        )


def test_remote_openai_payload_batches_and_never_uses_implicit_url():
    payloads = []

    def opener(request, *, timeout):
        payload = json.loads(request.data)
        payloads.append(payload)
        return FakeResponse(
            {"choices": [{"message": {"content": f"answer-{len(payloads)}-{index}"}} for index in range(payload["n"])]}
        )

    comparator = RemoteComparator(
        url="https://api.example/chat",
        num_agents=1,
        api_format="openai_chat",
        model="test-model",
        max_candidates_per_request=1,
        opener=opener,
    )
    result = comparator.generate(["question"], num_candidates=2)

    assert [payload["n"] for payload in payloads] == [1, 1]
    assert payloads[0]["messages"] == [{"role": "user", "content": "question"}]
    assert result.candidates_by_agent == (("answer-1-0", "answer-2-0"),)
    with pytest.raises(ValueError, match="explicit comparator API URL"):
        RemoteComparator(url="", num_agents=1)


def test_remote_centralized_supports_three_agents_with_tagged_adapter():
    payloads = []

    def opener(request, *, timeout):
        del timeout
        payloads.append(json.loads(request.data))
        return FakeResponse({"completions": ["<agent_0>one</agent_0><agent_1>two</agent_1><agent_2>three</agent_2>"]})

    comparator = RemoteComparator(
        url="https://api.example/generate",
        num_agents=3,
        generation_mode="centralized",
        centralized_agent_index=2,
        opener=opener,
    )
    result = comparator.generate(["p0", "p1", "p2"], num_candidates=1)

    assert payloads[0]["agent_idx"] == 2
    assert "Agent 2 original prompt:\np2" in payloads[0]["prompt"]
    assert result.candidates_by_agent == (("one",), ("two",), ("three",))


def test_remote_anthropic_messages_uses_one_request_per_candidate_and_env_key(monkeypatch):
    calls = []

    def opener(request, *, timeout):
        del timeout
        calls.append(request)
        return FakeResponse({"content": [{"type": "text", "text": f"answer-{len(calls)}"}, {"type": "tool_use"}]})

    monkeypatch.setenv("ANTHROPIC_TEST_KEY", "anthropic-secret")
    comparator = RemoteComparator(
        url="https://api.anthropic.example/v1/messages",
        num_agents=1,
        api_format="anthropic_messages",
        model="claude-test",
        api_key_env="ANTHROPIC_TEST_KEY",
        max_new_tokens=128,
        temperature=0.4,
        opener=opener,
    )
    result = comparator.generate(["question"], num_candidates=2)

    assert len(calls) == 2
    assert calls[0].get_header("X-api-key") == "anthropic-secret"
    assert calls[0].get_header("Anthropic-version") == "2023-06-01"
    assert json.loads(calls[0].data) == {
        "model": "claude-test",
        "max_tokens": 128,
        "messages": [{"role": "user", "content": "question"}],
        "temperature": 0.4,
    }
    assert result.candidates_by_agent == (("answer-1", "answer-2"),)
    assert result.provenance["api_format"] == "anthropic"


def test_remote_openai_responses_uses_native_payload_and_extracts_output_text():
    payloads = []

    def opener(request, *, timeout):
        del timeout
        payloads.append(json.loads(request.data))
        return FakeResponse(
            {
                "output": [
                    {
                        "content": [
                            {"type": "reasoning", "text": "hidden"},
                            {"type": "output_text", "text": f"response-{len(payloads)}"},
                        ]
                    }
                ]
            }
        )

    comparator = RemoteComparator(
        url="https://api.openai.example/v1/responses",
        num_agents=1,
        api_format="codex",
        model="gpt-test",
        max_new_tokens=96,
        max_candidates_per_request=10,
        opener=opener,
    )
    result = comparator.generate(["question"], num_candidates=2)

    assert payloads == [
        {"model": "gpt-test", "input": "question", "max_output_tokens": 96},
        {"model": "gpt-test", "input": "question", "max_output_tokens": 96},
    ]
    assert result.candidates_by_agent == (("response-1", "response-2"),)
    assert result.provenance["api_format"] == "openai_responses"


@pytest.mark.parametrize("timeout", [0, -1])
def test_remote_rejects_non_positive_timeout(timeout):
    with pytest.raises(ValueError, match="timeout must be positive"):
        RemoteComparator(url="https://api.example/generate", num_agents=1, timeout=timeout)


def test_remote_candidate_count_http_json_and_empty_response_errors():
    short = RemoteComparator(
        url="https://api.example/generate",
        num_agents=1,
        opener=lambda request, timeout: FakeResponse({"completions": ["only-one"]}),
    )
    with pytest.raises(ValueError, match="expected exactly 2"):
        short.generate(["prompt"], num_candidates=2)

    def http_error(_request, *, timeout):
        raise urlerror.HTTPError("https://api.example", 503, "unavailable", {}, io.BytesIO(b"try later"))

    failing = RemoteComparator(url="https://api.example/generate", num_agents=1, opener=http_error)
    with pytest.raises(RuntimeError, match="HTTP 503: try later"):
        failing.generate(["prompt"], num_candidates=1)

    class RawResponse(FakeResponse):
        def __init__(self, body):
            self.body = body

    invalid_json = RemoteComparator(
        url="https://api.example/generate",
        num_agents=1,
        opener=lambda request, timeout: RawResponse(b"not-json"),
    )
    with pytest.raises(ValueError, match="invalid JSON"):
        invalid_json.generate(["prompt"], num_candidates=1)

    empty = RemoteComparator(
        url="https://api.example/generate",
        num_agents=1,
        opener=lambda request, timeout: RawResponse(b""),
    )
    with pytest.raises(ValueError, match="empty response body"):
        empty.generate(["prompt"], num_candidates=1)
