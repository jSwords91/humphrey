"""OpenAI provider: settings wiring, get_provider() precedence, adapter shape.

No live network calls — the AsyncOpenAI client is monkeypatched, same as
MockProvider stands in for Azure elsewhere in this suite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _clean_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """conftest already scrubs these; re-assert per test since some cases set them."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)


def _reset_settings() -> None:
    from app.settings import reset_settings_for_tests

    reset_settings_for_tests()


def test_settings_picks_up_openai_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.settings import get_settings

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test-model")
    _reset_settings()

    settings = get_settings()
    assert settings.openai_api_configured is True
    assert settings.openai_api_key == "sk-test-123"
    assert settings.openai_model == "gpt-test-model"


def test_settings_openai_model_has_a_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.settings import get_settings

    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    _reset_settings()

    assert get_settings().openai_model  # non-empty default


def test_get_provider_mock_wins_even_with_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """conftest's HUMPHREY_MOCK=1 must beat a configured OPENAI_API_KEY."""
    from providers import get_provider
    from providers.mock import MockProvider

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    _reset_settings()

    assert isinstance(get_provider(), MockProvider)


def test_get_provider_prefers_openai_over_azure_when_both_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from providers import get_provider
    from providers.openai_provider import OpenAIProvider

    monkeypatch.setenv("HUMPHREY_MOCK", "0")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    # Explicit rather than relying on OPENAI_MODEL's default (which is a
    # reasoning model as of this codebase) - this test is about provider
    # preference, not about what that default happens to be.
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.cognitiveservices.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4-mini")
    _reset_settings()

    provider = get_provider()
    assert isinstance(provider, OpenAIProvider)
    assert provider.deployment_name == "gpt-4o-mini"


def test_get_provider_falls_back_to_azure_without_openai_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from providers import get_provider
    from providers.azure import AzureOpenAIProvider

    monkeypatch.setenv("HUMPHREY_MOCK", "0")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.cognitiveservices.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4-mini")
    _reset_settings()

    assert isinstance(get_provider(), AzureOpenAIProvider)


def test_get_provider_explains_missing_azure_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    from providers import get_provider

    monkeypatch.setenv("HUMPHREY_MOCK", "0")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.cognitiveservices.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4-mini")
    monkeypatch.setitem(sys.modules, "providers.azure", None)  # simulates the SDK being absent
    _reset_settings()

    with pytest.raises(RuntimeError, match="uv sync --group azure"):
        get_provider()


def test_openai_provider_requires_key() -> None:
    from providers.openai_provider import OpenAIProvider
    from app.settings import get_settings

    _reset_settings()
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIProvider(get_settings())


# --- streaming adapter shape, fully stubbed (no network) -------------------


@dataclass
class _FakeFunction:
    name: str | None = None
    arguments: str | None = None


@dataclass
class _FakeToolCallDelta:
    index: int
    id: str | None
    function: _FakeFunction


@dataclass
class _FakeDelta:
    content: str | None = None
    tool_calls: list[_FakeToolCallDelta] | None = None


@dataclass
class _FakeChoice:
    delta: _FakeDelta


@dataclass
class _FakeUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class _FakeChunk:
    choices: list[_FakeChoice]
    usage: Any = None


class _FakeStream:
    def __init__(self, chunks: list[_FakeChunk]) -> None:
        self._chunks = chunks

    def __aiter__(self) -> _FakeStream:
        self._iter = iter(self._chunks)
        return self

    async def __anext__(self) -> _FakeChunk:
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration from None


class _FakeCompletions:
    def __init__(self, chunks: list[_FakeChunk]) -> None:
        self._chunks = chunks
        self.received_kwargs: dict[str, Any] | None = None

    async def create(self, **kwargs: Any) -> _FakeStream:
        self.received_kwargs = kwargs
        return _FakeStream(self._chunks)


class _FakeChat:
    def __init__(self, chunks: list[_FakeChunk]) -> None:
        self.completions = _FakeCompletions(chunks)


class _FakeAsyncOpenAI:
    def __init__(self, chunks: list[_FakeChunk]) -> None:
        self.chat = _FakeChat(chunks)


@pytest.mark.asyncio
async def test_openai_stream_maps_text_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    import providers.openai_provider as provider_module
    from brain.messages import UserMessage
    from providers.openai_provider import OpenAIProvider
    from app.settings import get_settings

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    # This fake client only implements chat.completions, not responses - pin
    # to a non-reasoning model so it doesn't route through the Responses API
    # branch (OPENAI_MODEL's default is a reasoning model in this codebase).
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    _reset_settings()

    fake_client = _FakeAsyncOpenAI(
        [
            _FakeChunk(choices=[_FakeChoice(delta=_FakeDelta(content="hello"))]),
            _FakeChunk(
                choices=[],
                usage=_FakeUsage(prompt_tokens=3, completion_tokens=5, total_tokens=8),
            ),
        ]
    )
    provider_module._client.cache_clear()
    monkeypatch.setattr(provider_module, "_client", lambda api_key: fake_client)

    provider = OpenAIProvider(get_settings())
    chunks = [
        chunk
        async for chunk in provider.stream(
            [UserMessage(content="hi")], [], system="be terse", signal=None
        )
    ]

    assert chunks[0].content == "hello"
    assert chunks[1].usage == {"input": 3, "output": 5, "total_tokens": 8}
    assert fake_client.chat.completions.received_kwargs["model"] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_openai_stream_maps_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    import providers.openai_provider as provider_module
    from brain.messages import UserMessage
    from providers.openai_provider import OpenAIProvider
    from app.settings import get_settings
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    _reset_settings()

    fake_client = _FakeAsyncOpenAI(
        [
            _FakeChunk(
                choices=[
                    _FakeChoice(
                        delta=_FakeDelta(
                            tool_calls=[
                                _FakeToolCallDelta(
                                    index=0,
                                    id="call_1",
                                    function=_FakeFunction(name="status", arguments="{}"),
                                )
                            ]
                        )
                    )
                ]
            ),
        ]
    )
    provider_module._client.cache_clear()
    monkeypatch.setattr(provider_module, "_client", lambda api_key: fake_client)

    provider = OpenAIProvider(get_settings())
    chunks = [
        chunk
        async for chunk in provider.stream(
            [UserMessage(content="status?")], [], system="be terse", signal=None
        )
    ]

    assert len(chunks) == 1
    delta = chunks[0].tool_calls[0]
    assert delta.id == "call_1"
    assert delta.name == "status"
    assert delta.arguments == "{}"


# --- Responses API branch (reasoning models), fully stubbed -----------------


@dataclass
class _FakeResponsesEvent:
    """Stands in for the SDK's typed streaming event — only .model_dump() is used."""

    data: dict[str, Any]

    def model_dump(self) -> dict[str, Any]:
        return self.data


class _FakeResponsesStream:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = events

    def __aiter__(self) -> _FakeResponsesStream:
        self._iter = iter(self._events)
        return self

    async def __anext__(self) -> _FakeResponsesEvent:
        try:
            return _FakeResponsesEvent(next(self._iter))
        except StopIteration:
            raise StopAsyncIteration from None


class _FakeResponses:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = events
        self.received_kwargs: dict[str, Any] | None = None

    async def create(self, **kwargs: Any) -> _FakeResponsesStream:
        self.received_kwargs = kwargs
        return _FakeResponsesStream(self._events)


class _FakeResponsesAsyncOpenAI:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self.responses = _FakeResponses(events)


@pytest.mark.asyncio
async def test_openai_uses_responses_api_for_reasoning_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import providers.openai_provider as provider_module
    from brain.messages import UserMessage
    from providers.openai_provider import OpenAIProvider
    from app.settings import get_settings

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.6-sol")
    _reset_settings()

    fake_client = _FakeResponsesAsyncOpenAI(
        [
            {"type": "response.reasoning_summary_text.delta", "delta": "considering..."},
            {"type": "response.output_text.delta", "delta": "the answer"},
            {
                "type": "response.completed",
                "response": {
                    "usage": {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12}
                },
            },
        ]
    )
    provider_module._client.cache_clear()
    monkeypatch.setattr(provider_module, "_client", lambda api_key: fake_client)

    provider = OpenAIProvider(get_settings())
    chunks = [
        chunk
        async for chunk in provider.stream(
            [UserMessage(content="hi")], [], system="be terse", signal=None
        )
    ]

    assert chunks[0].thinking == "considering..."
    assert chunks[1].content == "the answer"
    assert chunks[2].usage == {"input": 5, "output": 7, "total_tokens": 12}
    assert fake_client.responses.received_kwargs["model"] == "gpt-5.6-sol"
    assert fake_client.responses.received_kwargs["input"] == [{"role": "user", "content": "hi"}]


@pytest.mark.asyncio
async def test_azure_v1_client_sends_a_real_bearer_token_not_a_stringified_callable() -> None:
    """Regression test for a live 401 against real Azure infra.

    ``AsyncOpenAI`` (unlike ``AsyncAzureOpenAI``) has no azure_ad_token_provider
    hook — api_key is baked into the header as a literal string at
    construction. Passing the token-provider callable there directly never
    invokes it; f-string interpolation silently stringifies the callable
    itself into the header ("Bearer <function ... at 0x...>"), which Azure
    correctly rejects with a generic "invalid subscription key" 401. The fix
    is an httpx.Auth that calls the provider fresh on every request.
    """
    import httpx

    from providers.azure import _AzureADTokenAuth

    captured: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers.get("authorization")
        return httpx.Response(200, json={"data": []})

    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url="https://example.test/openai/v1/",
        api_key="unused-real-token-injected-by-http-client-auth",
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            auth=_AzureADTokenAuth(lambda: "REAL-AAD-TOKEN"),
        ),
    )

    await client.models.list()

    assert captured["authorization"] == "Bearer REAL-AAD-TOKEN"
