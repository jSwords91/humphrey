"""Model request failures reach the user as readable text with their code.

They used to arrive as an SDK repr ("Error code: 400 - {'error': {...}}"),
as a generic fallback that dropped a nested error's message, or not at all:
a stream that stopped early passed for a complete reply. The error bodies
below are the shapes the OpenAI API returned live.
"""

from __future__ import annotations

from typing import Any

import httpx
import openai
import pytest

from brain.messages import UserMessage
from providers.errors import ProviderRequestError, error_from_body, from_openai_error
from providers.openai_responses import ResponsesStreamParser, read_responses_stream

_REQUEST = httpx.Request("POST", "https://api.openai.com/v1/responses")


def _status_error(cls: type[openai.APIStatusError], status: int, body: Any) -> openai.APIStatusError:
    response = httpx.Response(status, request=_REQUEST, headers={"x-request-id": "req_123"}, json=body)
    # The SDK unwraps {"error": {...}} before building the exception; mirror that.
    inner = body.get("error", body) if isinstance(body, dict) else body
    return cls(f"Error code: {status} - {body}", response=response, body=inner)


# --- error shapes -------------------------------------------------------------


def test_a_standard_error_object_keeps_message_code_status_and_request_id() -> None:
    exc = _status_error(
        openai.AuthenticationError,
        401,
        {"error": {"message": "Unauthorized", "type": "rejected_by_access_enforcement", "code": "no_matching_rule"}},
    )
    error = from_openai_error(exc)
    assert (error.message, error.code, error.status, error.request_id) == (
        "Unauthorized",
        "no_matching_rule",
        401,
        "req_123",
    )
    assert str(error) == "Unauthorized (no_matching_rule, HTTP 401, request req_123)"
    assert "Error code" not in str(error)


def test_a_detail_body_becomes_the_message() -> None:
    exc = _status_error(openai.BadRequestError, 400, {"detail": "Store must be set to false"})
    error = from_openai_error(exc)
    assert error.message == "Store must be set to false"
    assert error.status == 400


def test_an_error_raised_mid_stream_keeps_its_code() -> None:
    exc = openai.APIError(
        "You've reached this app's usage limit.",
        _REQUEST,
        body={"code": "subscription_sharing_usage_limit_exceeded", "message": "You've reached this app's usage limit."},
    )
    error = from_openai_error(exc)
    assert error.code == "subscription_sharing_usage_limit_exceeded"
    # Plan codes get text a person can act on, with where to go next.
    assert "https://chatgpt.com/settings/usage" in error.message


def test_a_connection_failure_says_so() -> None:
    error = from_openai_error(openai.APIConnectionError(request=_REQUEST))
    assert error.code == "connection_error"
    assert "Couldn't reach" in str(error)


@pytest.mark.parametrize(
    "body",
    [
        {"type": "error", "code": "server_is_overloaded", "message": "Our servers are overloaded."},
        {"type": "error", "error": {"code": "server_is_overloaded", "message": "Our servers are overloaded."}},
    ],
)
def test_an_error_event_is_read_at_the_top_level_or_nested(body: dict) -> None:
    error = error_from_body(body)
    assert (error.code, error.message) == ("server_is_overloaded", "Our servers are overloaded.")


def test_an_empty_body_still_says_something() -> None:
    assert str(error_from_body(None)) == "The model request failed."


# --- the parser and stream reader --------------------------------------------


def test_response_failed_raises_with_its_code() -> None:
    parser = ResponsesStreamParser()
    with pytest.raises(ProviderRequestError) as caught:
        parser.feed(
            {
                "type": "response.failed",
                "response": {"error": {"code": "subscription_sharing_usage_limit_exceeded", "message": "Limit reached."}},
            }
        )
    assert caught.value.code == "subscription_sharing_usage_limit_exceeded"
    assert parser.finished


@pytest.mark.parametrize("terminal", ["response.completed", "response.incomplete"])
def test_a_terminal_event_finishes_the_parser(terminal: str) -> None:
    parser = ResponsesStreamParser()
    assert not parser.finished
    parser.feed({"type": terminal, "response": {}})
    assert parser.finished


class _Event:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def model_dump(self) -> dict[str, Any]:
        return self._data


async def _events(*events: dict[str, Any]):
    for event in events:
        yield _Event(event)


class _Cancelled:
    def is_cancelled(self) -> bool:
        return True


async def test_a_stream_that_stops_early_is_an_error() -> None:
    reader = read_responses_stream(
        _events({"type": "response.output_text.delta", "delta": "Half a reply"}), ResponsesStreamParser(), None
    )
    with pytest.raises(ProviderRequestError) as caught:
        async for _ in reader:
            pass
    assert caught.value.code == "stream_incomplete"


async def test_a_completed_stream_is_not_an_error() -> None:
    chunks = [
        chunk
        async for chunk in read_responses_stream(
            _events(
                {"type": "response.output_text.delta", "delta": "Done."},
                {"type": "response.completed", "response": {"usage": {"input_tokens": 1, "output_tokens": 2}}},
            ),
            ResponsesStreamParser(),
            None,
        )
    ]
    assert chunks[0].content == "Done."


async def test_a_cancelled_stream_is_not_an_error() -> None:
    chunks = [
        chunk
        async for chunk in read_responses_stream(
            _events({"type": "response.output_text.delta", "delta": "x"}), ResponsesStreamParser(), _Cancelled()
        )
    ]
    assert chunks == []


# --- end to end through the provider and the brain loop ----------------------


class _FailingResponses:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def create(self, **_: Any) -> Any:
        raise self._exc


class _FailingClient:
    def __init__(self, exc: Exception) -> None:
        self.responses = _FailingResponses(exc)


async def test_the_user_sees_the_reason_not_an_sdk_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    import providers.openai_provider as provider_module
    from app.settings import get_settings, reset_settings_for_tests
    from brain.harness import AgentHarness, AgentHarnessConfig
    from brain.messages import AssistantMessage

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.6-sol")
    monkeypatch.setenv("HUMPHREY_MOCK", "0")
    reset_settings_for_tests()
    exc = _status_error(openai.BadRequestError, 400, {"detail": "Store must be set to false"})
    monkeypatch.setattr(provider_module, "_client", lambda api_key, max_retries=2: _FailingClient(exc))

    harness = AgentHarness(
        AgentHarnessConfig(provider=provider_module.OpenAIProvider(get_settings()), system="s")
    )
    async for _ in harness.prompt_message(UserMessage(content="hi")):
        pass
    [failed] = [m for m in harness.messages if isinstance(m, AssistantMessage)]
    assert failed.stop_reason == "error"
    assert failed.error_message == "Store must be set to false (HTTP 400, request req_123)"


async def test_the_azure_provider_converts_errors_the_same_way(monkeypatch: pytest.MonkeyPatch) -> None:
    import providers.azure as azure_module
    from app.settings import get_settings, reset_settings_for_tests

    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.cognitiveservices.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.6-sol")
    reset_settings_for_tests()
    exc = _status_error(
        openai.RateLimitError, 429, {"error": {"message": "Slow down.", "code": "rate_limit_exceeded"}}
    )
    monkeypatch.setattr(azure_module, "_v1_client", lambda endpoint: _FailingClient(exc))

    provider = azure_module.AzureOpenAIProvider(get_settings())
    with pytest.raises(ProviderRequestError) as caught:
        async for _ in provider.stream([UserMessage(content="hi")], [], system="s"):
            pass
    assert str(caught.value) == "Slow down. (rate_limit_exceeded, HTTP 429, request req_123)"
