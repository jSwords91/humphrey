"""Retries for passing failures, and credentials supplied per request.

Both serve the ChatGPT plan route (app/auth), where the access token
rotates hourly and the docs say which failures to retry, but neither
knows anything about sign-in: the provider just asks for a credential.
"""

from __future__ import annotations

from typing import Any

import httpx
import openai
import pytest

from brain.messages import UserMessage
from brain.provider import ModelChunk
from providers.errors import ProviderRequestError
from providers.retry import stream_with_retries

_REQUEST = httpx.Request("POST", "https://api.openai.com/v1/responses")


def _http_error(status: int, code: str) -> openai.APIStatusError:
    response = httpx.Response(status, request=_REQUEST, json={"error": {"code": code, "message": code}})
    return openai.APIStatusError(code, response=response, body={"code": code, "message": code})


class _Attempts:
    """Each call opens one attempt: chunks, then optionally an error."""

    def __init__(self, *attempts: tuple[list[str], Exception | None]) -> None:
        self._attempts = list(attempts)
        self.calls = 0

    def __call__(self):
        chunks, error = self._attempts[min(self.calls, len(self._attempts) - 1)]
        self.calls += 1

        async def run():
            for text in chunks:
                yield ModelChunk(content=text)
            if error is not None:
                raise error

        return run()


async def _collect(attempts: _Attempts, *, retry_http: bool = False, signal: Any = None):
    slept: list[float] = []

    async def sleep(seconds: float) -> None:
        slept.append(seconds)

    chunks = [
        c.content
        async for c in stream_with_retries(attempts, signal=signal, retry_http=retry_http, sleep=sleep)
    ]
    return chunks, slept


async def test_a_passing_failure_before_any_reply_is_retried() -> None:
    attempts = _Attempts(
        ([], ProviderRequestError("cut short", code="stream_incomplete")),
        (["Hello"], None),
    )
    chunks, slept = await _collect(attempts)
    assert chunks == ["Hello"]
    assert attempts.calls == 2
    assert slept == [1.0]


async def test_nothing_is_retried_once_part_of_a_reply_was_streamed() -> None:
    attempts = _Attempts((["Hel"], ProviderRequestError("cut short", code="stream_incomplete")))
    with pytest.raises(ProviderRequestError):
        await _collect(attempts)
    assert attempts.calls == 1


async def test_a_lasting_failure_is_not_retried() -> None:
    attempts = _Attempts(([], ProviderRequestError("bad input", code="invalid_request_error", status=400)))
    with pytest.raises(ProviderRequestError):
        await _collect(attempts, retry_http=True)
    assert attempts.calls == 1


async def test_retries_stop_after_two() -> None:
    attempts = _Attempts(([], openai.APIError("busy", _REQUEST, body={"code": "server_is_overloaded"})))
    with pytest.raises(ProviderRequestError) as caught:
        await _collect(attempts)
    assert attempts.calls == 3
    assert caught.value.code == "server_is_overloaded"
    assert isinstance(caught.value.__cause__, openai.APIError)


async def test_a_usage_limit_is_never_retried() -> None:
    attempts = _Attempts(([], _http_error(429, "subscription_sharing_usage_limit_exceeded")))
    with pytest.raises(ProviderRequestError):
        await _collect(attempts, retry_http=True)
    assert attempts.calls == 1


@pytest.mark.parametrize(("retry_http", "calls"), [(False, 1), (True, 3)])
async def test_http_failures_are_retried_only_when_the_sdk_does_not(retry_http: bool, calls: int) -> None:
    attempts = _Attempts(([], _http_error(503, "subscription_sharing_usage_unavailable")))
    with pytest.raises(ProviderRequestError):
        await _collect(attempts, retry_http=retry_http)
    assert attempts.calls == calls


async def test_a_cancelled_run_is_not_retried() -> None:
    class Cancelled:
        def is_cancelled(self) -> bool:
            return True

    attempts = _Attempts(([], ProviderRequestError("cut short", code="stream_incomplete")))
    with pytest.raises(ProviderRequestError):
        await _collect(attempts, signal=Cancelled())
    assert attempts.calls == 1


# --- credentials and Responses-only routing ----------------------------------


class _Completed:
    def __init__(self) -> None:
        self.received: dict[str, Any] | None = None

    async def create(self, **kwargs: Any):
        self.received = kwargs

        class Event:
            def model_dump(self) -> dict[str, Any]:
                return {"type": "response.completed", "response": {}}

        async def events():
            yield Event()

        return events()


class _Client:
    def __init__(self) -> None:
        self.responses = _Completed()


async def test_a_credential_is_fetched_for_every_request(monkeypatch: pytest.MonkeyPatch) -> None:
    import providers.openai_provider as provider_module
    from app.settings import get_settings

    built: list[tuple[str, int]] = []
    client = _Client()

    def fake_client(api_key: str, max_retries: int = 2) -> _Client:
        built.append((api_key, max_retries))
        return client

    monkeypatch.setattr(provider_module, "_client", fake_client)
    tokens = iter(["token-one", "token-two"])

    async def credential() -> str:
        return next(tokens)

    provider = provider_module.OpenAIProvider(
        get_settings(), model="gpt-4o-mini", credential=credential, responses_only=True
    )
    for _ in range(2):
        async for _chunk in provider.stream([UserMessage(content="hi")], [], system="s"):
            pass

    # A fresh credential per request, SDK retries off, and the Responses API
    # even for a model that would otherwise use Chat Completions.
    assert built == [("token-one", 0), ("token-two", 0)]
    assert client.responses.received is not None
    assert client.responses.received["model"] == "gpt-4o-mini"


async def test_the_api_key_route_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    import providers.openai_provider as provider_module
    from app.settings import get_settings, reset_settings_for_tests

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    reset_settings_for_tests()
    built: list[tuple[str, int]] = []

    def fake_client(api_key: str, max_retries: int = 2) -> _Client:
        built.append((api_key, max_retries))
        return _Client()

    monkeypatch.setattr(provider_module, "_client", fake_client)
    provider = provider_module.OpenAIProvider(get_settings(), model="gpt-5.6-sol")
    async for _chunk in provider.stream([UserMessage(content="hi")], [], system="s"):
        pass
    assert built == [("sk-test-not-real", 2)]


def test_without_a_credential_an_api_key_is_still_required(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.settings import get_settings, reset_settings_for_tests
    from providers.openai_provider import OpenAIProvider

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    reset_settings_for_tests()
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIProvider(get_settings())

    async def credential() -> str:
        return "token"

    OpenAIProvider(get_settings(), credential=credential)  # no key needed with a credential
