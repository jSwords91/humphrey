"""Retry a model request that failed for a passing reason, before any reply.

Only failures that say "try again" are retried: an overloaded or briefly
unavailable service, a dropped connection, a stream cut short. Everything
else (bad input, auth, a usage limit) fails at once, because repeating it
would only fail again or, for a usage limit, waste the user's allowance.

A retry only happens before the first chunk is passed on: once part of a
reply has been streamed to the user, starting again would repeat it.

The OpenAI SDK already retries HTTP 429 and 5xx responses itself, so for a
client that keeps those built-in retries, only failures the SDK never sees
(errors inside the stream) are retried here. A client built with
``max_retries=0`` (the ChatGPT plan route) passes ``retry_http=True`` and
relies on this alone.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence

import openai

from brain.cancel import CancellationToken
from brain.provider import ModelChunk
from providers.errors import ProviderRequestError, from_openai_error

# Passing failures, by code. Includes the ChatGPT plan's "temporarily
# unavailable" codes, which its docs say to retry with bounded backoff.
TRANSIENT_CODES = frozenset(
    {
        "connection_error",
        "stream_incomplete",
        "server_is_overloaded",
        "server_error",
        "subscription_sharing_usage_unavailable",
        "subscription_sharing_user_unavailable",
    }
)
TRANSIENT_STATUSES = frozenset({500, 502, 503, 504})
DELAYS_SECONDS = (1.0, 3.0)  # two retries, then the error stands


def is_transient(error: ProviderRequestError, *, retry_http: bool) -> bool:
    if error.status is not None and not retry_http:
        return False  # the SDK has already retried HTTP failures
    return error.code in TRANSIENT_CODES or error.status in TRANSIENT_STATUSES


async def stream_with_retries(
    open_stream: Callable[[], AsyncIterator[ModelChunk]],
    *,
    signal: CancellationToken | None,
    retry_http: bool,
    delays: Sequence[float] = DELAYS_SECONDS,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> AsyncIterator[ModelChunk]:
    attempt = 0
    while True:
        started = False
        try:
            async for chunk in open_stream():
                started = True
                yield chunk
            return
        except openai.APIError as exc:
            error = from_openai_error(exc)
            cause: BaseException = exc
        except ProviderRequestError as exc:
            error, cause = exc, exc
        cancelled = signal is not None and signal.is_cancelled()
        give_up = started or cancelled or attempt >= len(delays)
        if give_up or not is_transient(error, retry_http=retry_http):
            if cause is error:
                raise error
            raise error from cause
        await sleep(delays[attempt])
        attempt += 1
