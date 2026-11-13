"""One error type for a failed model request, whatever shape it arrived in.

The OpenAI API reports a failure in several shapes, and each used to reach
the user differently, often as an SDK repr such as ``Error code: 400 -
{'error': {...}}``, or as a generic fallback that dropped the real reason:

  - an HTTP error before the stream opens (``openai.APIStatusError``), whose
    body is a standard ``error`` object or a bare ``{"detail": "..."}``
  - an SSE event whose data carries an ``error`` object, which the SDK
    raises as ``openai.APIError`` before our parser sees it
  - a ``response.failed`` event, or an ``error`` event with top-level
    fields, which reach ResponsesStreamParser

All of them become a ProviderRequestError: ``str()`` is text a person can
act on (the brain loop shows it as the failed turn's error), and ``code``,
``status`` and ``request_id`` stay available for diagnosis. Nothing here
ever includes a request body or credentials.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import openai


class ProviderRequestError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status: int | None = None,
        request_id: str | None = None,
    ) -> None:
        self.message = message
        self.code = code
        self.status = status
        self.request_id = request_id
        super().__init__(self.__str__())

    def __str__(self) -> str:
        details = [
            part
            for part in (
                self.code,
                f"HTTP {self.status}" if self.status else None,
                f"request {self.request_id}" if self.request_id else None,
            )
            if part
        ]
        return f"{self.message} ({', '.join(details)})" if details else self.message


def error_from_body(
    body: Any, *, status: int | None = None, request_id: str | None = None
) -> ProviderRequestError:
    """Build a ProviderRequestError from an error object, a ``{"detail": ...}``
    body, or an event carrying either (top level or under ``error``)."""
    if isinstance(body, Mapping) and isinstance(body.get("error"), Mapping):
        body = body["error"]
    message: str | None = None
    code: str | None = None
    param: str | None = None
    if isinstance(body, Mapping):
        message = _text(body.get("message")) or _text(body.get("detail"))
        code = _text(body.get("code")) or _text(body.get("type"))
        param = _text(body.get("param"))
    if code in PLAN_MESSAGES:
        message = PLAN_MESSAGES[code] + (f" (Unsupported: {param}.)" if param else "")
    return ProviderRequestError(
        message or "The model request failed.", code=code, status=status, request_id=request_id
    )


# What a ChatGPT plan's error codes mean for the person using it, per OpenAI's
# Sign in with ChatGPT docs ("Errors and recovery"). The code is kept on the
# error, so the app can still act on it (for example, offer to sign in).
PLAN_MESSAGES = {
    "subscription_sharing_usage_limit_exceeded": (
        "You've reached the usage limit for using your ChatGPT plan here. Review your "
        "plan or this app's limit at https://chatgpt.com/settings/usage."
    ),
    "subscription_sharing_usage_unavailable": (
        "ChatGPT plan usage is unavailable right now. Try again in a little while."
    ),
    "subscription_sharing_user_unavailable": (
        "Your ChatGPT account is unavailable right now. Try again in a little while."
    ),
    "subscription_sharing_user_not_eligible": (
        "This ChatGPT account can't use its plan in other apps. Use an API key instead."
    ),
    "subscription_sharing_invalid_user": "Your ChatGPT sign-in is no longer valid. Sign in again.",
    "subscription_sharing_unsupported_capability": (
        "Part of this request isn't available with a ChatGPT plan."
    ),
    "subscription_sharing_route_not_supported": (
        "This kind of request isn't available with a ChatGPT plan."
    ),
}


def from_openai_error(exc: openai.APIError) -> ProviderRequestError:
    """Convert an SDK error, keeping its code and request id but not its repr."""
    if isinstance(exc, openai.APIStatusError):
        return error_from_body(exc.body, status=exc.status_code, request_id=exc.request_id)
    if isinstance(exc, openai.APIConnectionError):
        return ProviderRequestError("Couldn't reach the model provider.", code="connection_error")
    return error_from_body(exc.body if exc.body is not None else {"message": exc.message})


def _text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None
