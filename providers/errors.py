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
    if isinstance(body, Mapping):
        message = _text(body.get("message")) or _text(body.get("detail"))
        code = _text(body.get("code")) or _text(body.get("type"))
    return ProviderRequestError(
        message or "The model request failed.", code=code, status=status, request_id=request_id
    )


def from_openai_error(exc: openai.APIError) -> ProviderRequestError:
    """Convert an SDK error, keeping its code and request id but not its repr."""
    if isinstance(exc, openai.APIStatusError):
        return error_from_body(exc.body, status=exc.status_code, request_id=exc.request_id)
    if isinstance(exc, openai.APIConnectionError):
        return ProviderRequestError("Couldn't reach the model provider.", code="connection_error")
    return error_from_body(exc.body if exc.body is not None else {"message": exc.message})


def _text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None
