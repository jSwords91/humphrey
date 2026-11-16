"""Asking the person something mid-run, the way tau's extensions do.

Mirrors tau_coding's ``UiBridge``: a tool awaits a dialog on the bridge,
and whichever frontend is attached answers it. tau's terminal UI pushes a
modal and resolves an ``asyncio.Future`` from its callback; here the
browser plays the modal. ``WebUiBridge`` sends one ``ui_request`` frame
down the run's SSE stream and waits for ``POST /api/sessions/{id}/ui/{request_id}``
to resolve it. With no frontend attached, ``NullUiBridge`` answers nothing,
as tau's does.

A bridge lives for one run. ``close()`` resolves anything still waiting
with no answer, so a stopped or disconnected run never hangs on a dialog.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from typing import Any, Protocol

Answer = dict[str, Any]


class UiBridge(Protocol):
    @property
    def has_ui(self) -> bool:
        """Whether a person can answer dialogs on this run."""
        ...

    async def ask_questions(self, call_id: str, questions: list[dict[str, Any]]) -> Answer | None:
        """Show questions and wait. None means no answer: no UI, or the run stopped."""
        ...


class NullUiBridge:
    """No frontend attached (tests, headless use): every dialog goes unanswered."""

    @property
    def has_ui(self) -> bool:
        return False

    async def ask_questions(self, call_id: str, questions: list[dict[str, Any]]) -> Answer | None:
        return None


class WebUiBridge:
    """The browser answers, over the run's SSE stream and a POST."""

    def __init__(self, send: Callable[[dict[str, Any]], None]) -> None:
        self._send = send
        self._pending: dict[str, asyncio.Future[Answer | None]] = {}
        self._closed = False

    @property
    def has_ui(self) -> bool:
        return not self._closed

    @property
    def waiting(self) -> bool:
        return bool(self._pending)

    async def ask_questions(self, call_id: str, questions: list[dict[str, Any]]) -> Answer | None:
        if self._closed:
            return None
        request_id = uuid.uuid4().hex
        future: asyncio.Future[Answer | None] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        self._send(
            {
                "type": "ui_request",
                "kind": "question",
                "request_id": request_id,
                "call_id": call_id,
                "questions": questions,
            }
        )
        try:
            return await future
        finally:
            self._pending.pop(request_id, None)

    def resolve(self, request_id: str, answer: Answer) -> bool:
        """Answer a waiting request. False if it isn't waiting (unknown or done)."""
        future = self._pending.get(request_id)
        if future is None or future.done():
            return False
        future.set_result(answer)
        return True

    def close(self) -> None:
        """End the bridge: anything still waiting gets no answer."""
        self._closed = True
        for future in self._pending.values():
            if not future.done():
                future.set_result(None)
