"""Runs: turns the server owns, whoever is watching.

A turn used to last only as long as the request that started it, so a
reload, another chat opened, or a laptop asleep stopped it. Now each run
keeps a log of the frames it sends, and a browser follows that log from
the start: it can leave and come back, and two tabs can watch one run.
Only ``stop()`` ends a run early.

The registry here is how the routes find a session's run: to follow it,
steer it, answer its questions or stop it.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Coroutine
from typing import Any

from app.ui_bridge import WebUiBridge
from brain.harness import AgentHarness


class FrameLog:
    """Every frame a run sends, in order, for any number of followers."""

    def __init__(self) -> None:
        self._frames: list[str] = []
        self._finished = False
        # Replaced on every publish, so a follower waits on the one it saw.
        self._more = asyncio.Event()

    def publish(self, frame: str) -> None:
        if self._finished:
            return
        self._frames.append(frame)
        self._wake()

    def finish(self) -> None:
        self._finished = True
        self._wake()

    async def follow(self, *, idle: float) -> AsyncIterator[str | None]:
        """Every frame from the first, then live ones until the log ends.
        Yields None after ``idle`` quiet seconds, so the caller can send a
        keepalive."""
        sent = 0
        while True:
            while sent < len(self._frames):
                yield self._frames[sent]
                sent += 1
            if self._finished:
                return
            more = self._more
            try:
                await asyncio.wait_for(more.wait(), idle)
            except TimeoutError:
                yield None

    def _wake(self) -> None:
        self._more.set()
        self._more = asyncio.Event()


class Run:
    """One turn of one session: its harness, its question bridge, its log."""

    def __init__(
        self,
        session_id: str,
        *,
        harness: AgentHarness,
        bridge: WebUiBridge,
        log: FrameLog,
        prompt: str,
        started_ms: int,
        new_chat: bool,
    ) -> None:
        self.session_id = session_id
        self.harness = harness
        self.bridge = bridge
        self.log = log
        # What the person typed, and when: a browser that opens the chat
        # mid-turn shows the prompt itself, and leaves out of the stored
        # history anything from this turn, since the log replays it.
        self.prompt = prompt
        self.started_ms = started_ms
        # Nothing is stored for a new chat until this turn is, so until
        # then the run is what lists and opens it.
        self.new_chat = new_chat
        self.stopped = False
        self._saved = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    @property
    def saved(self) -> bool:
        """Whether this turn is in the stored session, so the next may start."""
        return self._saved.is_set()

    def mark_saved(self) -> None:
        self._saved.set()

    async def wait_saved(self) -> None:
        await self._saved.wait()

    def stop(self) -> None:
        """End the run early: the model call, any tool, and any question."""
        self.stopped = True
        self.harness.cancel()
        self.bridge.close()


_runs: dict[str, Run] = {}


def start(run: Run, work: Coroutine[Any, Any, None]) -> None:
    """Register ``run`` and do its ``work`` in a task of its own. The run
    stays findable until the work is done (its turn saved, a new chat
    named), then its log ends."""

    async def main() -> None:
        try:
            await work
        finally:
            run.mark_saved()  # even if the work failed: never block the next turn
            run.log.finish()
            if _runs.get(run.session_id) is run:
                del _runs[run.session_id]

    _runs[run.session_id] = run
    # The run holds its task: asyncio keeps only weak references to tasks.
    run._task = asyncio.create_task(main())


def get(session_id: str) -> Run | None:
    return _runs.get(session_id)


def current() -> list[Run]:
    return list(_runs.values())


def clear_for_tests() -> None:
    _runs.clear()
