"""A turn is saved before a new chat is named.

Found running Humphrey for real: naming the chat (a model call) came
first, so a failure or a reload during it lost the whole chat. The save
now comes first, and the next turn may begin as soon as it's done. That a
reload can't interrupt a turn at all is tested in test_runs.py.
"""

from __future__ import annotations

import asyncio

from app.main import finish_turn
from app.sessions import MemorySessionStore, Session
from brain.messages import AssistantMessage, UserMessage
from brain.provider import ModelChunk


class SlowTitles:
    """A title model that takes a while to answer, as a real one does."""

    def __init__(self, delay: float) -> None:
        self.delay = delay

    async def stream(self, messages, tools, *, system, signal=None):
        await asyncio.sleep(self.delay)
        yield ModelChunk(content="Counting Workspace Files")


def _new_chat() -> Session:
    return Session(
        id="newchat000000001",
        messages=[UserMessage(content="how many files?"), AssistantMessage(content="Three.", model="m")],
    )


async def test_a_normal_turn_saves_and_names() -> None:
    store = MemorySessionStore()
    await finish_turn(_new_chat(), store, SlowTitles(0), "how many files?")
    saved = await store.get("newchat000000001")
    assert saved is not None and saved.title == "Counting Workspace Files"


async def test_a_failed_title_model_still_saves_with_a_fallback_title() -> None:
    class BrokenTitles:
        async def stream(self, messages, tools, *, system, signal=None):
            raise RuntimeError("model unavailable")
            yield  # pragma: no cover

    store = MemorySessionStore()
    await finish_turn(_new_chat(), store, BrokenTitles(), "how many files?")
    saved = await store.get("newchat000000001")
    assert saved is not None
    assert saved.title == "how many files"  # generate_session_title's own fallback


async def test_the_turn_counts_as_saved_before_the_chat_is_named() -> None:
    store = MemorySessionStore()
    saved_at: list[str | None] = []

    def on_saved() -> None:
        stored = store._sessions.get("newchat000000001")
        saved_at.append(stored.title if stored else "not stored")

    await finish_turn(_new_chat(), store, SlowTitles(0.05), "how many files?", on_saved=on_saved)
    assert saved_at == [None]  # stored, not yet named
