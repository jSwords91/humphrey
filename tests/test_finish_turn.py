"""A new chat survives the browser leaving while it's being named.

Found running Humphrey for real: reloading the page a second or two after a
new chat's first reply lost the whole chat. The turn's end named the chat
(a model call) before saving it, and the reload cancelled the request in
the middle of that call, so the save never ran.
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


async def test_a_reload_while_naming_still_saves_the_chat_and_then_names_it() -> None:
    store = MemorySessionStore()
    session = _new_chat()
    turn = asyncio.create_task(finish_turn(session, store, SlowTitles(0.2), "how many files?"))
    await asyncio.sleep(0.05)  # saved, and now waiting on the title
    turn.cancel()  # the browser went away
    await asyncio.sleep(0)

    saved = await store.get("newchat000000001")
    assert saved is not None
    assert [m.text for m in saved.messages] == ["how many files?", "Three."]

    await asyncio.sleep(0.3)  # the title call carries on regardless
    saved = await store.get("newchat000000001")
    assert saved is not None and saved.title == "Counting Workspace Files"


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


async def test_leaving_during_the_first_save_still_names_the_chat() -> None:
    """The real case: writing a session can be slow (a Windows drive under
    WSL), and the reload landed during that save, so naming never began."""

    class SlowStore(MemorySessionStore):
        async def save(self, session: Session) -> None:
            await asyncio.sleep(0.1)
            await super().save(session)

    store = SlowStore()
    turn = asyncio.create_task(finish_turn(_new_chat(), store, SlowTitles(0.05), "how many files?"))
    await asyncio.sleep(0.02)  # mid-save
    turn.cancel()
    await asyncio.sleep(0.5)
    saved = await store.get("newchat000000001")
    assert saved is not None and saved.title == "Counting Workspace Files"
