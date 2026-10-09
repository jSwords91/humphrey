"""Session storage: ObjectStoreSessionStore's not-found/error distinction, and
FallbackSessionStore's demotion to memory on a real storage failure.

Regression coverage for a critical bug: the store's get() used to catch every
exception (including auth failures) and return None, indistinguishable from
"this session doesn't exist yet" - so once storage broke, every turn of an
ongoing conversation looked like a brand new session, even mid-chat. That
contract lives in ObjectStoreSessionStore now, on top of any ObjectStore
(Azure Blob or local disk - see app/storage.py), so a fake ObjectStore is
enough to cover it without touching either real backend.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.sessions import FallbackSessionStore, MemorySessionStore, ObjectStoreSessionStore, Session
from app.storage import ObjectMeta


class _FakeObjectStore:
    def __init__(self, behavior: str, payload: bytes = b"") -> None:
        self._behavior = behavior
        self._payload = payload

    def get_bytes(self, key: str) -> bytes | None:
        if self._behavior == "not_found":
            return None
        if self._behavior == "auth_error":
            raise PermissionError("This request is not authorized to perform this operation.")
        if self._behavior == "corrupt":
            return b"not json"
        return self._payload

    def put_bytes(self, key: str, data: bytes, *, content_type: str = "") -> None:
        pass

    def list_keys(self, prefix: str = "") -> list[ObjectMeta]:
        return []


def _session_store(behavior: str, payload: bytes = b"") -> ObjectStoreSessionStore:
    settings = type("S", (), {"sessions_prefix": "sessions"})()
    return ObjectStoreSessionStore(_FakeObjectStore(behavior, payload), settings)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_object_store_session_get_returns_none_for_genuinely_new_session() -> None:
    store = _session_store("not_found")
    assert await store.get("abcdef1234567890") is None


@pytest.mark.asyncio
async def test_object_store_session_get_returns_none_for_corrupt_object_but_logs() -> None:
    store = _session_store("corrupt")
    assert await store.get("abcdef1234567890") is None


@pytest.mark.asyncio
async def test_object_store_session_get_propagates_a_real_storage_error() -> None:
    """The actual regression: an auth/network/disk failure must NOT be
    silently swallowed as "session not found" - the caller needs to be able
    to tell the difference so it doesn't quietly start the conversation over.
    """
    store = _session_store("auth_error")
    with pytest.raises(PermissionError):
        await store.get("abcdef1234567890")


@pytest.mark.asyncio
async def test_fallback_store_uses_primary_when_healthy() -> None:
    primary = MemorySessionStore()
    fallback = MemorySessionStore()
    store = FallbackSessionStore(primary, fallback)

    session = Session(id="abcdef1234567890")
    await store.save(session)

    assert await primary.get("abcdef1234567890") is not None
    assert await fallback.get("abcdef1234567890") is None


@pytest.mark.asyncio
async def test_fallback_store_demotes_and_preserves_conversation_continuity() -> None:
    """The exact bug scenario: a real storage error on every call must not
    make each turn look like a new session. After demotion, save-then-get
    with the same id must round-trip correctly via the fallback store.
    """

    class _AlwaysBroken:
        async def get(self, session_id: str) -> Session | None:
            raise PermissionError("not authorized")

        async def save(self, session: Session) -> None:
            raise PermissionError("not authorized")

        async def list_recent(self, *, limit: int = 20) -> list:
            raise PermissionError("not authorized")

    fallback = MemorySessionStore()
    store = FallbackSessionStore(_AlwaysBroken(), fallback)

    session = Session(id="abcdef1234567890")
    session.messages = ["turn one"]  # type: ignore[list-item]
    await store.save(session)  # primary raises -> demotes -> saved via fallback

    retrieved = await store.get("abcdef1234567890")
    assert retrieved is not None
    assert retrieved.messages == ["turn one"]


@pytest.mark.asyncio
async def test_fallback_store_only_logs_demotion_once() -> None:
    calls = {"count": 0}

    class _BrokenOnce:
        async def get(self, session_id: str) -> Session | None:
            calls["count"] += 1
            raise PermissionError("not authorized")

        async def save(self, session: Session) -> None:
            raise PermissionError("not authorized")

        async def list_recent(self, *, limit: int = 20) -> list:
            raise PermissionError("not authorized")

    store = FallbackSessionStore(_BrokenOnce(), MemorySessionStore())
    await store.get("abcdef1234567890")
    await store.get("abcdef1234567890")
    # Second call must not hit the (already demoted) primary again.
    assert calls["count"] == 1


class _RecordingObjectStore:
    """In-memory ObjectStore double for exercising the real save/get/list_recent
    round trip through ObjectStoreSessionStore, independent of Azure vs local."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def get_bytes(self, key: str) -> bytes | None:
        return self.objects.get(key)

    def put_bytes(self, key: str, data: bytes, *, content_type: str = "") -> None:
        self.objects[key] = data

    def list_keys(self, prefix: str = "") -> list[ObjectMeta]:
        return [ObjectMeta(key=k, last_modified=None) for k in self.objects if k.startswith(prefix)]


@pytest.mark.asyncio
async def test_object_store_session_round_trips_save_get_list() -> None:
    settings = type("S", (), {"sessions_prefix": "sessions"})()
    backing = _RecordingObjectStore()
    store = ObjectStoreSessionStore(backing, settings)  # type: ignore[arg-type]

    session = Session(id="abcdef1234567890", title="My chat")
    await store.save(session)

    assert "sessions/abcdef1234567890.json" in backing.objects
    fetched = await store.get("abcdef1234567890")
    assert fetched is not None
    assert fetched.title == "My chat"

    recent = await store.list_recent()
    assert [s.id for s in recent] == ["abcdef1234567890"]
    assert recent[0].title == "My chat"
