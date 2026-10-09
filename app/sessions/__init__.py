"""Session persistence — memory locally, or Azure Blob / local disk when
STORAGE_BACKEND selects one (see app/storage.py)."""

from __future__ import annotations

import json
import logging
import re
import string
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

from app.settings import Settings, get_settings
from app.storage import ObjectStore, get_object_store
from brain.messages import (
    AgentMessage,
    UserMessage,
    message_from_dict,
    message_to_dict,
    ui_messages,
)
from brain.provider import ModelProvider

logger = logging.getLogger(__name__)

SAFE_SESSION_ID = re.compile(r"^[a-zA-Z0-9_-]{8,64}$")
SESSION_SCHEMA_VERSION = 3


def new_session_id() -> str:
    return uuid.uuid4().hex


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class ChatMessage:
    """UI-facing user/assistant text (projection of the full transcript)."""

    role: str
    content: str
    ts: int | None = None


@dataclass(slots=True)
class Session:
    id: str
    messages: list[AgentMessage] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    title: str | None = None
    # Files the agent created or changed (app/artifacts.py), in order, each
    # with created_ms so the UI can place it back in the transcript.
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    schema_version: int = SESSION_SCHEMA_VERSION

    def ui_chat_messages(self) -> list[ChatMessage]:
        return [
            ChatMessage(role=row["role"], content=row["content"], ts=row.get("ts"))
            for row in ui_messages(self.messages)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "schema_version": self.schema_version,
            "messages": [message_to_dict(m) for m in self.messages],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "title": self.title,
            "artifacts": self.artifacts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Session:
        version = int(data.get("schema_version") or 1)
        raw_messages = data.get("messages", [])
        messages: list[AgentMessage] = []
        if isinstance(raw_messages, list):
            for item in raw_messages:
                if not isinstance(item, dict):
                    continue
                if version >= 2 or item.get("role") in {
                    "user",
                    "assistant",
                    "tool_result",
                    "tool",
                }:
                    # v1 was user/assistant strings; v2+ is full AgentMessage JSON.
                    # Assistant v1 rows are plain {"role","content"} — still parse.
                    try:
                        parsed = message_from_dict(item)
                    except Exception:  # noqa: BLE001 — skip a malformed row, not the session
                        logger.warning("dropping malformed session row: %r", item)
                        parsed = None
                    if parsed is not None:
                        messages.append(parsed)
                        continue
                role = str(item.get("role") or "")
                content = str(item.get("content") or "")
                if role == "user" and content:
                    messages.append(UserMessage(content=content))
                elif role == "assistant" and content:
                    from brain.messages import AssistantMessage, TextContent

                    messages.append(
                        AssistantMessage(content=[TextContent(text=content)])
                    )
        return cls(
            id=str(data["id"]),
            messages=messages,
            created_at=str(data.get("created_at") or utc_now()),
            updated_at=str(data.get("updated_at") or utc_now()),
            title=data.get("title"),
            artifacts=[a for a in data.get("artifacts") or [] if isinstance(a, dict)],
            schema_version=max(version, SESSION_SCHEMA_VERSION),
        )


@dataclass(slots=True)
class SessionSummary:
    id: str
    updated_at: str
    title: str | None
    message_count: int


class SessionStore(Protocol):
    async def get(self, session_id: str) -> Session | None: ...

    async def save(self, session: Session) -> None: ...

    async def list_recent(self, *, limit: int = 20) -> list[SessionSummary]: ...


class MemorySessionStore:
    """Process-local sessions (tests / no storage account)."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    async def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    async def save(self, session: Session) -> None:
        session.updated_at = utc_now()
        session.schema_version = SESSION_SCHEMA_VERSION
        self._sessions[session.id] = session

    async def list_recent(self, *, limit: int = 20) -> list[SessionSummary]:
        items = sorted(
            self._sessions.values(),
            key=lambda s: s.updated_at,
            reverse=True,
        )
        return [
            SessionSummary(
                id=s.id,
                updated_at=s.updated_at,
                title=s.title,
                message_count=len(s.ui_chat_messages()),
            )
            for s in items[:limit]
        ]


class ObjectStoreSessionStore:
    """Persist sessions as JSON, one object per session at {prefix}/{id}.json,
    via any ObjectStore (Azure Blob or local disk — see app/storage.py).
    """

    def __init__(self, store: ObjectStore, settings: Settings | None = None) -> None:
        self._store = store
        self._settings = settings or get_settings()

    def _key(self, session_id: str) -> str:
        return f"{self._settings.sessions_prefix}/{session_id}.json"

    async def get(self, session_id: str) -> Session | None:
        if not SAFE_SESSION_ID.match(session_id):
            return None
        from asyncio import to_thread

        # A genuinely new session (never written) is expected and quiet -
        # ObjectStore.get_bytes() returns None for that. Anything else - auth
        # failure, network error, a permissions problem on the local dir - is
        # a real problem and must NOT be treated the same way: silently
        # returning None here previously made every turn look like a fresh
        # session once storage broke, even mid-conversation, since the
        # caller can't tell "doesn't exist" from "couldn't check". Let it
        # propagate so FallbackSessionStore can react instead of hiding it.
        raw = await to_thread(self._store.get_bytes, self._key(session_id))
        if raw is None:
            return None
        try:
            return Session.from_dict(json.loads(raw.decode("utf-8")))
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError) as exc:
            logger.warning("session %s object is corrupt, starting fresh: %s", session_id, exc)
            return None

    async def save(self, session: Session) -> None:
        from asyncio import to_thread

        session.updated_at = utc_now()
        session.schema_version = SESSION_SCHEMA_VERSION
        payload = json.dumps(session.to_dict(), ensure_ascii=False).encode("utf-8")
        await to_thread(
            self._store.put_bytes,
            self._key(session.id),
            payload,
            content_type="application/json",
        )

    async def list_recent(self, *, limit: int = 20) -> list[SessionSummary]:
        from asyncio import to_thread

        prefix = f"{self._settings.sessions_prefix}/"

        # No "expected" failure mode here (unlike get()'s not-found case) -
        # let any failure (auth, network, missing directory) propagate so
        # FallbackSessionStore can react to it instead of it being hidden
        # behind an empty-looking session list.
        metas = await to_thread(self._store.list_keys, prefix)
        metas.sort(
            key=lambda m: m.last_modified or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        summaries: list[SessionSummary] = []
        for meta in metas[:limit]:
            sid = meta.key.removeprefix(prefix).removesuffix(".json")
            if not SAFE_SESSION_ID.match(sid):
                continue
            title: str | None = None
            message_count = 0
            try:
                raw = await to_thread(self._store.get_bytes, meta.key)
                if raw is not None:
                    session = Session.from_dict(json.loads(raw.decode("utf-8")))
                    title = session.title
                    message_count = len(session.ui_chat_messages())
            except Exception:  # noqa: BLE001 — list should still show the id
                logger.debug("failed to read session summary %s", sid, exc_info=True)
            summaries.append(
                SessionSummary(
                    id=sid,
                    updated_at=meta.last_modified.isoformat() if meta.last_modified else utc_now(),
                    title=title,
                    message_count=message_count,
                )
            )
        return summaries


class FallbackSessionStore:
    """Wraps a primary store (Azure Blob or local disk) with an in-memory fallback.

    If the primary store fails for any reason other than a session
    genuinely not existing yet (see ObjectStoreSessionStore.get()), that is a
    real problem - not a signal to quietly start every conversation over. This
    logs it once, loudly, and demotes to memory for the rest of the
    process: a storage outage degrades to "sessions don't survive a
    restart", never to "the agent forgets what you just said".
    """

    def __init__(self, primary: SessionStore, fallback: SessionStore) -> None:
        self._primary: SessionStore | None = primary
        self._fallback = fallback

    def _demote(self, exc: Exception) -> None:
        if self._primary is None:
            return
        logger.error(
            "Session storage backend failed (%s: %s) - falling back to "
            "in-memory sessions for the rest of this process. Conversations "
            "will keep working but won't survive a restart until the "
            "underlying storage problem is fixed.",
            type(exc).__name__,
            exc,
        )
        self._primary = None

    async def get(self, session_id: str) -> Session | None:
        if self._primary is not None:
            try:
                return await self._primary.get(session_id)
            except Exception as exc:  # noqa: BLE001
                self._demote(exc)
        return await self._fallback.get(session_id)

    async def save(self, session: Session) -> None:
        if self._primary is not None:
            try:
                await self._primary.save(session)
                return
            except Exception as exc:  # noqa: BLE001
                self._demote(exc)
        await self._fallback.save(session)

    async def list_recent(self, *, limit: int = 20) -> list[SessionSummary]:
        if self._primary is not None:
            try:
                return await self._primary.list_recent(limit=limit)
            except Exception as exc:  # noqa: BLE001
                self._demote(exc)
        return await self._fallback.list_recent(limit=limit)


_store: SessionStore | None = None


def get_session_store() -> SessionStore:
    global _store
    if _store is not None:
        return _store
    settings = get_settings()
    if settings.storage_backend != "memory":
        try:
            object_store = get_object_store()
        except Exception:  # noqa: BLE001
            logger.exception("falling back to memory session store")
            object_store = None
        if object_store is not None:
            _store = FallbackSessionStore(
                ObjectStoreSessionStore(object_store, settings), MemorySessionStore()
            )
            return _store
    _store = MemorySessionStore()
    return _store


def reset_session_store_for_tests() -> None:
    """Clear the process-wide store singleton (tests only)."""
    global _store
    _store = None


SESSION_NAME_SYSTEM_PROMPT = (
    "You write concise chat session names. Reply with only a short title, "
    "maximum four words, no quotes, no punctuation-only output."
)


def _sanitize_title(text: str) -> str | None:
    """Collapse to a handful of bare words — tau_coding.session's sanitizer."""
    cleaned = " ".join(text.split()).strip()
    cleaned = cleaned.strip("\"'`“”‘’")
    cleaned = cleaned.strip(string.punctuation + " ")
    words = [word.strip(string.punctuation + "\"'`“”‘’") for word in cleaned.split()]
    words = [word for word in words if word]
    if not words:
        return None
    return " ".join(words[:4])


async def generate_session_title(provider: ModelProvider, first_message: str) -> str | None:
    """Name a session from its first message with a tiny, tool-free model call.

    Mirrors tau_coding.session's ``_generate_session_name``: never raises —
    naming must not break the real turn — and falls back to sanitizing the
    raw message itself so a provider failure still yields a short title
    instead of nothing.
    """
    title: str | None = None
    try:
        text_parts: list[str] = []
        prompt = (
            "Create a concise session name for this first user message. "
            f"Use at most four words.\n\nUser message:\n{first_message}"
        )
        async for chunk in provider.stream(
            [UserMessage(content=prompt)],
            [],
            system=SESSION_NAME_SYSTEM_PROMPT,
        ):
            if chunk.content:
                text_parts.append(chunk.content)
        title = _sanitize_title("".join(text_parts))
    except Exception:  # noqa: BLE001 — naming must never break the real turn
        logger.debug("session auto-naming failed", exc_info=True)
    return title or _sanitize_title(first_message)
