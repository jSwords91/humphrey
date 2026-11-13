"""Session persistence — memory locally, or Azure Blob / local disk when
STORAGE_BACKEND selects one (see app/storage.py)."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import re
import string
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

from app.settings import Settings, get_settings
from app.storage import ObjectStore, get_object_store
from brain.messages import (
    COMPACTION_SUMMARY_PREFIX,
    AgentMessage,
    UserMessage,
    message_from_dict,
    message_to_dict,
    ui_messages,
)
from brain.provider import ModelProvider

logger = logging.getLogger(__name__)

SAFE_SESSION_ID = re.compile(r"^[a-zA-Z0-9_-]{8,64}$")
# 4: compactions are stored as records next to the full transcript, which
#    is no longer rewritten (older sessions load unchanged).
SESSION_SCHEMA_VERSION = 4


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


@dataclass(frozen=True, slots=True)
class Compaction:
    """One auto-compaction (app/context_window.py), recorded alongside the
    full transcript instead of replacing it.

    ``first_kept_index`` is where the model's context resumes in
    ``Session.messages``; ``summary`` stands in for everything before it.
    ``created_ms`` is when it happened, which is also where the UI shows it.
    """

    summary: str
    first_kept_index: int
    created_ms: int


@dataclass(slots=True)
class Session:
    id: str
    # The full transcript. Compaction never rewrites it, so the user's
    # history survives; only model_context() gets shorter.
    messages: list[AgentMessage] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    title: str | None = None
    # Files the agent created or changed (app/artifacts.py), in order, each
    # with created_ms so the UI can place it back in the transcript.
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    # Oldest first; the model's context is built from the latest.
    compactions: list[Compaction] = field(default_factory=list)
    schema_version: int = SESSION_SCHEMA_VERSION

    def model_context(self) -> list[AgentMessage]:
        """What the model sees: the latest summary, then the messages kept after it.

        The summary message carries the compaction's own time, which is later
        than every message kept after it. app/context_window relies on that to
        ignore token usage recorded before the compaction.
        """
        if not self.compactions:
            return list(self.messages)
        latest = self.compactions[-1]
        summary = UserMessage(
            content=COMPACTION_SUMMARY_PREFIX + latest.summary, timestamp_ms=latest.created_ms
        )
        return [summary, *self.messages[latest.first_kept_index :]]

    def record_compaction(self, summary: str, *, kept_from: int) -> None:
        """Record a compaction of ``model_context()``, where ``kept_from``
        indexes that context, as a position in the full transcript."""
        head = 1 if self.compactions else 0
        base = self.compactions[-1].first_kept_index if self.compactions else 0
        if not head <= kept_from <= head + len(self.messages) - base:
            raise ValueError(f"kept_from {kept_from} is outside the model context")
        self.compactions.append(
            Compaction(
                summary=summary,
                first_kept_index=base + kept_from - head,
                created_ms=time.time_ns() // 1_000_000,
            )
        )

    def ui_chat_messages(self) -> list[ChatMessage]:
        rows = ui_messages(self.messages)
        # Each compaction shows where it happened, so rows stay in time
        # order (the frontend places artefacts by timestamp).
        for compaction in self.compactions:
            when = compaction.created_ms
            at = next((i for i, r in enumerate(rows) if r["ts"] >= when), len(rows))
            rows.insert(at, {"role": "context_summary", "content": compaction.summary, "ts": when})
        _order_legacy_summaries(rows)
        return [
            ChatMessage(role=row["role"], content=row["content"], ts=row.get("ts"))
            for row in rows
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
            "compactions": [asdict(c) for c in self.compactions],
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
            compactions=_load_compactions(data.get("compactions"), len(messages)),
            schema_version=max(version, SESSION_SCHEMA_VERSION),
        )


def _load_compactions(raw: object, message_count: int) -> list[Compaction]:
    """Parse stored compaction records, dropping any that don't fit the
    transcript. Without a valid record the model just sees more history,
    which the next turn compacts again, so a bad record costs a summary,
    never the session."""
    if not isinstance(raw, list):
        return []
    compactions: list[Compaction] = []
    for item in raw:
        try:
            compaction = Compaction(
                summary=str(item["summary"]),
                first_kept_index=int(item["first_kept_index"]),
                created_ms=int(item["created_ms"]),
            )
        except (KeyError, TypeError, ValueError):
            logger.warning("dropping malformed compaction record: %r", item)
            continue
        previous = compactions[-1].first_kept_index if compactions else 0
        if not previous <= compaction.first_kept_index <= message_count:
            logger.warning("dropping compaction record outside the transcript: %r", item)
            continue
        compactions.append(compaction)
    return compactions


def _order_legacy_summaries(rows: list[dict[str, Any]]) -> None:
    """Sessions compacted before schema 4 stored the summary inline, as the
    first message, stamped later than the messages kept after it. Pull such
    a row's time back to just before the next row so artefacts from the
    kept turns are placed after it, not above it."""
    for i in range(len(rows) - 2, -1, -1):
        if rows[i]["role"] == "context_summary" and rows[i]["ts"] > rows[i + 1]["ts"]:
            rows[i]["ts"] = rows[i + 1]["ts"] - 1


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
    """Persist sessions through any ObjectStore (Azure Blob or local disk -
    see app/storage.py), as three kinds of object:

    {prefix}/{id}.json            the session, with each image replaced by a reference
    {prefix}-images/{sha256}      an image's bytes, stored once however many turns
                                  or sessions include it
    {prefix}-summaries/{id}.json  its title and message count, so the session list
                                  reads a few hundred bytes per session, not all of it

    Images dominate a session's size (one is up to 5MB) and never change, so
    keeping them out of the session object means a turn rewrites tens of KB,
    not every image again. Everything above this class still sees ordinary
    ImageContent: references are filled back in on load. Sessions saved
    before this format load as they are and convert on their next save.
    """

    # Bounds the record of image digests known to be stored (64-char strings).
    _KNOWN_IMAGES_LIMIT = 50_000

    def __init__(self, store: ObjectStore, settings: Settings | None = None) -> None:
        self._store = store
        self._settings = settings or get_settings()
        # Digests this process has written or read: re-saving them is skipped.
        self._known_images: set[str] = set()

    def _key(self, session_id: str) -> str:
        return f"{self._settings.sessions_prefix}/{session_id}.json"

    def _image_key(self, digest: str) -> str:
        return f"{self._settings.sessions_prefix}-images/{digest}"

    def _summary_key(self, session_id: str) -> str:
        return f"{self._settings.sessions_prefix}-summaries/{session_id}.json"

    def _remember_image(self, digest: str) -> None:
        if len(self._known_images) >= self._KNOWN_IMAGES_LIMIT:
            self._known_images.clear()  # only costs some idempotent re-writes
        self._known_images.add(digest)

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
            data = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.warning("session %s object is corrupt, starting fresh: %s", session_id, exc)
            return None
        if not isinstance(data, dict):
            # Corrupt data, not a storage failure: it mustn't reach
            # FallbackSessionStore as an exception and demote to memory.
            logger.warning("session %s object is not a session, starting fresh", session_id)
            return None
        await to_thread(self._fill_in_images, data)  # storage errors propagate, as above
        try:
            return Session.from_dict(data)
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("session %s object is corrupt, starting fresh: %s", session_id, exc)
            return None

    async def save(self, session: Session) -> None:
        from asyncio import to_thread

        session.updated_at = utc_now()
        session.schema_version = SESSION_SCHEMA_VERSION
        await to_thread(self._save_sync, session.to_dict(), _summary_of(session))

    def _save_sync(self, data: dict[str, Any], summary: dict[str, Any]) -> None:
        # Images first, so a stored session never references a missing image.
        for digest, (blob, mime_type) in self._take_out_images(data).items():
            if digest not in self._known_images:
                self._store.put_bytes(self._image_key(digest), blob, content_type=mime_type)
                self._remember_image(digest)
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self._store.put_bytes(self._key(data["id"]), payload, content_type="application/json")
        self._put_summary(summary)

    def _put_summary(self, summary: dict[str, Any]) -> None:
        payload = json.dumps(summary, ensure_ascii=False).encode("utf-8")
        self._store.put_bytes(
            self._summary_key(summary["id"]), payload, content_type="application/json"
        )

    @staticmethod
    def _image_blocks(data: dict[str, Any]) -> list[dict[str, Any]]:
        """Every image block in a serialized session (they only appear in
        user messages - see brain.messages.ImageContent)."""
        blocks: list[dict[str, Any]] = []
        for message in data.get("messages") or []:
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, list):
                blocks.extend(
                    b for b in content if isinstance(b, dict) and b.get("type") == "image"
                )
        return blocks

    def _take_out_images(self, data: dict[str, Any]) -> dict[str, tuple[bytes, str]]:
        """Swap each image's base64 for a reference to its bytes, in place,
        and return the bytes to store by digest."""
        images: dict[str, tuple[bytes, str]] = {}
        for block in self._image_blocks(data):
            encoded = block.get("data")
            if not isinstance(encoded, str):
                continue
            try:
                blob = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError):
                continue  # keep it inline rather than lose it
            digest = hashlib.sha256(blob).hexdigest()
            # Serialized by alias (brain.messages.WireModel), so "mimeType".
            mime_type = block.get("mimeType") or block.get("mime_type")
            images[digest] = (blob, str(mime_type or "application/octet-stream"))
            del block["data"]
            block["ref"] = digest
        return images

    def _fill_in_images(self, data: dict[str, Any]) -> None:
        """Put each referenced image's base64 back, in place. A missing image
        becomes a visible note, so one lost object never loses the session."""
        for message in data.get("messages") or []:
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for i, block in enumerate(content):
                if not (isinstance(block, dict) and block.get("type") == "image"):
                    continue
                digest = block.get("ref")
                if "data" in block or not isinstance(digest, str):
                    continue
                blob = None
                if _SHA256.match(digest):  # never build a key from anything else
                    blob = self._store.get_bytes(self._image_key(digest))
                if blob is None:
                    logger.warning("session image %r is missing; showing a note instead", digest)
                    content[i] = {"type": "text", "text": MISSING_IMAGE_NOTE}
                    continue
                block["data"] = base64.b64encode(blob).decode("ascii")
                del block["ref"]
                self._remember_image(digest)

    async def list_recent(self, *, limit: int = 20) -> list[SessionSummary]:
        from asyncio import to_thread

        prefix = f"{self._settings.sessions_prefix}/"

        # No "expected" failure mode here (unlike get()'s not-found case) -
        # let any failure (auth, network, missing directory) propagate so
        # FallbackSessionStore can react to it instead of it being hidden
        # behind an empty-looking session list.
        metas = await to_thread(self._store.list_keys, prefix)
        sessions = [
            (meta, sid)
            for meta in metas
            if SAFE_SESSION_ID.match(sid := meta.key.removeprefix(prefix).removesuffix(".json"))
            and meta.key == self._key(sid)
        ]
        sessions.sort(
            key=lambda pair: pair[0].last_modified or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        summaries: list[SessionSummary] = []
        for meta, sid in sessions[:limit]:
            title: str | None = None
            message_count = 0
            try:
                summary = await self._read_summary(sid)
                if summary is not None:
                    title, message_count = summary
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

    async def _read_summary(self, session_id: str) -> tuple[str | None, int] | None:
        """The stored summary, or, for a session saved before summaries
        existed, one built from the session and stored for next time."""
        from asyncio import to_thread

        raw = await to_thread(self._store.get_bytes, self._summary_key(session_id))
        if raw is not None:
            try:
                data = json.loads(raw.decode("utf-8"))
                return data.get("title"), int(data.get("message_count") or 0)
            except (ValueError, TypeError, AttributeError):  # bad JSON, encoding or shape
                logger.warning("session %s summary is corrupt; rebuilding it", session_id)
        session = await self.get(session_id)
        if session is None:
            return None
        summary = _summary_of(session)
        try:
            await to_thread(self._put_summary, summary)
        except Exception:  # noqa: BLE001 - the list still works; retried next time
            logger.warning("could not store the summary for session %s", session_id, exc_info=True)
        return summary["title"], summary["message_count"]


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
MISSING_IMAGE_NOTE = "(An image was attached here, but its stored copy is missing.)"


def _summary_of(session: Session) -> dict[str, Any]:
    return {
        "id": session.id,
        "title": session.title,
        "message_count": len(session.ui_chat_messages()),
    }


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
