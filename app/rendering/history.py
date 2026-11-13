"""The transcript as the web UI's history view shows it: one row per visible
user or assistant turn, plus a row for each context summary.

A projection for one frontend, so it lives here beside the SSE renderer and
not in brain/: brain/ keeps the full typed transcript and knows nothing
about how a particular UI shows it. Images are referenced by their digest
(app.image_processing.image_digest), never inlined, so reopening a chat
costs its text; the UI fetches each image by URL, lazily, and caches it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from app.context_window import COMPACTION_SUMMARY_PREFIX
from app.image_processing import image_bytes, image_digest
from brain.messages import AgentMessage, AssistantMessage, UserMessage

UNSHOWABLE_IMAGE_NOTE = "(An image was attached here, but it can't be shown.)"


@dataclass(slots=True)
class HistoryRow:
    role: Literal["user", "assistant", "context_summary"]
    content: str
    # Milliseconds. Rows are in time order, which the UI relies on to put
    # artefacts back where they happened.
    ts: int
    # Digests of the images attached to a user turn, in order.
    images: tuple[str, ...] = ()


def history_rows(
    messages: Sequence[AgentMessage], summaries: Sequence[tuple[int, str]] = ()
) -> list[HistoryRow]:
    """Project ``messages`` for the history view, with each ``(created_ms,
    text)`` context summary shown where it happened."""
    rows: list[HistoryRow] = []
    for message in messages:
        if isinstance(message, UserMessage):
            row = _user_row(message)
            if row is not None:
                rows.append(row)
        elif isinstance(message, AssistantMessage):
            text = message.text.strip()
            if text and message.stop_reason not in {"error", "aborted"}:
                rows.append(HistoryRow("assistant", text, message.timestamp_ms))

    for created_ms, text in summaries:
        at = next((i for i, r in enumerate(rows) if r.ts >= created_ms), len(rows))
        rows.insert(at, HistoryRow("context_summary", text, created_ms))
    _order_inline_summaries(rows)
    return rows


def _user_row(message: UserMessage) -> HistoryRow | None:
    text = message.text.strip()
    if text.startswith(COMPACTION_SUMMARY_PREFIX):
        # A summary stored inline, as sessions compacted before schema 4 did.
        return HistoryRow(
            "context_summary", text.removeprefix(COMPACTION_SUMMARY_PREFIX), message.timestamp_ms
        )
    digests: list[str] = []
    unshowable = 0
    for image in message.images:
        data = image_bytes(image.data)
        if data is None:
            unshowable += 1
        else:
            digests.append(image_digest(data))
    if unshowable:
        # Don't let an image vanish without a trace.
        text = "\n".join(part for part in (text, UNSHOWABLE_IMAGE_NOTE) if part)
    if not text and not digests:
        return None
    return HistoryRow("user", text, message.timestamp_ms, tuple(digests))


def _order_inline_summaries(rows: list[HistoryRow]) -> None:
    """An inline summary is stamped at compaction time, later than the
    messages kept after it. Pull its time back to just before the next row
    so artefacts from the kept turns are placed after it, not above it."""
    for i in range(len(rows) - 2, -1, -1):
        if rows[i].role == "context_summary" and rows[i].ts > rows[i + 1].ts:
            rows[i].ts = rows[i + 1].ts - 1
