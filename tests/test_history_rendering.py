"""app/rendering/history.py: the transcript as the UI's history view shows it.

Moved out of brain/ (it was brain.messages.ui_messages): it's a projection
for one frontend, like the SSE renderer. Images are now referenced by
digest, so a reopened chat can show them, instead of a text note.
"""

from __future__ import annotations

import base64

from app.context_window import COMPACTION_SUMMARY_PREFIX
from app.image_processing import image_digest
from app.rendering.history import UNSHOWABLE_IMAGE_NOTE, history_rows
from brain.messages import (
    AssistantMessage,
    ImageContent,
    TextContent,
    ToolResultMessage,
    UserMessage,
)


def _image(raw: bytes) -> ImageContent:
    return ImageContent(data=base64.b64encode(raw).decode("ascii"), mime_type="image/png")


def test_text_turns_become_rows_in_order() -> None:
    rows = history_rows(
        [
            UserMessage(content="hello", timestamp_ms=1),
            AssistantMessage(content="hi there", model="mock", timestamp_ms=2),
        ]
    )
    assert [(r.role, r.content, r.ts, r.images) for r in rows] == [
        ("user", "hello", 1, ()),
        ("assistant", "hi there", 2, ()),
    ]


def test_images_are_referenced_by_digest_with_no_text_note() -> None:
    rows = history_rows(
        [UserMessage(content=[TextContent(text="what is this?"), _image(b"one"), _image(b"two")])]
    )
    assert rows[0].content == "what is this?"
    assert rows[0].images == (image_digest(b"one"), image_digest(b"two"))


def test_an_image_only_turn_still_gets_a_row() -> None:
    rows = history_rows([UserMessage(content=[_image(b"one")])])
    assert [(r.role, r.content, r.images) for r in rows] == [("user", "", (image_digest(b"one"),))]


def test_an_image_that_cannot_be_decoded_leaves_a_note_not_a_gap() -> None:
    broken = ImageContent(data="not base64!", mime_type="image/png")
    rows = history_rows([UserMessage(content=[broken])])
    assert [(r.content, r.images) for r in rows] == [(UNSHOWABLE_IMAGE_NOTE, ())]


def test_tool_results_failed_turns_and_empty_turns_are_not_shown() -> None:
    rows = history_rows(
        [
            UserMessage(content="   "),
            AssistantMessage(content="", model="mock"),
            AssistantMessage(content="partial", model="mock", stop_reason="error"),
            ToolResultMessage(tool_call_id="c1", tool_name="bash", content="output"),
        ]
    )
    assert rows == []


def test_a_summary_is_shown_where_it_happened() -> None:
    rows = history_rows(
        [
            UserMessage(content="old", timestamp_ms=100),
            UserMessage(content="new", timestamp_ms=300),  # same ms as the summary
        ],
        summaries=[(300, "the summary")],
    )
    assert [(r.role, r.content) for r in rows] == [
        ("user", "old"),
        ("context_summary", "the summary"),
        ("user", "new"),
    ]


def test_an_inline_summary_from_an_old_session_is_shown_and_ordered() -> None:
    rows = history_rows(
        [
            UserMessage(content=f"{COMPACTION_SUMMARY_PREFIX}## Goal\n- do the thing", timestamp_ms=900),
            UserMessage(content="kept", timestamp_ms=500),
        ]
    )
    assert rows[0].role == "context_summary"
    assert rows[0].content == "## Goal\n- do the thing"
    assert [r.ts for r in rows] == sorted(r.ts for r in rows)
