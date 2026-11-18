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
    ToolCall,
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


def test_failed_turns_and_empty_turns_are_not_shown() -> None:
    rows = history_rows(
        [
            UserMessage(content="   "),
            AssistantMessage(content="", model="mock"),
            AssistantMessage(content="partial", model="mock", stop_reason="error"),
        ]
    )
    assert rows == []


def test_a_tool_run_comes_back_as_the_card_the_live_stream_showed() -> None:
    rows = history_rows(
        [
            UserMessage(content="list files", timestamp_ms=1),
            AssistantMessage(
                content=[ToolCall(id="c1", name="bash", arguments={"command": "ls", "description": "Listing files"})],
                model="mock",
                timestamp_ms=2,
            ),
            ToolResultMessage(
                tool_call_id="c1",
                tool_name="bash",
                content="a.txt",
                details={"exit_code": 0},
                timestamp_ms=3,
            ),
            AssistantMessage(content="One file.", model="mock", timestamp_ms=4),
        ]
    )
    assert [(r.role, r.content, r.ts) for r in rows] == [
        ("user", "list files", 1),
        ("tool", "a.txt", 3),
        ("assistant", "One file.", 4),
    ]
    assert rows[1].tool == {
        "name": "bash",
        "label": "Listing files",
        "call_id": "c1",
        "args": {"command": "ls", "description": "Listing files"},
        "is_error": False,
        "step": 1,
        "exit_code": 0,
    }


def test_an_edit_keeps_its_diff_and_a_failure_stays_a_failure() -> None:
    rows = history_rows(
        [
            ToolResultMessage(
                tool_call_id="c1", tool_name="edit", content="ok", details={"patch": "@@ -1 +1 @@\n-a\n+b"}
            ),
            ToolResultMessage(tool_call_id="c2", tool_name="read", content="nope", is_error=True),
        ]
    )
    assert rows[0].tool["diff"] == "@@ -1 +1 @@\n-a\n+b"
    assert rows[1].tool["is_error"] is True
    assert rows[1].tool["args"] == {}  # no matching call (an old or repaired transcript)


def test_a_long_tool_result_is_shortened_for_the_card() -> None:
    from app.rendering.history import MAX_HISTORY_RESULT_CHARS, SHORTENED_RESULT_NOTE

    long = "x" * (MAX_HISTORY_RESULT_CHARS + 10)
    rows = history_rows([ToolResultMessage(tool_call_id="c1", tool_name="read", content=long)])
    assert rows[0].content == "x" * MAX_HISTORY_RESULT_CHARS + SHORTENED_RESULT_NOTE


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


def test_each_tool_run_says_which_model_step_made_it() -> None:
    def call(i: str) -> ToolCall:
        return ToolCall(id=i, name="read", arguments={"path": f"{i}.txt"})

    rows = history_rows(
        [
            AssistantMessage(content=[call("a"), call("b")], model="mock"),  # one step, two calls
            ToolResultMessage(tool_call_id="a", tool_name="read", content="A"),
            ToolResultMessage(tool_call_id="b", tool_name="read", content="B"),
            AssistantMessage(content=[call("c")], model="mock"),  # the next step
            ToolResultMessage(tool_call_id="c", tool_name="read", content="C"),
        ]
    )
    assert [r.tool["step"] for r in rows] == [0, 0, 3]
