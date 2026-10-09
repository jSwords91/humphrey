"""brain/messages.py: ImageContent + UserMessage's widened content shape, and
the ui_messages() projection's "no dropped information" handling of images.
"""

from __future__ import annotations

from brain.messages import (
    COMPACTION_SUMMARY_PREFIX,
    ImageContent,
    TextContent,
    UserMessage,
    message_from_dict,
    message_to_dict,
    ui_messages,
)


def test_user_message_content_still_accepts_a_bare_string() -> None:
    message = UserMessage(content="hello")
    assert message.text == "hello"
    assert message.images == []


def test_user_message_with_image_blocks() -> None:
    message = UserMessage(
        content=[
            TextContent(text="what is this?"),
            ImageContent(data="AAAA", mime_type="image/png"),
        ]
    )
    assert message.text == "what is this?"
    assert len(message.images) == 1
    assert message.images[0].mime_type == "image/png"


def test_user_message_image_only_has_empty_text() -> None:
    message = UserMessage(content=[ImageContent(data="AAAA", mime_type="image/png")])
    assert message.text == ""
    assert len(message.images) == 1


def test_image_message_round_trips_through_dict_serialization() -> None:
    original = UserMessage(
        content=[
            TextContent(text="check this"),
            ImageContent(data="AAAA", mime_type="image/jpeg"),
        ]
    )
    restored = message_from_dict(message_to_dict(original))
    assert isinstance(restored, UserMessage)
    assert restored.text == "check this"
    assert restored.images == original.images


def test_ui_messages_notes_image_only_turn_instead_of_dropping_it() -> None:
    messages = [UserMessage(content=[ImageContent(data="AAAA", mime_type="image/png")])]
    rows = ui_messages(messages)
    assert len(rows) == 1
    assert rows[0]["role"] == "user"
    assert "1 image" in rows[0]["content"]


def test_ui_messages_notes_multiple_image_only_turn() -> None:
    messages = [
        UserMessage(
            content=[
                ImageContent(data="AAAA", mime_type="image/png"),
                ImageContent(data="BBBB", mime_type="image/png"),
            ]
        )
    ]
    rows = ui_messages(messages)
    assert "2 images" in rows[0]["content"]


def test_ui_messages_notes_images_alongside_a_caption() -> None:
    messages = [
        UserMessage(
            content=[
                TextContent(text="what's in this photo?"),
                ImageContent(data="AAAA", mime_type="image/png"),
            ]
        )
    ]
    rows = ui_messages(messages)
    assert rows[0]["content"].startswith("what's in this photo?")
    assert "1 image attached" in rows[0]["content"]


def test_ui_messages_does_not_confuse_a_compaction_summary_with_an_image_message() -> None:
    messages = [UserMessage(content=f"{COMPACTION_SUMMARY_PREFIX}## Goal\n- do the thing")]
    rows = ui_messages(messages)
    assert rows[0]["role"] == "context_summary"
    assert rows[0]["content"] == "## Goal\n- do the thing"
