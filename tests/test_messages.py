"""brain/messages.py: ImageContent and UserMessage's widened content shape.
(The UI's history projection lives in app/rendering/history.py; see
tests/test_history_rendering.py.)
"""

from __future__ import annotations

from brain.messages import (
    ImageContent,
    TextContent,
    UserMessage,
    message_from_dict,
    message_to_dict,
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
