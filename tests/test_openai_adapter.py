"""Image serialization on both wire shapes: Chat Completions' image_url
blocks (providers/openai_adapter.py) and the Responses API's input_image
blocks (providers/openai_responses.py) — ported from tau_ai/openai_compatible.py's
equivalent blocks, scaled to Humphrey's user-only images.
"""

from __future__ import annotations

from brain.messages import ImageContent, TextContent, UserMessage
from providers.openai_adapter import messages_to_openai
from providers.openai_responses import messages_to_responses_input


def test_text_only_user_message_stays_a_plain_string_on_chat_completions() -> None:
    out = messages_to_openai([UserMessage(content="hello")], system="sys")
    user_row = out[1]
    assert user_row == {"role": "user", "content": "hello"}


def test_image_message_becomes_multipart_content_on_chat_completions() -> None:
    message = UserMessage(
        content=[
            TextContent(text="what is this?"),
            ImageContent(data="AAAA", mime_type="image/png"),
        ]
    )
    out = messages_to_openai([message], system="sys")
    user_row = out[1]
    assert user_row["role"] == "user"
    assert user_row["content"] == [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]


def test_image_only_message_omits_the_text_block_on_chat_completions() -> None:
    message = UserMessage(content=[ImageContent(data="AAAA", mime_type="image/jpeg")])
    out = messages_to_openai([message], system="sys")
    user_row = out[1]
    assert user_row["content"] == [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}},
    ]


def test_text_only_user_message_stays_a_plain_string_on_responses_api() -> None:
    items = messages_to_responses_input([UserMessage(content="hello")])
    assert items == [{"role": "user", "content": "hello"}]


def test_image_message_becomes_multipart_content_on_responses_api() -> None:
    message = UserMessage(
        content=[
            TextContent(text="what is this?"),
            ImageContent(data="AAAA", mime_type="image/png"),
        ]
    )
    items = messages_to_responses_input([message])
    assert items == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "what is this?"},
                {
                    "type": "input_image",
                    "detail": "auto",
                    "image_url": "data:image/png;base64,AAAA",
                },
            ],
        }
    ]


def test_multiple_images_all_serialize_on_both_wire_shapes() -> None:
    message = UserMessage(
        content=[
            ImageContent(data="AAAA", mime_type="image/png"),
            ImageContent(data="BBBB", mime_type="image/jpeg"),
        ]
    )
    chat_row = messages_to_openai([message], system="sys")[1]
    assert len(chat_row["content"]) == 2

    responses_item = messages_to_responses_input([message])[0]
    assert len(responses_item["content"]) == 2
