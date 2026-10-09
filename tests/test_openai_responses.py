"""Responses API support: model detection, payload shape, SSE parsing.

Event shapes here are taken directly from tau_ai's _ResponsesStreamParser and
Microsoft/OpenAI's Responses API docs — no live network calls.
"""

from __future__ import annotations

import json

import pytest

from app.tools import TOOLS
from brain.messages import AssistantMessage, TextContent, ThinkingContent, ToolCall, UserMessage
from providers.openai_responses import (
    ResponsesStreamParser,
    build_responses_payload,
    messages_to_responses_input,
    uses_responses_api,
)


def test_uses_responses_api_matches_known_reasoning_models() -> None:
    assert uses_responses_api("gpt-5.6-sol") is True
    assert uses_responses_api("gpt-5.6") is True
    assert uses_responses_api("gpt-5.4-mini") is True
    assert uses_responses_api("GPT-5.5-preview") is True  # case-insensitive
    assert uses_responses_api("gpt-6-sol") is True
    assert uses_responses_api("gpt-6.1-sol") is True
    assert uses_responses_api("o4-mini") is True


def test_uses_responses_api_false_for_non_reasoning_models() -> None:
    assert uses_responses_api("gpt-4o-mini") is False
    assert uses_responses_api("gpt-4o") is False
    assert uses_responses_api("gpt-5.3-chat-latest") is False
    assert uses_responses_api("gpt-60") is False  # family match, not a bare prefix


def test_messages_to_responses_input_maps_roles() -> None:
    messages = [
        UserMessage(content="hi"),
        AssistantMessage(content=[TextContent(text="hello")]),
    ]
    items = messages_to_responses_input(messages)
    assert items[0] == {"role": "user", "content": "hi"}
    assert items[1] == {"role": "assistant", "content": "hello"}


def test_messages_to_responses_input_replays_tool_calls() -> None:
    from brain.messages import ToolResultMessage

    messages = [
        AssistantMessage(content=[ToolCall(id="call_1", name="status", arguments={})]),
        ToolResultMessage(tool_call_id="call_1", tool_name="status", content=[]),
    ]
    items = messages_to_responses_input(messages)
    assert items[0]["type"] == "function_call"
    assert items[0]["call_id"] == "call_1"
    assert items[0]["name"] == "status"
    assert items[1]["type"] == "function_call_output"
    assert items[1]["call_id"] == "call_1"


def test_messages_to_responses_input_replays_reasoning_signature() -> None:
    raw_item = {"type": "reasoning", "id": "rs_1", "summary": []}
    messages = [
        AssistantMessage(
            content=[
                ThinkingContent(thinking="I considered X", thinking_signature=json.dumps(raw_item)),
            ]
        ),
    ]
    items = messages_to_responses_input(messages)
    assert items[0] == raw_item


def test_messages_to_responses_input_strips_output_only_status_from_reasoning_item() -> None:
    """Regression test: replaying "status" verbatim fails live against gpt-5.6-sol
    with "Unknown parameter: 'input[N].status'" — it's populated only on items
    the API *returns*, per the openai SDK's own ResponseReasoningItemParam docs.
    """
    raw_item_from_api = {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [{"type": "summary_text", "text": "..."}],
        "status": "completed",
    }
    messages = [
        AssistantMessage(
            content=[ThinkingContent(thinking="...", thinking_signature=json.dumps(raw_item_from_api))]
        ),
    ]
    items = messages_to_responses_input(messages)
    assert "status" not in items[0]
    assert items[0]["id"] == "rs_1"


def test_messages_to_responses_input_ignores_unparseable_signature() -> None:
    messages = [
        AssistantMessage(content=[ThinkingContent(thinking="x", thinking_signature="not json")]),
    ]
    assert messages_to_responses_input(messages) == []


def test_build_responses_payload_shape() -> None:
    payload = build_responses_payload(
        model="gpt-5.6-sol",
        system="be terse",
        messages=[UserMessage(content="hi")],
        tools=list(TOOLS)[:1],
        reasoning_effort="medium",
    )
    assert payload["model"] == "gpt-5.6-sol"
    assert payload["instructions"] == "be terse"
    assert payload["store"] is False
    assert payload["input"] == [{"role": "user", "content": "hi"}]
    assert payload["reasoning"] == {"effort": "medium", "summary": "auto"}
    assert payload["tools"][0]["type"] == "function"


def test_build_responses_payload_omits_reasoning_when_no_effort_configured() -> None:
    payload = build_responses_payload(
        model="gpt-5.6-sol", system="s", messages=[], tools=[], reasoning_effort=None
    )
    assert "reasoning" not in payload
    assert "tools" not in payload


# --- SSE parsing -------------------------------------------------------------


def test_parser_text_delta() -> None:
    parser = ResponsesStreamParser()
    chunk = parser.feed({"type": "response.output_text.delta", "delta": "hello"})
    assert chunk is not None
    assert chunk.content == "hello"


def test_parser_thinking_delta() -> None:
    parser = ResponsesStreamParser()
    chunk = parser.feed({"type": "response.reasoning_summary_text.delta", "delta": "I think"})
    assert chunk is not None
    assert chunk.thinking == "I think"


def test_parser_reasoning_summary_part_done_inserts_paragraph_break() -> None:
    parser = ResponsesStreamParser()
    chunk = parser.feed({"type": "response.reasoning_summary_part.done"})
    assert chunk is not None
    assert chunk.thinking == "\n\n"


def test_parser_function_call_added_then_arguments_then_done() -> None:
    parser = ResponsesStreamParser()

    added = parser.feed(
        {
            "type": "response.output_item.added",
            "item": {"type": "function_call", "id": "item_1", "call_id": "call_1", "name": "status"},
        }
    )
    assert added is not None
    assert added.tool_calls[0].id == "call_1"
    assert added.tool_calls[0].name == "status"
    index = added.tool_calls[0].index

    delta = parser.feed(
        {"type": "response.function_call_arguments.delta", "item_id": "item_1", "delta": "{}"}
    )
    assert delta is not None
    assert delta.tool_calls[0].index == index
    assert delta.tool_calls[0].arguments == "{}"

    # .done after a delta was already seen must not duplicate the arguments.
    done = parser.feed(
        {
            "type": "response.output_item.done",
            "item": {"type": "function_call", "id": "item_1", "call_id": "call_1", "arguments": "{}"},
        }
    )
    assert done is None


def test_parser_function_call_done_without_prior_delta_emits_arguments() -> None:
    parser = ResponsesStreamParser()
    done = parser.feed(
        {
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "id": "item_1",
                "call_id": "call_1",
                "arguments": '{"a": 1}',
            },
        }
    )
    assert done is not None
    assert done.tool_calls[0].arguments == '{"a": 1}'


def test_parser_completed_extracts_usage_and_reasoning_signature() -> None:
    parser = ResponsesStreamParser()
    parser.feed(
        {
            "type": "response.output_item.added",
            "item": {"type": "reasoning", "id": "rs_1", "summary": []},
        }
    )
    chunk = parser.feed(
        {
            "type": "response.completed",
            "response": {
                "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
            },
        }
    )
    assert chunk is not None
    assert chunk.usage == {"input": 10, "output": 20, "total_tokens": 30}
    signature = json.loads(chunk.thinking_signature)
    assert signature["id"] == "rs_1"


def test_parser_failed_event_raises() -> None:
    parser = ResponsesStreamParser()
    with pytest.raises(RuntimeError, match="boom"):
        parser.feed(
            {"type": "response.failed", "response": {"error": {"message": "boom"}}}
        )


def test_parser_error_event_raises() -> None:
    parser = ResponsesStreamParser()
    with pytest.raises(RuntimeError):
        parser.feed({"type": "error", "message": "nope"})


def test_parser_unknown_event_is_ignored() -> None:
    parser = ResponsesStreamParser()
    assert parser.feed({"type": "response.some_future_event"}) is None


def test_reasoning_from_a_different_model_is_not_replayed() -> None:
    """Switching models mid-conversation must not hand one model another's
    encrypted reasoning; the visible text still carries over."""
    raw_item = {"type": "reasoning", "id": "rs_1", "summary": []}
    messages = [
        AssistantMessage(
            model="gpt-a",
            content=[
                ThinkingContent(thinking="hmm", thinking_signature=json.dumps(raw_item)),
                TextContent(text="answer"),
            ],
        ),
    ]
    assert messages_to_responses_input(messages, model="gpt-a")[0] == raw_item
    assert messages_to_responses_input(messages, model="gpt-b") == [
        {"role": "assistant", "content": "answer"}
    ]
