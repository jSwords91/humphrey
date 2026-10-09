"""OpenAI chat-completions adapters for the portable transcript."""

from __future__ import annotations

import json
from typing import Any

from brain.messages import (
    AgentMessage,
    AssistantMessage,
    TextContent,
    ToolResultMessage,
    UserMessage,
)
from brain.tools import AgentTool
from providers.content import text_and_images


def _user_content_to_openai_chat(content: str | list) -> Any:
    """Chat-completions ``image_url`` blocks (tau_ai/openai_compatible.py's
    ``_messages_to_openai_chat``, scaled to Humphrey's user-only images).
    """
    text, images = text_and_images(content)
    if not images:
        return text
    parts: list[dict[str, Any]] = []
    if text:
        parts.append({"type": "text", "text": text})
    parts.extend(
        {"type": "image_url", "image_url": {"url": f"data:{img.mime_type};base64,{img.data}"}}
        for img in images
    )
    return parts


def tools_to_openai(tools: list[AgentTool]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.openai_parameters(),
            },
        }
        for tool in tools
    ]


def messages_to_openai(
    messages: list[AgentMessage],
    *,
    system: str,
) -> list[dict[str, Any]]:
    """Map portable transcript → OpenAI chat messages (system prepended)."""
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for message in messages:
        if isinstance(message, UserMessage):
            out.append({"role": "user", "content": _user_content_to_openai_chat(message.content)})
        elif isinstance(message, AssistantMessage):
            if message.stop_reason in {"error", "aborted"} and not message.content:
                # Empty failures must not poison the next provider request.
                continue
            tool_calls = message.tool_calls
            row: dict[str, Any] = {
                "role": "assistant",
                "content": message.text or None,
            }
            if tool_calls:
                row["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments, ensure_ascii=False),
                        },
                    }
                    for call in tool_calls
                ]
            out.append(row)
        elif isinstance(message, ToolResultMessage):
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": message.tool_call_id,
                    "content": message.text,
                }
            )
    return out


def user_messages_from_wire(rows: list[dict[str, Any]]) -> list[AgentMessage]:
    """Bootstrap a transcript from the UI's user/assistant wire format."""
    messages: list[AgentMessage] = []
    for row in rows:
        role = str(row.get("role") or "")
        content = str(row.get("content") or "")
        if not content.strip():
            continue
        if role == "user":
            messages.append(UserMessage(content=content))
        elif role == "assistant":
            messages.append(
                AssistantMessage(content=[TextContent(text=content)], stop_reason="stop")
            )
    return messages
