"""Responses API support.

Reasoning models (gpt-5.4/5.5/5.6 and later) reject function tools on
/v1/chat/completions unless reasoning_effort is "none" — confirmed identical
on OpenAI-direct and Azure OpenAI. The only way to combine reasoning with
tools is the Responses API. This module holds everything both providers
share: which models need it, how to build a request, and how to parse the
SSE stream — scaled down from tau's version (user-only images, since our
tools never produce image tool results unlike tau's; no prompt-cache-key,
no multi-vendor compat flag, no codex handling; Humphrey has one message
shape and two providers, not tau's config-driven N).

Tool-call items from the Responses API are deliberately mapped onto the same
``ToolCallDelta`` shape the existing Chat Completions path already produces,
so brain/loop.py's accumulate_tool_call/finalize_tool_calls need no changes
at all — only the new ``thinking``/``thinking_signature`` fields on
``ModelChunk`` are genuinely new to callers.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from brain.messages import (
    AgentMessage,
    AssistantMessage,
    ToolResultMessage,
    UserMessage,
)
from brain.provider import ModelChunk, ToolCallDelta
from brain.tools import AgentTool
from providers.content import text_and_images


def _user_content_to_responses(content: str | list) -> Any:
    """Responses-API ``input_image`` blocks (tau_ai/openai_compatible.py's
    ``_messages_to_responses_input``, scaled to Humphrey's user-only images).
    """
    text, images = text_and_images(content)
    if not images:
        return text
    parts: list[dict[str, Any]] = []
    if text:
        parts.append({"type": "input_text", "text": text})
    parts.extend(
        {"type": "input_image", "detail": "auto", "image_url": f"data:{img.mime_type};base64,{img.data}"}
        for img in images
    )
    return parts

# Tau's own list ("gpt-5.4", "gpt-5.5") predates gpt-5.6 and is already
# stale — this is deliberately an explicit set we maintain, not a prefix
# guess, so the exact model we're targeting is never silently missed.
_RESPONSES_ONLY_MODELS: frozenset[str] = frozenset({"gpt-5.4", "gpt-5.5", "gpt-5.6"})


def uses_responses_api(model: str) -> bool:
    """Return whether ``model`` must be served over /v1/responses, not /v1/chat/completions."""
    normalized = model.strip().lower()
    return any(normalized == m or normalized.startswith(f"{m}-") for m in _RESPONSES_ONLY_MODELS)


def build_responses_payload(
    *,
    model: str,
    system: str,
    messages: list[AgentMessage],
    tools: list[AgentTool],
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "stream": True,
        # Stateless: the full transcript is resent every turn, so there is
        # no need for (and no risk from) server-side retention.
        "store": False,
        "instructions": system,
        "input": messages_to_responses_input(messages),
    }
    if reasoning_effort:
        # summary: "auto" streams response.reasoning_summary_text.delta
        # events so the agent's reasoning is visible, not just billed.
        payload["reasoning"] = {"effort": reasoning_effort, "summary": "auto"}
    if tools:
        payload["tools"] = [_tool_to_responses(tool) for tool in tools]
    return payload


def _tool_to_responses(tool: AgentTool) -> dict[str, Any]:
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": dict(tool.openai_parameters()),
    }


# Fields the Responses API accepts on a *replayed* reasoning input item.
# Confirmed against the openai SDK's own ResponseReasoningItemParam schema —
# "status" is populated only on items the API *returns*, not ones you send
# back, and re-sending it fails with "Unknown parameter: 'input[N].status'"
# (caught live, against gpt-5.6-sol, in this session's testing).
_REASONING_REPLAY_KEYS = frozenset({"id", "summary", "type", "content", "encrypted_content"})


def messages_to_responses_input(messages: list[AgentMessage]) -> list[dict[str, Any]]:
    """Map the portable transcript to Responses API ``input`` items.

    Replays each assistant turn's reasoning item (from
    ``ThinkingContent.thinking_signature``) ahead of its text/tool calls —
    required so a reasoning model continues its line of reasoning instead of
    starting over on every turn (Azure/OpenAI's own guidance).
    """
    items: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, UserMessage):
            items.append({"role": "user", "content": _user_content_to_responses(message.content)})
        elif isinstance(message, AssistantMessage):
            signature = message.thinking_signature
            if signature:
                try:
                    reasoning_item = json.loads(signature)
                except (TypeError, ValueError):
                    reasoning_item = None
                if isinstance(reasoning_item, dict):
                    items.append(
                        {k: v for k, v in reasoning_item.items() if k in _REASONING_REPLAY_KEYS}
                    )
            if message.text:
                items.append({"role": "assistant", "content": message.text})
            for call in message.tool_calls:
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call.id,
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    }
                )
        elif isinstance(message, ToolResultMessage):
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": message.text or "(no tool output)",
                }
            )
    return items


class ResponsesStreamParser:
    """Stateful parser for one /v1/responses SSE stream.

    Feed each decoded event dict via ``feed()``; each call returns the
    ``ModelChunk`` to yield, or ``None`` for events that only update internal
    bookkeeping (tool-call registration, reasoning-item tracking).
    """

    def __init__(self) -> None:
        self._tool_index_by_item_id: dict[str, int] = {}
        self._tool_has_delta: set[str] = set()
        self._reasoning_items: dict[str, dict[str, Any]] = {}
        self._next_index = 0

    def _index_for(self, item_id: str) -> int:
        if item_id not in self._tool_index_by_item_id:
            self._tool_index_by_item_id[item_id] = self._next_index
            self._next_index += 1
        return self._tool_index_by_item_id[item_id]

    def feed(self, event: Mapping[str, Any]) -> ModelChunk | None:  # noqa: PLR0911
        event_type = event.get("type")
        if not isinstance(event_type, str):
            return None

        if event_type == "response.output_text.delta":
            delta = event.get("delta")
            return ModelChunk(content=delta) if isinstance(delta, str) and delta else None

        if event_type in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
            delta = event.get("delta")
            return ModelChunk(thinking=delta) if isinstance(delta, str) and delta else None

        if event_type == "response.reasoning_summary_part.done":
            return ModelChunk(thinking="\n\n")

        if event_type == "response.output_item.added":
            return self._register_item(event.get("item"))

        if event_type == "response.function_call_arguments.delta":
            item_id = event.get("item_id")
            delta = event.get("delta")
            if isinstance(item_id, str) and isinstance(delta, str) and delta:
                self._tool_has_delta.add(item_id)
                return ModelChunk(
                    tool_calls=[ToolCallDelta(index=self._index_for(item_id), arguments=delta)]
                )
            return None

        if event_type == "response.output_item.done":
            item = event.get("item")
            if isinstance(item, Mapping) and item.get("type") == "reasoning":
                self._register_reasoning_item(item)
                return None
            if isinstance(item, Mapping) and item.get("type") == "function_call":
                item_id = item.get("id")
                # Only emit final arguments here if no delta ever arrived for
                # this item — otherwise this would duplicate what's already
                # been accumulated from response.function_call_arguments.delta.
                if isinstance(item_id, str) and item_id not in self._tool_has_delta:
                    arguments = item.get("arguments")
                    if isinstance(arguments, str) and arguments:
                        return ModelChunk(
                            tool_calls=[
                                ToolCallDelta(index=self._index_for(item_id), arguments=arguments)
                            ]
                        )
            return None

        if event_type in ("response.completed", "response.incomplete"):
            usage = self._extract_usage(event)
            signature = self._first_reasoning_signature()
            if usage or signature:
                return ModelChunk(usage=usage, thinking_signature=signature)
            return None

        if event_type == "response.failed":
            raise RuntimeError(self._failure_message(event))

        if event_type == "error":
            message = event.get("message")
            raise RuntimeError(message if isinstance(message, str) else "Responses API error")

        return None

    def _register_item(self, item: object) -> ModelChunk | None:
        if not isinstance(item, Mapping):
            return None
        if item.get("type") == "reasoning":
            # Registered on both added and done — some streams may not repeat
            # the item at .done, so don't rely on .done alone.
            self._register_reasoning_item(item)
            return None
        if item.get("type") == "function_call":
            item_id = item.get("id")
            call_id = item.get("call_id")
            name = item.get("name")
            if isinstance(item_id, str):
                index = self._index_for(item_id)
                return ModelChunk(
                    tool_calls=[
                        ToolCallDelta(
                            index=index,
                            id=call_id if isinstance(call_id, str) else None,
                            name=name if isinstance(name, str) else None,
                        )
                    ]
                )
        return None

    def _register_reasoning_item(self, item: Mapping[str, Any]) -> None:
        item_id = item.get("id")
        if isinstance(item_id, str):
            self._reasoning_items[item_id] = dict(item)

    def _first_reasoning_signature(self) -> str | None:
        if not self._reasoning_items:
            return None
        first = next(iter(self._reasoning_items.values()))
        return json.dumps(first, ensure_ascii=False)

    @staticmethod
    def _extract_usage(event: Mapping[str, Any]) -> dict[str, int] | None:
        response = event.get("response")
        if not isinstance(response, Mapping):
            return None
        usage = response.get("usage")
        if not isinstance(usage, Mapping):
            return None
        return {
            "input": int(usage.get("input_tokens") or 0),
            "output": int(usage.get("output_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
        }

    @staticmethod
    def _failure_message(event: Mapping[str, Any]) -> str:
        response = event.get("response")
        if isinstance(response, Mapping):
            error = response.get("error")
            if isinstance(error, Mapping):
                message = error.get("message")
                if isinstance(message, str) and message:
                    return message
        return "Responses API request failed"
