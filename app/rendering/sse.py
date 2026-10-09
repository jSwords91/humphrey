"""Project typed AgentEvents onto the existing flat SSE wire contract.

This is the one place that knows about the browser's field names
(``type``/``text``/``tool``/``call_id``/``result``/...) — mirrors how tau
keeps ``tau_agent.events`` free of rendering concerns and puts frontend-
specific projection in ``tau_coding/rendering/*.py``. The wire shape here is
intentionally unchanged from before this refactor so the existing frontend
and ``test_agent_sse.py`` don't need to change.
"""

from __future__ import annotations

import json
from typing import Any

from app.tools import tool_label
from brain.events import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from brain.messages import AssistantMessage, ToolResultMessage, UserMessage



def encode_sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


MAX_DIFF_CHARS = 20_000


class SseRenderer:
    """Stateful projector for one run's event stream (turn count, tool labels)."""

    def __init__(self, *, session_id: str | None) -> None:
        self._session_id = session_id
        self._turn = 0
        self._labels: dict[str, str] = {}

    def render(self, event: AgentEvent) -> dict[str, Any] | None:
        payload = self._flatten(event)
        if payload is None:
            return None
        if self._session_id is not None:
            payload.setdefault("session_id", self._session_id)
        return {k: v for k, v in payload.items() if v is not None}

    def to_sse(self, event: AgentEvent) -> str | None:
        payload = self.render(event)
        if payload is None:
            return None
        return encode_sse(payload)

    def _flatten(self, event: AgentEvent) -> dict[str, Any] | None:
        if isinstance(event, AgentStartEvent):
            return {"type": "agent_start"}
        if isinstance(event, AgentEndEvent):
            return {"type": "agent_end"}
        if isinstance(event, TurnStartEvent):
            self._turn += 1
            return {"type": "turn_start", "turn": self._turn}
        if isinstance(event, TurnEndEvent):
            return {"type": "turn_end"}
        if isinstance(event, MessageStartEvent):
            return self._message_start(event)
        if isinstance(event, MessageUpdateEvent):
            wire_type = "thinking_delta" if event.channel == "thinking" else "message_delta"
            return {"type": wire_type, "text": event.delta}
        if isinstance(event, MessageEndEvent):
            return self._message_end(event)
        if isinstance(event, ToolExecutionStartEvent):
            label = tool_label(event.tool_name, event.args)
            self._labels[event.tool_call_id] = label
            return {
                "type": "tool_start",
                "turn": self._turn,
                "tool": event.tool_name,
                "label": label,
                "call_id": event.tool_call_id,
                "args": event.args,
            }
        if isinstance(event, ToolExecutionUpdateEvent):
            return None  # nothing streams partial tool results today
        if isinstance(event, ToolExecutionEndEvent):
            payload: dict[str, Any] = {
                "type": "tool_end",
                "tool": event.tool_name,
                "label": self._labels.pop(event.tool_call_id, event.tool_name),
                "call_id": event.tool_call_id,
                "result": event.result.text,
                "is_error": event.is_error,
            }
            # The UI shows edits as a diff and bash runs by exit code; both
            # live in the tool's details rather than its model-facing text.
            details = event.result.details
            if isinstance(details, dict):
                patch = details.get("patch")
                if isinstance(patch, str) and patch:
                    payload["diff"] = patch[:MAX_DIFF_CHARS]
                if "exit_code" in details:
                    payload["exit_code"] = details["exit_code"]
            return payload
        return None

    def _message_start(self, event: MessageStartEvent) -> dict[str, Any] | None:
        message = event.message
        if isinstance(message, AssistantMessage):
            if message.stop_reason in {"error", "aborted"}:
                return None  # flattened to "error" on message_end instead
            return {"type": "message_start", "role": "assistant"}
        return None  # user/tool-result starts carry no wire event of their own

    def _message_end(self, event: MessageEndEvent) -> dict[str, Any] | None:
        message = event.message
        if isinstance(message, UserMessage):
            return {"type": "user_message", "text": message.text}
        if isinstance(message, ToolResultMessage):
            return None  # superseded by tool_execution_end
        if isinstance(message, AssistantMessage):
            if message.stop_reason in {"error", "aborted"}:
                return {
                    "type": "error",
                    "detail": message.error_message or "Unknown agent error",
                }
            return {
                "type": "message_end",
                "deployment": message.model,
                "usage": message.usage or None,
            }
        return None
