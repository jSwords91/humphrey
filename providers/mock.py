"""Offline model provider for local UI/API work without a model provider."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from brain.cancel import CancellationToken
from brain.messages import AgentMessage, ToolResultMessage, UserMessage
from brain.provider import ModelChunk, ToolCallDelta
from brain.tools import AgentTool


def _pick_tool(latest: str) -> tuple[str, dict[str, Any]] | None:
    lower = latest.lower()
    if any(t in lower for t in ("architecture", "diagram", "how is this wired", "how it flows")):
        return "read_skill", {"name": "mermaid-diagrams"}
    if any(t in lower for t in ("status", "health", "deployment", "config", "environment")):
        return "status", {}
    if any(t in lower for t in ("calculate", "compute")):
        return "calculate", {"expression": "(3 + 4) * 2"}
    if any(t in lower for t in ("storage", "skill", "how do")):
        return "search_docs", {"query": latest[:80]}
    if any(t in lower for t in ("capabilit", "tool", "what can you", "help")):
        return "capabilities", {}
    if any(t in lower for t in ("doc", "readme", "knowledge", "platform")):
        return "list_docs", {}
    return None


class MockProvider:
    """
    Deterministic stand-in for a real model provider.

    Emits the same ModelChunk stream shape so the harness stays provider-agnostic.
    Tool execution still uses real tool handlers.
    """

    @property
    def deployment_name(self) -> str:
        return "mock"

    def _latest_user_message(self, messages: list[AgentMessage]) -> UserMessage | None:
        for message in reversed(messages):
            if isinstance(message, UserMessage):
                return message
        return None

    def _latest_user(self, messages: list[AgentMessage]) -> str:
        message = self._latest_user_message(messages)
        return message.text if message else ""

    def _awaiting_tool_followup(self, messages: list[AgentMessage]) -> bool:
        """True when the latest user turn already has tool results (final prose next)."""
        last_user = -1
        for i, message in enumerate(messages):
            if isinstance(message, UserMessage):
                last_user = i
        if last_user < 0:
            return False
        return any(isinstance(m, ToolResultMessage) for m in messages[last_user + 1 :])

    async def stream(
        self,
        messages: list[AgentMessage],
        tools: list[AgentTool],
        *,
        system: str,
        signal: CancellationToken | None = None,
    ) -> AsyncIterator[ModelChunk]:
        _ = system
        latest = self._latest_user(messages)
        picked = _pick_tool(latest)
        awaiting_followup = self._awaiting_tool_followup(messages)

        if signal is not None and signal.is_cancelled():
            return

        if picked and not awaiting_followup:
            name, args = picked
            call_id = f"mock_{uuid.uuid4().hex[:8]}"
            yield ModelChunk(
                tool_calls=[
                    ToolCallDelta(
                        index=0,
                        id=call_id,
                        name=name,
                        arguments=json.dumps(args),
                    )
                ]
            )
            return

        if picked and awaiting_followup:
            name, _ = picked
            text = (
                "Offline mock provider — no model was called.\n\n"
                f"I ran `{name}` via the real tool handlers.\n\n"
                f"Available tools: {', '.join(t.name for t in tools)}."
            )
        else:
            preview = latest.strip().replace("\n", " ")
            if len(preview) > 100:
                preview = f"{preview[:97]}..."
            latest_message = self._latest_user_message(messages)
            image_note = ""
            if latest_message and latest_message.images:
                count = len(latest_message.images)
                noun = "image" if count == 1 else "images"
                image_note = (
                    f"(Received {count} {noun} — not inspected in offline mock mode.)\n\n"
                )
            text = (
                "Offline mock provider — no model was called.\n\n"
                f"You asked: {preview}\n\n"
                f"{image_note}"
                "Try **capabilities**, **status**, **calculate**, or **list docs**."
            )

        chunk_size = 28
        for i in range(0, len(text), chunk_size):
            if signal is not None and signal.is_cancelled():
                return
            yield ModelChunk(content=text[i : i + chunk_size])
            await asyncio.sleep(0.012)

        # Synthetic usage so the UI/stats path can be exercised offline.
        yield ModelChunk(
            usage={
                "input": max(1, len(latest) // 4),
                "output": max(1, len(text) // 4),
                "total_tokens": max(2, (len(latest) + len(text)) // 4),
            }
        )
