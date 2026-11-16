"""Offline model provider for local UI/API work without a model provider."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from brain.cancel import CancellationToken
from brain.messages import AgentMessage, ToolResultMessage, UserMessage
from brain.provider import ModelChunk, ToolCallDelta
from brain.tools import AgentTool


_MERMAID_SKILL = Path(__file__).resolve().parent.parent / "skills" / "mermaid-diagrams" / "SKILL.md"


def _pick_tool(latest: str) -> tuple[str, dict[str, Any]] | None:
    lower = latest.lower()
    if any(t in lower for t in ("architecture", "diagram", "how is this wired", "how it flows")):
        return "read", {"path": str(_MERMAID_SKILL)}
    if any(t in lower for t in ("write a note", "save a note", "create a file")):
        return "write", {"path": "notes.md", "content": f"# Note\n\n{latest}\n"}
    if any(t in lower for t in ("list files", "workspace", "what files")):
        return "bash", {"command": "ls -la", "description": "Listing workspace files"}
    if "ask me" in lower:
        return "ask_user_question", {"questions": _MOCK_QUESTIONS}
    return None


# What "ask me" asks offline: one single choice, one multiple choice.
_MOCK_QUESTIONS = [
    {
        "question": "Which months should the sales dashboard cover?",
        "header": "Period",
        "options": [
            {"label": "This year only", "description": "January to October 2026"},
            {
                "label": "Last 12 months",
                "description": "Matches the CSV's full range",
                "recommended": True,
            },
        ],
    },
    {
        "question": "Which charts should the dashboard include?",
        "header": "Charts",
        "multiSelect": True,
        "options": [
            {"label": "Revenue by month", "description": "A line over the period"},
            {"label": "Top products", "description": "The ten best sellers"},
            {"label": "Regions", "description": "Revenue split by region"},
        ],
    },
]


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
        if picked and picked[0] not in {t.name for t in tools}:
            picked = None  # e.g. ask_user_question switched off
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
            result = next(
                (m for m in reversed(messages) if isinstance(m, ToolResultMessage)), None
            )
            if name == "ask_user_question" and result is not None:
                text += f"\n\nIt returned:\n\n{result.text}"
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
                "Try **list files**, **write a note**, **draw a diagram**, or **ask me**."
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
