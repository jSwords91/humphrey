"""Steer / follow-up queue behaviour and usage events."""

from __future__ import annotations

import pytest

from brain.events import MessageEndEvent, ToolExecutionStartEvent
from brain.harness import AgentHarness, AgentHarnessConfig
from brain.messages import AssistantMessage, UserMessage
from providers.mock import MockProvider
from app.tools import TOOLS


def _user_message_events(events: list) -> list[MessageEndEvent]:
    return [
        e for e in events if isinstance(e, MessageEndEvent) and isinstance(e.message, UserMessage)
    ]


@pytest.mark.asyncio
async def test_follow_up_drains_after_settle() -> None:
    harness = AgentHarness(AgentHarnessConfig(provider=MockProvider(), system="test", tools=list(TOOLS)))
    harness.follow_up("greetings from the test again")
    events = [e async for e in harness.prompt("greetings from the test")]

    assert _user_message_events(events)
    assert events[-1].type == "agent_end"
    users = [m for m in harness.messages if isinstance(m, UserMessage)]
    assert len(users) >= 2
    assert users[-1].content == "greetings from the test again"


@pytest.mark.asyncio
async def test_steer_injects_after_tool_turn() -> None:
    harness = AgentHarness(AgentHarnessConfig(provider=MockProvider(), system="test", tools=list(TOOLS)))
    events = []
    stream = harness.prompt("list files")
    async for event in stream:
        events.append(event)
        if isinstance(event, ToolExecutionStartEvent) and harness.queued_messages.count == 0:
            harness.steer("greetings from the test")

    assert any("greetings" in e.message.text for e in _user_message_events(events))
    users = [m for m in harness.messages if isinstance(m, UserMessage)]
    assert any("greetings" in m.content for m in users)


@pytest.mark.asyncio
async def test_message_end_includes_usage_from_mock() -> None:
    harness = AgentHarness(AgentHarnessConfig(provider=MockProvider(), system="test", tools=list(TOOLS)))
    events = [e async for e in harness.prompt("greetings from the test")]
    ends = [
        e
        for e in events
        if isinstance(e, MessageEndEvent) and isinstance(e.message, AssistantMessage) and e.message.usage
    ]
    assert ends
    assert ends[-1].message.usage.get("total_tokens", 0) > 0


@pytest.mark.asyncio
async def test_http_steer_requires_active_run(client) -> None:
    res = await client.post(
        "/api/sessions/abcd1234efgh5678/steer",
        json={"content": "nudge"},
    )
    assert res.status_code == 409
