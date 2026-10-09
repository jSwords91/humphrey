"""Unit tests for tool-history repair and policy hooks."""

from __future__ import annotations

import pytest

from brain.events import MessageEndEvent, ToolExecutionEndEvent, ToolExecutionStartEvent
from brain.harness import AgentHarness, AgentHarnessConfig
from brain.messages import (
    INTERRUPTED_TOOL_RESULT,
    AssistantMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from providers.mock import MockProvider
from brain.tool_history import repair_tool_history
from app.tools import TOOLS
from brain.tools import ToolResult


def test_repair_synthesizes_missing_tool_result() -> None:
    messages = (
        UserMessage(content="status"),
        AssistantMessage(
            content=[
                ToolCall(id="c1", name="status", arguments={}),
            ],
            stop_reason="tool_use",
        ),
    )
    repaired = repair_tool_history(messages)
    assert repaired.changed
    assert repaired.synthesized_results == 1
    assert isinstance(repaired.messages[-1], ToolResultMessage)
    assert repaired.messages[-1].is_error
    assert repaired.messages[-1].text == INTERRUPTED_TOOL_RESULT


def test_repair_drops_orphan_results() -> None:
    messages = (
        UserMessage(content="hi"),
        ToolResultMessage(
            tool_call_id="missing",
            tool_name="status",
            content=[TextContent(text="orphan")],
        ),
    )
    repaired = repair_tool_history(messages)
    assert repaired.dropped_orphan_results == 1
    assert all(not isinstance(m, ToolResultMessage) for m in repaired.messages)


@pytest.mark.asyncio
async def test_before_tool_call_blocks_execution() -> None:
    async def block(_call: ToolCall) -> tuple[bool, str | None]:
        return True, "not allowed in test"

    harness = AgentHarness(
        AgentHarnessConfig(
            provider=MockProvider(),
            system="test",
            tools=list(TOOLS),
            before_tool_call=block,
        )
    )
    events = [e async for e in harness.prompt("status")]
    assert any(isinstance(e, ToolExecutionStartEvent) for e in events)
    ends = [e for e in events if isinstance(e, ToolExecutionEndEvent)]
    assert ends[0].is_error is True
    assert "not allowed" in ends[0].result.text


@pytest.mark.asyncio
async def test_cancel_stops_stream_and_emits_agent_end() -> None:
    harness = AgentHarness(AgentHarnessConfig(provider=MockProvider(), system="test", tools=list(TOOLS)))
    events = []
    # Avoid the substring "tool" — mock keyword match is naive.
    stream = harness.prompt("greetings from the test")
    async for event in stream:
        events.append(event)
        if event.type == "message_update":
            harness.cancel()
    types = [e.type for e in events]
    assert types[0] == "agent_start"
    assert types[-1] == "agent_end"
    assert not any(isinstance(e, ToolExecutionStartEvent) for e in events)
    aborted = [
        e
        for e in events
        if isinstance(e, MessageEndEvent)
        and isinstance(e.message, AssistantMessage)
        and e.message.stop_reason == "aborted"
    ]
    assert aborted, "expected a synthetic aborted AssistantMessage, tau_agent style"


@pytest.mark.asyncio
async def test_listener_receives_events() -> None:
    seen: list[str] = []

    def on_event(event: object) -> None:
        seen.append(getattr(event, "type"))

    harness = AgentHarness(AgentHarnessConfig(provider=MockProvider(), system="test", tools=list(TOOLS)))
    harness.subscribe(on_event)
    async for _ in harness.prompt("hello"):
        pass
    assert seen[0] == "agent_start"
    assert seen[-1] == "agent_end"


@pytest.mark.asyncio
async def test_listener_exception_does_not_kill_run() -> None:
    def bad(_event: object) -> None:
        raise RuntimeError("boom")

    harness = AgentHarness(AgentHarnessConfig(provider=MockProvider(), system="test", tools=list(TOOLS)))
    harness.subscribe(bad)
    events = [e async for e in harness.prompt("hello")]
    assert events[-1].type == "agent_end"


@pytest.mark.asyncio
async def test_unknown_tool_returns_error_result() -> None:
    from brain.loop import _execute_tool_call

    events = [
        e
        async for e in _execute_tool_call(
            call=ToolCall(id="x", name="does_not_exist", arguments={}),
            tool_by_name={},
            signal=None,
            before_tool_call=None,
            after_tool_call=None,
        )
    ]
    ends = [e for e in events if isinstance(e, ToolExecutionEndEvent)]
    assert ends[-1].is_error is True


@pytest.mark.asyncio
async def test_harness_appends_interrupted_results() -> None:
    harness = AgentHarness(AgentHarnessConfig(provider=MockProvider(), system="test", tools=[]))
    harness.replace_messages(
        [
            AssistantMessage(
                content=[ToolCall(id="c9", name="status", arguments={})],
                stop_reason="tool_use",
            )
        ]
    )
    n = harness.append_interrupted_tool_results()
    assert n == 1
    last = harness.messages[-1]
    assert isinstance(last, ToolResultMessage)
    assert last.is_error


def test_tool_result_helpers() -> None:
    ok = ToolResult.ok({"a": 1})
    assert ok.is_error is False
    assert '"a"' in ok.text
    err = ToolResult.error("nope")
    assert err.is_error is True


class _FakeReasoningProvider:
    """Emits a thinking delta, a text delta, then completes — no tool calls."""

    deployment_name = "fake-reasoning-model"

    async def stream(self, messages, tools, *, system, signal=None):
        from brain.provider import ModelChunk

        yield ModelChunk(thinking="considering the question")
        yield ModelChunk(content="the answer")
        yield ModelChunk(thinking_signature='{"type": "reasoning", "id": "rs_1"}')


@pytest.mark.asyncio
async def test_thinking_content_flows_through_loop_and_message() -> None:
    from brain.events import MessageUpdateEvent
    from brain.messages import ThinkingContent

    harness = AgentHarness(AgentHarnessConfig(provider=_FakeReasoningProvider(), system="test", tools=[]))
    events = [e async for e in harness.prompt("what should I do?")]

    thinking_updates = [
        e for e in events if isinstance(e, MessageUpdateEvent) and e.channel == "thinking"
    ]
    assert thinking_updates
    assert thinking_updates[0].delta == "considering the question"

    text_updates = [
        e for e in events if isinstance(e, MessageUpdateEvent) and e.channel == "text"
    ]
    assert text_updates and text_updates[0].delta == "the answer"

    final = next(
        m
        for m in harness.messages
        if isinstance(m, AssistantMessage) and m.stop_reason != "error"
    )
    assert isinstance(final.content[0], ThinkingContent)
    assert final.thinking_text == "considering the question"
    assert final.thinking_signature == '{"type": "reasoning", "id": "rs_1"}'
    assert final.text == "the answer"


def test_sse_renderer_flattens_thinking_channel_to_its_own_wire_type() -> None:
    from brain.events import MessageUpdateEvent
    from brain.messages import AssistantMessage as _AssistantMessage
    from app.rendering.sse import SseRenderer

    renderer = SseRenderer(session_id=None)
    partial = _AssistantMessage(model="m")

    text_wire = renderer.render(MessageUpdateEvent(message=partial, delta="hi", channel="text"))
    assert text_wire == {"type": "message_delta", "text": "hi"}

    thinking_wire = renderer.render(
        MessageUpdateEvent(message=partial, delta="hmm", channel="thinking")
    )
    assert thinking_wire == {"type": "thinking_delta", "text": "hmm"}
