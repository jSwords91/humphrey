"""Typed agent events — a discriminated union matching tau_agent.events.

The harness/loop only ever produce and consume these 10 event types (tau's
exact set — no extra "user_message" or "error" event: a prompt/steer/follow-up
entering the transcript is a MessageStart/MessageEnd pair around a
UserMessage, and every failure (cancelled, max-turns, provider exception) is a
MessageStart/MessageEnd pair around a synthetic AssistantMessage with
stop_reason="error"/"aborted" — exactly tau's approach). Nothing here knows
about SSE or a particular frontend's field casing; that projection lives in
``app/rendering/sse.py``, mirroring how tau keeps ``tau_agent.events`` free of
``tau_coding.rendering`` concerns.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from brain.messages import AgentMessage, ToolResultMessage, WireModel
from brain.tools import ToolResult


class AgentStartEvent(WireModel):
    type: Literal["agent_start"] = "agent_start"


class AgentEndEvent(WireModel):
    type: Literal["agent_end"] = "agent_end"
    messages: list[AgentMessage] = Field(default_factory=list)


class TurnStartEvent(WireModel):
    type: Literal["turn_start"] = "turn_start"


class TurnEndEvent(WireModel):
    type: Literal["turn_end"] = "turn_end"
    message: AgentMessage
    tool_results: list[ToolResultMessage] = Field(default_factory=list)


class MessageStartEvent(WireModel):
    type: Literal["message_start"] = "message_start"
    message: AgentMessage


class MessageUpdateEvent(WireModel):
    """The in-progress assistant message plus the delta that produced it.

    Tau's equivalent carries a provider sub-event (``AssistantMessageEvent``)
    because ``tau_ai`` models fine-grained per-provider deltas. This harness
    has no such layer (out of scope here), so ``delta`` stands in for that
    sub-event — same position in the union, thinner payload. ``message`` is
    the accumulated-so-far partial, matching tau's ``event.partial``.

    ``channel`` distinguishes a visible-answer delta from a reasoning-model's
    thinking delta (Responses API only) — both stream the same way, but a
    renderer needs to route them to different places in the UI.
    """

    type: Literal["message_update"] = "message_update"
    message: AgentMessage
    delta: str
    channel: Literal["text", "thinking"] = "text"


class MessageEndEvent(WireModel):
    type: Literal["message_end"] = "message_end"
    message: AgentMessage


class ToolExecutionStartEvent(WireModel):
    type: Literal["tool_execution_start"] = "tool_execution_start"
    tool_call_id: str
    tool_name: str
    args: dict[str, object] = Field(default_factory=dict)


class ToolExecutionUpdateEvent(WireModel):
    """Partial tool result. Nothing wires this yet — ``tools.py``'s

    ``on_update`` hook already exists but is called with ``None`` today, so
    no tool streams partial results. Kept for shape parity so adding one
    later doesn't reshape the union.
    """

    type: Literal["tool_execution_update"] = "tool_execution_update"
    tool_call_id: str
    tool_name: str
    args: dict[str, object] = Field(default_factory=dict)
    partial_result: ToolResult


class ToolExecutionEndEvent(WireModel):
    type: Literal["tool_execution_end"] = "tool_execution_end"
    tool_call_id: str
    tool_name: str
    result: ToolResult
    is_error: bool


AgentEvent = Annotated[
    AgentStartEvent
    | AgentEndEvent
    | TurnStartEvent
    | TurnEndEvent
    | MessageStartEvent
    | MessageUpdateEvent
    | MessageEndEvent
    | ToolExecutionStartEvent
    | ToolExecutionUpdateEvent
    | ToolExecutionEndEvent,
    Field(discriminator="type"),
]
