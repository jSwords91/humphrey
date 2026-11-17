"""Reading a finished Humphrey run: the reply, the tool calls, how it ended, the tokens.

agentic-evals reads these from its Session, the append-only log. Humphrey's harness keeps its
own messages instead (brain/harness.py), so this reads them from there: the same messages
production stores in a session.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from brain.loop import MAX_TURNS_PREFIX
from brain.messages import AgentMessage, AssistantMessage, ToolResultMessage
from brain.messages import ToolCall as CallBlock
from brain.tools import AgentTool


@dataclass(frozen=True)
class ToolCall:
    """One tool run: what was asked for and what came back."""

    name: str
    input: dict[str, Any]
    output: str
    is_error: bool


def tool_log(messages: Sequence[AgentMessage]) -> list[ToolCall]:
    results = {m.tool_call_id: m for m in messages if isinstance(m, ToolResultMessage)}
    log = []
    for message in messages:
        if not isinstance(message, AssistantMessage):
            continue
        for block in message.content:
            if isinstance(block, CallBlock):
                result = results.get(block.id)
                log.append(
                    ToolCall(
                        block.name,
                        dict(block.arguments),
                        result.text if result else "(no result)",
                        result.is_error if result else True,
                    )
                )
    return log


def last_assistant(messages: Sequence[AgentMessage]) -> AssistantMessage | None:
    return next((m for m in reversed(messages) if isinstance(m, AssistantMessage)), None)


def reply_text(messages: Sequence[AgentMessage]) -> str:
    last = last_assistant(messages)
    return last.text if last else ""


def stop_reason(messages: Sequence[AgentMessage]) -> str:
    """How the run ended: stop, max_turns, error (the provider), aborted, length."""
    last = last_assistant(messages)
    if last is None:
        return "error"
    if last.stop_reason == "error" and (last.error_message or "").startswith(MAX_TURNS_PREFIX):
        return "max_turns"
    return last.stop_reason


def tokens(messages: Sequence[AgentMessage]) -> dict[str, int]:
    total = {"input": 0, "output": 0, "cached_input": 0}
    calls = 0
    for m in messages:
        if isinstance(m, AssistantMessage) and m.usage:
            calls += 1
            for key in total:
                total[key] += int(m.usage.get(key, 0) or 0)
    return {**total, "model_calls": calls}


async def execute(
    tools: Sequence[AgentTool], name: str, args: dict[str, Any], call_id: str
) -> ToolCall:
    """Run one tool call outside the loop, for reference solutions."""
    tool = next((t for t in tools if t.name == name), None)
    if tool is None:
        return ToolCall(name, args, f"no tool called {name}", True)
    result = await tool.execute(call_id, args)
    return ToolCall(name, args, result.text, result.is_error)
