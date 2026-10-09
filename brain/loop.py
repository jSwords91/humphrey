"""Pure provider/tool agent loop — no HTTP, sessions, rendering, or UI.

Mirrors tau_agent.loop's shape: prompts/steering/follow-ups enter the
transcript as MessageStart/MessageEnd pairs around a UserMessage, and every
failure (cancelled, max-turns, provider exception) becomes a synthetic
AssistantMessage with stop_reason "aborted"/"error" wrapped the same way —
there is no separate "user_message" or "error" event type, matching tau's
event union exactly. Label/wire-format concerns (e.g. "Reading {path}" tool
labels) live in the SSE renderer, not here.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence

from brain.cancel import CancellationToken
from brain.events import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from brain.messages import (
    AgentMessage,
    AssistantMessage,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    assistant_from_stream,
)
from brain.provider import ModelProvider
from brain.tool_history import repair_tool_history
from brain.tools import AgentTool, ToolResult

BeforeToolCall = Callable[[ToolCall], Awaitable[tuple[bool, str | None]]]
AfterToolCall = Callable[
    [ToolCall, ToolResult, bool],
    Awaitable[tuple[ToolResult, bool]],
]
GetQueuedMessages = Callable[[], Sequence[AgentMessage]]


def accumulate_tool_call(
    bucket: dict[int, dict[str, str]],
    index: int,
    *,
    call_id: str | None,
    name: str | None,
    arguments: str | None,
) -> None:
    """Fold one ToolCallDelta into the running per-index accumulation.

    Generic over any provider's streaming shape (only needs index/id/name/
    arguments deltas), which is why this lives with the loop that consumes
    ModelChunk.tool_calls directly, not in an OpenAI-specific adapter -
    moved here from app/openai_adapter.py during the brain/providers/app
    reorg, since loop.py was its only caller.
    """
    entry = bucket.setdefault(index, {"id": "", "name": "", "arguments": ""})
    if call_id:
        entry["id"] = call_id
    if name:
        entry["name"] = name
    if arguments:
        entry["arguments"] += arguments


def finalize_tool_calls(bucket: dict[int, dict[str, str]]) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for idx, data in sorted(bucket.items()):
        raw = data["arguments"] or "{}"
        try:
            args_obj: object = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            args_obj = {"_raw": raw}
        if not isinstance(args_obj, dict):
            args_obj = {"value": args_obj}
        calls.append(
            ToolCall(
                id=data["id"] or f"call_{idx}",
                name=data["name"] or "unknown",
                arguments=args_obj,
            )
        )
    return calls


def _provider_context(messages: list[AgentMessage]) -> list[AgentMessage]:
    """Replayable context: drop empty aborted/error turns, then repair tools."""
    replayable = tuple(
        message
        for message in messages
        if not (
            isinstance(message, AssistantMessage)
            and message.stop_reason in {"error", "aborted"}
            and not message.content
        )
    )
    return list(repair_tool_history(replayable).messages)


def _failure_message(
    model: str,
    detail: str,
    *,
    stop_reason: str = "error",
    text_parts: list[str] | None = None,
    usage: dict[str, int] | None = None,
) -> AssistantMessage:
    content = [TextContent(text="".join(text_parts))] if text_parts else []
    return AssistantMessage(
        content=content,
        model=model,
        stop_reason=stop_reason,  # type: ignore[arg-type]
        error_message=detail,
        usage=dict(usage or {}),
    )


async def run_agent_loop(
    *,
    provider: ModelProvider,
    system: str,
    messages: list[AgentMessage],
    tools: list[AgentTool],
    prompts: Sequence[AgentMessage] = (),
    prelude_messages: Sequence[AgentMessage] = (),
    max_turns: int = 10,
    signal: CancellationToken | None = None,
    get_steering_messages: GetQueuedMessages | None = None,
    get_follow_up_messages: GetQueuedMessages | None = None,
    before_tool_call: BeforeToolCall | None = None,
    after_tool_call: AfterToolCall | None = None,
) -> AsyncIterator[AgentEvent]:
    """
    Run the model/tool loop and yield typed events (tau_agent.loop shape).

    Mutates ``messages`` in place with prompts, assistant turns, and tool
    results. Steering messages are drained between tool turns; follow-ups
    are drained after the agent would otherwise settle.
    """
    new_messages: list[AgentMessage] = list(prompts)
    if prompts:
        messages.extend(prompts)

    yield AgentStartEvent()
    yield TurnStartEvent()
    for message in prelude_messages:
        yield MessageStartEvent(message=message)
        yield MessageEndEvent(message=message)
    for prompt in prompts:
        yield MessageStartEvent(message=prompt)
        yield MessageEndEvent(message=prompt)

    if max_turns < 1:
        error = _failure_message(provider.deployment_name, "max_turns must be at least 1")
        messages.append(error)
        new_messages.append(error)
        yield MessageStartEvent(message=error)
        yield MessageEndEvent(message=error)
        yield TurnEndEvent(message=error)
        yield AgentEndEvent(messages=new_messages)
        return

    tool_by_name = {tool.name: tool for tool in tools}
    turn = 1
    first_turn = True
    pending: tuple[AgentMessage, ...] = tuple(
        get_steering_messages() if get_steering_messages else ()
    )

    try:
        while True:
            has_more_tools = True
            while has_more_tools or pending:
                if not first_turn:
                    yield TurnStartEvent()
                first_turn = False

                for message in pending:
                    messages.append(message)
                    new_messages.append(message)
                    yield MessageStartEvent(message=message)
                    yield MessageEndEvent(message=message)
                pending = ()

                if turn > max_turns:
                    error = _failure_message(
                        provider.deployment_name,
                        f"Stopped after {max_turns} tool turns to prevent loops.",
                    )
                    messages.append(error)
                    new_messages.append(error)
                    yield MessageStartEvent(message=error)
                    yield MessageEndEvent(message=error)
                    yield TurnEndEvent(message=error)
                    return  # `finally` below yields AgentEndEvent

                tool_bucket: dict[int, dict[str, str]] = {}
                text_parts: list[str] = []
                thinking_parts: list[str] = []
                thinking_signature: str | None = None
                usage: dict[str, int] = {}
                aborted = False

                yield MessageStartEvent(
                    message=AssistantMessage(model=provider.deployment_name)
                )

                context = _provider_context(messages)
                async for chunk in provider.stream(
                    context,
                    tools,
                    system=system,
                    signal=signal,
                ):
                    if signal is not None and signal.is_cancelled():
                        aborted = True
                        break
                    if chunk.usage:
                        usage.update(chunk.usage)
                    if chunk.thinking_signature:
                        thinking_signature = chunk.thinking_signature
                    if chunk.thinking:
                        thinking_parts.append(chunk.thinking)
                        yield MessageUpdateEvent(
                            message=AssistantMessage(
                                content=[ThinkingContent(thinking="".join(thinking_parts))],
                                model=provider.deployment_name,
                            ),
                            delta=chunk.thinking,
                            channel="thinking",
                        )
                    if chunk.content:
                        text_parts.append(chunk.content)
                        yield MessageUpdateEvent(
                            message=AssistantMessage(
                                content=[TextContent(text="".join(text_parts))],
                                model=provider.deployment_name,
                            ),
                            delta=chunk.content,
                        )
                    for delta in chunk.tool_calls:
                        accumulate_tool_call(
                            tool_bucket,
                            delta.index,
                            call_id=delta.id,
                            name=delta.name,
                            arguments=delta.arguments,
                        )

                if signal is not None and signal.is_cancelled():
                    aborted = True

                if aborted:
                    assistant = _failure_message(
                        provider.deployment_name,
                        "Run cancelled",
                        stop_reason="aborted",
                        text_parts=text_parts,
                        usage=usage,
                    )
                    messages.append(assistant)
                    new_messages.append(assistant)
                    yield MessageEndEvent(message=assistant)
                    yield TurnEndEvent(message=assistant)
                    return

                tool_calls = finalize_tool_calls(tool_bucket)
                assistant = assistant_from_stream(
                    text="".join(text_parts),
                    tool_calls=tool_calls,
                    model=provider.deployment_name,
                    usage=usage,
                    thinking="".join(thinking_parts) if thinking_parts else None,
                    thinking_signature=thinking_signature,
                )
                messages.append(assistant)
                new_messages.append(assistant)
                yield MessageEndEvent(message=assistant)

                if not tool_calls:
                    yield TurnEndEvent(message=assistant)
                    has_more_tools = False
                else:
                    tool_results: list[ToolResultMessage] = []
                    for call in tool_calls:
                        async for event in _execute_tool_call(
                            call=call,
                            tool_by_name=tool_by_name,
                            signal=signal,
                            before_tool_call=before_tool_call,
                            after_tool_call=after_tool_call,
                        ):
                            yield event
                            if isinstance(event, MessageEndEvent) and isinstance(
                                event.message, ToolResultMessage
                            ):
                                messages.append(event.message)
                                new_messages.append(event.message)
                                tool_results.append(event.message)

                    yield TurnEndEvent(message=assistant, tool_results=tool_results)
                    has_more_tools = True

                turn += 1
                pending = tuple(get_steering_messages() if get_steering_messages else ())

            follow_ups = tuple(get_follow_up_messages() if get_follow_up_messages else ())
            if follow_ups:
                pending = follow_ups
                continue
            break
    except Exception as exc:  # noqa: BLE001 — surface to listeners as a failed message
        error = _failure_message(provider.deployment_name, str(exc))
        messages.append(error)
        new_messages.append(error)
        yield MessageStartEvent(message=error)
        yield MessageEndEvent(message=error)
    finally:
        yield AgentEndEvent(messages=new_messages)


async def _execute_tool_call(
    *,
    call: ToolCall,
    tool_by_name: dict[str, AgentTool],
    signal: CancellationToken | None,
    before_tool_call: BeforeToolCall | None,
    after_tool_call: AfterToolCall | None,
) -> AsyncIterator[AgentEvent]:
    yield ToolExecutionStartEvent(
        tool_call_id=call.id,
        tool_name=call.name,
        args=call.arguments,
    )

    blocked = False
    block_reason: str | None = None
    if before_tool_call is not None:
        blocked, block_reason = await before_tool_call(call)

    if blocked:
        result = ToolResult.error(block_reason or "Tool execution was blocked")
        is_error = True
    elif signal is not None and signal.is_cancelled():
        result = ToolResult.error("Operation aborted")
        is_error = True
    else:
        tool = tool_by_name.get(call.name)
        if tool is None:
            result = ToolResult.error(f"Tool {call.name} not found")
            is_error = True
        else:
            try:
                result = await tool.execute(call.id, call.arguments, signal, None)
                is_error = result.is_error
            except Exception as exc:  # noqa: BLE001 — tools are an isolation boundary
                result = ToolResult.error(str(exc))
                is_error = True

    if after_tool_call is not None:
        result, is_error = await after_tool_call(call, result, is_error)

    yield ToolExecutionEndEvent(
        tool_call_id=call.id,
        tool_name=call.name,
        result=result,
        is_error=is_error,
    )
    message = ToolResultMessage(
        tool_call_id=call.id,
        tool_name=call.name,
        content=list(result.content),
        details=result.details,
        is_error=is_error,
    )
    yield MessageStartEvent(message=message)
    yield MessageEndEvent(message=message)
