"""Provider-safe repair of malformed tool-call transcripts."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from brain.messages import (
    INTERRUPTED_TOOL_RESULT,
    AgentMessage,
    AssistantMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
)


@dataclass(frozen=True, slots=True)
class ToolHistoryRepair:
    """A replayable transcript plus counters for diagnostics."""

    messages: tuple[AgentMessage, ...]
    changed: bool = False
    synthesized_results: int = 0
    dropped_orphan_results: int = 0
    dropped_duplicate_results: int = 0
    reordered_results: int = 0


def repair_tool_history(messages: tuple[AgentMessage, ...] | list[AgentMessage]) -> ToolHistoryRepair:
    """
    Ensure every assistant tool call has exactly one adjacent result.

    Missing results become a deterministic interruption error. Orphan results
    (no matching call) are dropped. Duplicate results prefer a real result over
    a synthetic interruption.
    """
    sequence = tuple(messages)
    call_occurrences: list[tuple[tuple[int, int], ToolCall, int]] = []
    for message_index, message in enumerate(sequence):
        if not isinstance(message, AssistantMessage):
            continue
        for call_offset, call in enumerate(message.tool_calls, start=1):
            call_occurrences.append(
                ((message_index, call_offset), call, message_index + call_offset)
            )

    results_by_id: dict[str, list[tuple[int, ToolResultMessage]]] = defaultdict(list)
    for message_index, message in enumerate(sequence):
        if isinstance(message, ToolResultMessage):
            results_by_id[message.tool_call_id].append((message_index, message))

    selected_results: dict[tuple[int, int], tuple[int | None, ToolResultMessage]] = {}
    used_result_positions: set[int] = set()

    for occurrence, call, expected_position in call_occurrences:
        if expected_position >= len(sequence):
            continue
        candidate = sequence[expected_position]
        if (
            isinstance(candidate, ToolResultMessage)
            and candidate.tool_call_id == call.id
            and expected_position not in used_result_positions
        ):
            selected_results[occurrence] = (expected_position, candidate)
            used_result_positions.add(expected_position)

    synthesized_results = 0
    for occurrence, call, _expected in call_occurrences:
        if occurrence in selected_results:
            continue
        candidates = [
            candidate
            for candidate in results_by_id.get(call.id, [])
            if candidate[0] not in used_result_positions
        ]
        if candidates:
            after_call = [c for c in candidates if c[0] > occurrence[0]]
            pool = after_call or candidates
            selected = next(
                (c for c in pool if not _is_interruption_result(c[1])),
                pool[0],
            )
            selected_results[occurrence] = selected
            used_result_positions.add(selected[0])
            continue
        selected_results[occurrence] = (
            None,
            ToolResultMessage(
                tool_call_id=call.id,
                tool_name=call.name,
                content=[TextContent(text=INTERRUPTED_TOOL_RESULT)],
                is_error=True,
            ),
        )
        synthesized_results += 1

    for occurrence, call, _expected in call_occurrences:
        selected_position, selected_result = selected_results[occurrence]
        if selected_position is None or not _is_interruption_result(selected_result):
            continue
        replacement = next(
            (
                candidate
                for candidate in results_by_id.get(call.id, [])
                if candidate[0] not in used_result_positions
                and not _is_interruption_result(candidate[1])
            ),
            None,
        )
        if replacement is None:
            continue
        used_result_positions.remove(selected_position)
        used_result_positions.add(replacement[0])
        selected_results[occurrence] = replacement

    repaired: list[AgentMessage] = []
    reordered_results = 0
    for message_index, message in enumerate(sequence):
        if isinstance(message, ToolResultMessage):
            continue
        repaired.append(message)
        if not isinstance(message, AssistantMessage):
            continue
        for call_offset, _call in enumerate(message.tool_calls, start=1):
            result_position, result = selected_results[(message_index, call_offset)]
            repaired.append(result)
            if result_position is not None and result_position != message_index + call_offset:
                reordered_results += 1

    call_ids = {call.id for _occ, call, _exp in call_occurrences}
    unused_results = [
        result
        for results in results_by_id.values()
        for position, result in results
        if position not in used_result_positions
    ]
    orphan_results = sum(result.tool_call_id not in call_ids for result in unused_results)
    dropped_duplicate_results = sum(result.tool_call_id in call_ids for result in unused_results)
    repaired_messages = tuple(repaired)
    return ToolHistoryRepair(
        messages=repaired_messages,
        changed=repaired_messages != sequence,
        synthesized_results=synthesized_results,
        dropped_orphan_results=orphan_results,
        dropped_duplicate_results=dropped_duplicate_results,
        reordered_results=reordered_results,
    )


def _is_interruption_result(message: ToolResultMessage) -> bool:
    return message.is_error and message.text == INTERRUPTED_TOOL_RESULT
