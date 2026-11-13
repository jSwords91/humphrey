"""Context-size estimation and auto-compaction, ported from tau_coding.context_window.

Ported near-directly: the token-estimation math, the auto-compaction
threshold, the "recent preserving" compaction boundary search, and the
structured summarization prompts are all taken essentially verbatim from
tau (adapted only for our message/tool shapes). What's NOT ported, because
our architecture doesn't have the concept:

- Provider-catalog-derived context window sizes (tau's ``RuntimeModelCatalog``)
  — we use one configured constant (``settings.context_window_tokens``)
    instead of per-model lookups across many vendors.
- Dynamically-added tools mid-session (tau's ``added_tool_names`` tracking on
  a tool result) — our TOOLS list is static for the lifetime of a harness,
  so the "trailing added tool tokens" component tau computes doesn't apply.
- Mid-loop overflow recovery (tau catches a context-overflow provider error
  and retries after compacting). We only compact *before* a turn starts
  (see ``compact_if_needed``, called from app/main.py). A turn that
  overflows despite that check will fail normally rather than auto-retry -
  a deliberate scope decision, not an oversight, documented here so it's
  easy to revisit.
- Tau's persisted, branchable "context entries". Instead, the session keeps
  the full transcript plus a list of compaction records (summary and where
  the kept messages start; see app/sessions), and rebuilds the model's
  context from the latest one. The user's history is never rewritten.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from brain.messages import (
    AgentMessage,
    AssistantMessage,
    ThinkingContent,
    ToolResultMessage,
    UserMessage,
)
from brain.provider import ModelProvider
from brain.tools import AgentTool

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4
MESSAGE_OVERHEAD_TOKENS = 4
TOOL_OVERHEAD_TOKENS = 16
SUMMARY_MESSAGE_CHAR_LIMIT = 500
DEFAULT_COMPACTION_RESERVE_TOKENS = 16_384
DEFAULT_COMPACTION_KEEP_RECENT_TOKENS = 20_000

# Starts the synthetic user message that stands in for compacted history in
# the model's context (app/sessions builds it from the latest compaction).
# Compaction is an app concern, so brain/ never sees this as anything but text.
COMPACTION_SUMMARY_PREFIX = "Previous conversation summary:\n"

SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a context summarization assistant. Your task is to read a conversation "
    "between a user and an AI assistant, then produce a structured summary following "
    "the exact format specified.\n\n"
    "Do NOT continue the conversation. Do NOT respond to any questions in the "
    "conversation. ONLY output the structured summary."
)

SUMMARIZATION_PROMPT = (
    "The messages above are a conversation to summarize. Create a structured context "
    "checkpoint summary that another LLM will use to continue the work.\n\n"
    "Use this EXACT format:\n\n"
    "## Goal\n"
    "[What is the user trying to accomplish? Can be multiple items if the session "
    "covers different tasks.]\n\n"
    "## Constraints & Preferences\n"
    "- [Any constraints, preferences, or requirements mentioned by user]\n"
    '- [Or "(none)" if none were mentioned]\n\n'
    "## Progress\n"
    "### Done\n"
    "- [x] [Completed tasks/changes]\n\n"
    "### In Progress\n"
    "- [ ] [Current work]\n\n"
    "### Blocked\n"
    "- [Issues preventing progress, if any]\n\n"
    "## Key Decisions\n"
    "- **[Decision]**: [Brief rationale]\n\n"
    "## Next Steps\n"
    "1. [Ordered list of what should happen next]\n\n"
    "## Critical Context\n"
    "- [Any data, examples, or references needed to continue]\n"
    '- [Or "(none)" if not applicable]\n\n'
    "Keep each section concise. Preserve exact names, ids, and error messages."
)

UPDATE_SUMMARIZATION_PROMPT = (
    "The messages above are NEW conversation messages to incorporate into the existing "
    "summary provided in <previous-summary> tags.\n\n"
    "Update the existing structured summary with new information. RULES:\n"
    "- PRESERVE all existing information from the previous summary\n"
    "- ADD new progress, decisions, and context from the new messages\n"
    '- UPDATE the Progress section: move items from "In Progress" to "Done" when '
    "completed\n"
    '- UPDATE "Next Steps" based on what was accomplished\n'
    "- PRESERVE exact names, ids, and error messages\n"
    "- If something is no longer relevant, you may remove it\n\n"
    "Use this EXACT format:\n\n"
    "## Goal\n"
    "[Preserve existing goals, add new ones if the task expanded]\n\n"
    "## Constraints & Preferences\n"
    "- [Preserve existing, add new ones discovered]\n\n"
    "## Progress\n"
    "### Done\n"
    "- [x] [Include previously done items AND newly completed items]\n\n"
    "### In Progress\n"
    "- [ ] [Current work - update based on progress]\n\n"
    "### Blocked\n"
    "- [Current blockers - remove if resolved]\n\n"
    "## Key Decisions\n"
    "- **[Decision]**: [Brief rationale] (preserve all previous, add new)\n\n"
    "## Next Steps\n"
    "1. [Update based on current state]\n\n"
    "## Critical Context\n"
    "- [Preserve important context, add new if needed]\n\n"
    "Keep each section concise. Preserve exact names, ids, and error messages."
)


def message_text(message: AgentMessage) -> str:
    return message.text


def estimate_text_tokens(text: str) -> int:
    """Return a deterministic rough token estimate for text."""
    if not text:
        return 0
    return max(1, (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN)


def estimate_message_tokens(message: AgentMessage) -> int:
    """Return a rough token estimate for one provider-neutral message."""
    tokens = MESSAGE_OVERHEAD_TOKENS + estimate_text_tokens(message_text(message))
    if isinstance(message, AssistantMessage):
        tokens += sum(
            estimate_text_tokens(block.thinking)
            for block in message.content
            if isinstance(block, ThinkingContent)
        )
        tokens += sum(
            estimate_text_tokens(call.name) + estimate_text_tokens(str(call.arguments))
            for call in message.tool_calls
        )
    elif isinstance(message, ToolResultMessage):
        tokens += estimate_text_tokens(message.tool_name)
    return tokens


def estimate_tool_tokens(tool: AgentTool) -> int:
    """Return a rough token estimate for one tool definition."""
    return (
        TOOL_OVERHEAD_TOKENS
        + estimate_text_tokens(tool.name)
        + estimate_text_tokens(tool.description)
        + estimate_text_tokens(str(tool.openai_parameters()))
    )


def auto_compaction_threshold_for_context_window(context_window_tokens: int) -> int | None:
    """Return the automatic compaction threshold for a model context window."""
    if context_window_tokens <= 0:
        return None
    return max(1, context_window_tokens - DEFAULT_COMPACTION_RESERVE_TOKENS)


def provider_context_tokens(message: AssistantMessage) -> int:
    """Return the provider-reported context represented by an assistant response."""
    usage = message.usage
    total = usage.get("total_tokens")
    if total:
        return total
    return usage.get("input", 0) + usage.get("output", 0)


def _last_applicable_provider_usage(
    messages: tuple[AgentMessage, ...],
) -> tuple[int, int] | None:
    """Find the latest valid usage block that still describes the active prefix.

    A newer prefix message can be inserted by compaction (a synthetic summary
    message effectively replaces everything before it). In that case an
    older assistant's usage describes the pre-compaction context and must
    not be reused.
    """
    latest_prefix_timestamp = -1
    usage_info: tuple[int, int] | None = None
    for index, message in enumerate(messages):
        if isinstance(message, AssistantMessage):
            tokens = provider_context_tokens(message)
            if (
                message.timestamp_ms >= latest_prefix_timestamp
                and message.stop_reason not in {"aborted", "error"}
                and tokens > 0
            ):
                usage_info = (index, tokens)
        latest_prefix_timestamp = max(latest_prefix_timestamp, message.timestamp_ms)
    return usage_info


def estimate_context_tokens(
    *,
    system: str,
    messages: tuple[AgentMessage, ...],
    tools: tuple[AgentTool, ...],
) -> int:
    """Return a rough estimate of the active provider context size."""
    usage_info = _last_applicable_provider_usage(messages)
    if usage_info is not None:
        usage_index, provider_tokens = usage_info
        trailing = messages[usage_index + 1 :]
        return provider_tokens + sum(estimate_message_tokens(m) for m in trailing)

    return (
        estimate_text_tokens(system)
        + sum(estimate_message_tokens(m) for m in messages)
        + sum(estimate_tool_tokens(t) for t in tools)
    )


def summarize_messages_for_compaction(messages: tuple[AgentMessage, ...]) -> str:
    """Build a deterministic compact summary from provider-neutral messages.

    Used as a fallback when the real (LLM-generated) summarization call
    fails - compaction must never block or crash the real turn.
    """
    if not messages:
        return "No prior messages."
    # A previous summary is carried over whole: truncating it like the other
    # messages would throw away most of what earlier compactions kept.
    previous, new_messages = _split_previous_compaction_summary(messages)
    lines = [f"Automatically compacted {len(new_messages)} prior message(s)."]
    for index, message in enumerate(new_messages, start=1):
        lines.append(f"{index}. {_message_text(message)}")
    text = "\n".join(lines)
    return f"{previous.rstrip()}\n\n{text}" if previous else text


def build_compaction_summary_prompt(messages: tuple[AgentMessage, ...]) -> str:
    """Build the model prompt used to summarize compacted history."""
    previous_summary, new_messages = _split_previous_compaction_summary(messages)
    conversation = serialize_messages_for_compaction(new_messages)
    prompt = f"<conversation>\n{conversation}\n</conversation>\n\n"
    base_prompt = (
        UPDATE_SUMMARIZATION_PROMPT if previous_summary is not None else SUMMARIZATION_PROMPT
    )
    if previous_summary is not None:
        prompt += f"<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"
    return f"{prompt}{base_prompt}"


def serialize_messages_for_compaction(messages: tuple[AgentMessage, ...]) -> str:
    """Serialize provider-neutral messages for the compaction summarizer."""
    if not messages:
        return "(no new messages)"

    lines: list[str] = []
    for index, message in enumerate(messages, start=1):
        attributes = f"index={index} role={message.role}"
        if isinstance(message, ToolResultMessage):
            attributes += f" name={message.tool_name} error={str(message.is_error).lower()}"
        lines.append(f"<message {attributes}>")
        text = message_text(message)
        if text:
            lines.append(text)
        if isinstance(message, AssistantMessage) and message.tool_calls:
            lines.append("<tool-calls>")
            for call in message.tool_calls:
                lines.append(f"- {call.name}: {call.arguments}")
            lines.append("</tool-calls>")
        lines.append("</message>")
    return "\n".join(lines)


def _message_text(message: AgentMessage) -> str:
    text = message_text(message)
    if isinstance(message, AssistantMessage) and message.tool_calls:
        names = ", ".join(call.name for call in message.tool_calls)
        text = f"{text} [tool calls: {names}]"
    elif isinstance(message, ToolResultMessage):
        status = "failed" if message.is_error else "ok"
        text = f"{message.tool_name} {status}: {text}"
    return _truncate_summary_text(text)


def _truncate_summary_text(text: str) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= SUMMARY_MESSAGE_CHAR_LIMIT:
        return collapsed
    return collapsed[: SUMMARY_MESSAGE_CHAR_LIMIT - 3].rstrip() + "..."


def _split_previous_compaction_summary(
    messages: tuple[AgentMessage, ...],
) -> tuple[str | None, tuple[AgentMessage, ...]]:
    if not messages:
        return None, messages
    first = messages[0]
    if not isinstance(first, UserMessage):
        return None, messages
    text = message_text(first)
    if not text.startswith(COMPACTION_SUMMARY_PREFIX):
        return None, messages
    return text.removeprefix(COMPACTION_SUMMARY_PREFIX), messages[1:]


def _next_user_message_index(
    rows: tuple[AgentMessage, ...],
    *,
    start: int,
) -> int | None:
    for index in range(start, len(rows)):
        if rows[index].role == "user":
            return index
    return None


def _first_recent_context_index(
    messages: tuple[AgentMessage, ...],
    *,
    keep_recent_tokens: int = DEFAULT_COMPACTION_KEEP_RECENT_TOKENS,
) -> int:
    """Find the boundary that keeps roughly the last ``keep_recent_tokens``.

    Walks backward accumulating token estimates, then snaps the boundary
    forward to the next clean user-message turn so a tool call and its
    result are never split apart, and the kept suffix always starts with a
    real user turn rather than mid-way through an assistant's tool use.
    """
    if keep_recent_tokens <= 0:
        return len(messages)

    accumulated = 0
    candidate_index: int | None = None
    for index in range(len(messages) - 1, -1, -1):
        accumulated += estimate_message_tokens(messages[index])
        if accumulated >= keep_recent_tokens:
            candidate_index = index
            break

    if candidate_index is None:
        return 0

    candidate = messages[candidate_index]
    if candidate.role == "user":
        if candidate_index > 0:
            return candidate_index
        next_user = _next_user_message_index(messages, start=1)
        return next_user if next_user is not None else 0

    next_user = _next_user_message_index(messages, start=candidate_index + 1)
    if next_user is not None:
        return next_user

    for index in range(candidate_index, len(messages)):
        if messages[index].role != "tool_result":
            return index
    return len(messages)


class CompactionPlan:
    __slots__ = ("messages_to_summarize", "kept_suffix")

    def __init__(
        self, messages_to_summarize: tuple[AgentMessage, ...], kept_suffix: tuple[AgentMessage, ...]
    ) -> None:
        self.messages_to_summarize = messages_to_summarize
        self.kept_suffix = kept_suffix


def build_recent_preserving_compaction_plan(
    messages: tuple[AgentMessage, ...],
) -> CompactionPlan | None:
    if len(messages) < 2:
        return None
    first_kept_index = _first_recent_context_index(messages)
    if first_kept_index <= 0 or first_kept_index >= len(messages):
        return None
    return CompactionPlan(
        messages_to_summarize=messages[:first_kept_index],
        kept_suffix=messages[first_kept_index:],
    )


async def _generate_summary_text(
    provider: ModelProvider, messages_to_summarize: tuple[AgentMessage, ...]
) -> str:
    """Call the model for a real summary; never raises.

    Mirrors app.sessions.generate_session_title's shape exactly: a tiny,
    tool-free call with a dedicated system prompt. On any failure, the
    caller falls back to summarize_messages_for_compaction() (deterministic,
    no LLM call) - compaction must never block or crash the real turn, the
    same philosophy already applied to session auto-naming.
    """
    prompt = build_compaction_summary_prompt(messages_to_summarize)
    try:
        text_parts: list[str] = []
        async for chunk in provider.stream(
            [UserMessage(content=prompt)], [], system=SUMMARIZATION_SYSTEM_PROMPT
        ):
            if chunk.content:
                text_parts.append(chunk.content)
        summary = "".join(text_parts).strip()
        if summary:
            return summary
    except Exception:  # noqa: BLE001 — compaction must never break the real turn
        logger.warning("compaction summarization call failed, using deterministic fallback")
    return summarize_messages_for_compaction(messages_to_summarize)


@dataclass(frozen=True, slots=True)
class ContextCompaction:
    """A decision to compact: the summary, and where the kept messages start.

    ``kept_from`` indexes the context that was passed in. The caller records
    this against its own full transcript (app/sessions), which is never
    rewritten; only the context sent to the model gets shorter.
    """

    summary: str
    kept_from: int


async def compact_if_needed(
    context: Sequence[AgentMessage],
    *,
    provider: ModelProvider,
    system: str,
    tools: Sequence[AgentTool],
    context_window_tokens: int,
) -> ContextCompaction | None:
    """Summarise the older part of ``context`` if it's grown past the
    auto-compaction threshold for the configured context window, or return
    None when it fits or there is no safe place to split it.

    Called once, before a turn starts (app/main.py) - see this module's
    docstring for what tau does that this deliberately doesn't (mid-loop
    overflow recovery, dynamic-tool-token tracking, per-model catalog
    lookups).
    """
    threshold = auto_compaction_threshold_for_context_window(context_window_tokens)
    if threshold is None:
        return None

    messages = tuple(context)
    current = estimate_context_tokens(system=system, messages=messages, tools=tuple(tools))
    if current <= threshold:
        return None

    plan = build_recent_preserving_compaction_plan(messages)
    if plan is None:
        # Not enough history to find a safe boundary (e.g. one huge turn) -
        # nothing sensible to do here; let the turn proceed and, if it
        # genuinely overflows, fail normally rather than force a bad split.
        return None
    previous, new_messages = _split_previous_compaction_summary(plan.messages_to_summarize)
    if previous is not None and not new_messages:
        # Only the previous summary falls before the boundary: re-summarising
        # it alone can't shrink the context, so same as no safe boundary.
        return None

    summary = await _generate_summary_text(provider, plan.messages_to_summarize)
    logger.info(
        "auto-compacted %d message(s) (%d tokens estimated) down to a summary + %d kept",
        len(plan.messages_to_summarize),
        current,
        len(plan.kept_suffix),
    )
    return ContextCompaction(summary=summary, kept_from=len(plan.messages_to_summarize))
