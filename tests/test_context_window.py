"""app/context_window.py: token estimation and auto-compaction, ported from
tau_coding.context_window.
"""

from __future__ import annotations

import pytest

from app.context_window import (
    COMPACTION_SUMMARY_PREFIX,
    DEFAULT_COMPACTION_RESERVE_TOKENS,
    auto_compaction_threshold_for_context_window,
    build_recent_preserving_compaction_plan,
    compact_if_needed,
    estimate_context_tokens,
    estimate_message_tokens,
    estimate_text_tokens,
    provider_context_tokens,
    summarize_messages_for_compaction,
)
from brain.messages import AssistantMessage, TextContent, ToolCall, ToolResultMessage, UserMessage
from brain.provider import ModelChunk


def _user(text: str) -> UserMessage:
    return UserMessage(content=text)


def _assistant(text: str = "", *, usage: dict | None = None, tool_calls: list | None = None) -> AssistantMessage:
    content = list(tool_calls or [])
    if text:
        content.append(TextContent(text=text))
    return AssistantMessage(content=content, model="test-model", usage=usage or {})


def _tool_result(text: str = "ok") -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id="call_1", tool_name="some_tool", content=[TextContent(text=text)]
    )


# --- pure token estimation ---------------------------------------------------


def test_estimate_text_tokens_empty_is_zero() -> None:
    assert estimate_text_tokens("") == 0


def test_estimate_text_tokens_rounds_up() -> None:
    assert estimate_text_tokens("abcd") == 1
    assert estimate_text_tokens("abcde") == 2


def test_estimate_message_tokens_counts_tool_calls_and_thinking() -> None:
    from brain.messages import ThinkingContent

    bare = _assistant("hello")
    with_extras = AssistantMessage(
        content=[
            ThinkingContent(thinking="a" * 40),
            ToolCall(id="c1", name="get_weather", arguments={"lat": 1.0}),
            TextContent(text="hello"),
        ],
        model="test-model",
    )
    assert estimate_message_tokens(with_extras) > estimate_message_tokens(bare)


def test_provider_context_tokens_prefers_total_then_sums() -> None:
    assert provider_context_tokens(_assistant(usage={"total_tokens": 500})) == 500
    assert provider_context_tokens(_assistant(usage={"input": 100, "output": 50})) == 150
    assert provider_context_tokens(_assistant(usage={})) == 0


def test_auto_compaction_threshold_subtracts_reserve() -> None:
    assert (
        auto_compaction_threshold_for_context_window(128_000)
        == 128_000 - DEFAULT_COMPACTION_RESERVE_TOKENS
    )


def test_auto_compaction_threshold_none_for_non_positive_window() -> None:
    assert auto_compaction_threshold_for_context_window(0) is None
    assert auto_compaction_threshold_for_context_window(-1) is None


def test_estimate_context_tokens_anchors_on_latest_provider_usage() -> None:
    messages = (
        _user("first question"),
        _assistant("first answer", usage={"total_tokens": 1000}),
        _user("second question"),
    )
    # Should be provider_tokens (1000) + estimate of the trailing user message,
    # NOT a fresh full-history estimate from scratch.
    total = estimate_context_tokens(system="be terse", messages=messages, tools=())
    assert total > 1000
    assert total < 1000 + 100  # trailing message is small


def test_estimate_context_tokens_falls_back_without_usage() -> None:
    messages = (_user("hello"), _assistant("hi there"))
    total = estimate_context_tokens(system="system prompt text", messages=messages, tools=())
    assert total > 0


def test_estimate_context_tokens_ignores_usage_from_error_or_aborted_turns() -> None:
    messages = (
        _user("q"),
        AssistantMessage(
            content=[TextContent(text="partial")],
            model="test-model",
            usage={"total_tokens": 999_999},
            stop_reason="error",
        ),
    )
    total = estimate_context_tokens(system="", messages=messages, tools=())
    assert total < 999_999


# --- compaction boundary search ----------------------------------------------


def test_no_plan_for_short_history() -> None:
    assert build_recent_preserving_compaction_plan((_user("hi"),)) is None
    assert build_recent_preserving_compaction_plan(()) is None


def test_plan_never_starts_a_kept_suffix_with_an_orphaned_tool_result() -> None:
    """The dangerous case: a kept suffix beginning with a tool_result whose
    matching tool_call got summarized away — that's an invalid transcript for
    most providers (a result with nothing to attach to). The boundary search
    must always snap forward past that, even when the token count would
    otherwise cut mid-turn.
    """
    big = "x" * 32_000  # ~8000 tokens per message at 4 chars/token
    messages = (
        _user("turn one " + big),
        _assistant("ok " + big, tool_calls=[ToolCall(id="c1", name="t", arguments={})]),
        _tool_result(big),
        _assistant("done " + big),
        _user("turn two " + big),
        _assistant("ok " + big, tool_calls=[ToolCall(id="c2", name="t", arguments={})]),
        _tool_result(big),
        _assistant("done " + big),
    )
    plan = build_recent_preserving_compaction_plan(messages)
    assert plan is not None
    assert plan.kept_suffix[0].role != "tool_result"
    summarized_roles = [m.role for m in plan.messages_to_summarize]
    kept_roles = [m.role for m in plan.kept_suffix]
    assert summarized_roles + kept_roles == [m.role for m in messages]


def test_plan_keeps_everything_when_history_is_short_enough() -> None:
    messages = (_user("hi"), _assistant("hello"))
    # keep_recent_tokens default (20_000) vastly exceeds this - no valid split.
    assert build_recent_preserving_compaction_plan(messages) is None


# --- deterministic fallback summary -------------------------------------------


def test_summarize_messages_for_compaction_is_deterministic_and_nonempty() -> None:
    messages = (_user("hello"), _assistant("hi"), _tool_result("done"))
    summary = summarize_messages_for_compaction(messages)
    assert "3 prior message" in summary
    assert summarize_messages_for_compaction(messages) == summary


def test_summarize_messages_for_compaction_empty() -> None:
    assert summarize_messages_for_compaction(()) == "No prior messages."


# --- compact_if_needed orchestration ------------------------------------------


class _FakeSummaryProvider:
    def __init__(self, text: str = "structured summary text") -> None:
        self._text = text
        self.called_with: dict | None = None

    async def stream(self, messages, tools, *, system, signal=None):
        self.called_with = {"messages": messages, "tools": tools, "system": system}
        yield ModelChunk(content=self._text)


class _FailingProvider:
    async def stream(self, messages, tools, *, system, signal=None):
        raise RuntimeError("provider unavailable")
        yield  # pragma: no cover — makes this an async generator


def _long_history(turns: int) -> list:
    history: list = []
    filler = "word " * 200  # ~250 tokens per message at 4 chars/token
    for i in range(turns):
        history.append(_user(f"turn {i} {filler}"))
        history.append(_assistant(f"answer {i} {filler}"))
    return history


@pytest.mark.asyncio
async def test_compact_if_needed_returns_none_below_threshold() -> None:
    history = [_user("hi"), _assistant("hello")]
    provider = _FakeSummaryProvider()
    result = await compact_if_needed(
        history, provider=provider, system="sys", tools=[], context_window_tokens=128_000
    )
    assert result is None
    assert provider.called_with is None


@pytest.mark.asyncio
async def test_compact_if_needed_summarises_the_prefix_and_keeps_a_clean_suffix() -> None:
    history = _long_history(200)  # comfortably over a tiny configured window
    provider = _FakeSummaryProvider("## Goal\nDo the thing.")
    result = await compact_if_needed(
        history, provider=provider, system="sys", tools=[], context_window_tokens=5_000
    )
    assert result is not None
    assert result.summary == "## Goal\nDo the thing."
    assert provider.called_with is not None
    assert provider.called_with["tools"] == []
    assert 0 < result.kept_from < len(history)
    # The kept part starts on a real user turn, and the input is untouched.
    assert isinstance(history[result.kept_from], UserMessage)
    assert len(history) == 400


@pytest.mark.asyncio
async def test_compact_if_needed_falls_back_when_summarization_call_fails() -> None:
    history = _long_history(200)
    result = await compact_if_needed(
        history,
        provider=_FailingProvider(),
        system="sys",
        tools=[],
        context_window_tokens=5_000,
    )
    assert result is not None
    assert result.summary.startswith("Automatically compacted")


@pytest.mark.asyncio
async def test_compact_if_needed_declines_with_no_safe_boundary() -> None:
    # A single oversized message: over threshold, but nothing safe to split.
    history = [_user("word " * 5000)]
    provider = _FakeSummaryProvider()
    result = await compact_if_needed(
        history, provider=provider, system="sys", tools=[], context_window_tokens=1_000
    )
    assert result is None
    assert provider.called_with is None


@pytest.mark.asyncio
async def test_compact_if_needed_does_not_re_summarise_a_summary_on_its_own() -> None:
    """When the kept turns alone are over the threshold, the boundary lands
    straight after the previous summary. Summarising just that summary can't
    shrink anything, so it's skipped rather than costing a model call."""
    filler = "word " * 20_000  # one ~25k-token turn, over keep-recent on its own
    history = [
        _user(COMPACTION_SUMMARY_PREFIX + "earlier work"),
        _user(f"big turn {filler}"),
        _assistant("ok"),
    ]
    provider = _FakeSummaryProvider()
    result = await compact_if_needed(
        history, provider=provider, system="sys", tools=[], context_window_tokens=5_000
    )
    assert result is None
    assert provider.called_with is None


def test_the_fallback_summary_keeps_a_previous_summary_whole() -> None:
    previous = "## Goal\n" + "keep every word of this. " * 100  # well over 500 chars
    messages = (
        _user(COMPACTION_SUMMARY_PREFIX + previous),
        _user("new question"),
        _assistant("new answer"),
    )
    text = summarize_messages_for_compaction(messages)
    assert text.startswith(previous.rstrip())
    assert "Automatically compacted 2 prior message(s)." in text
    assert "new question" in text
