"""Wires the portable brain.harness.AgentHarness up with Humphrey's own
tools, skills, prompt, and provider defaults.

Split out of app/harness.py during the brain/providers/app reorg: the
AgentHarness class itself moved to brain/harness.py (zero app knowledge),
and this is the app-specific factory left behind - the same role
tau_coding's CodingSession plays relative to tau_agent.harness.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from app.prompt import build_system_prompt
from app.skills import format_skills_for_prompt
from app.tools import TOOLS
from brain.harness import AfterToolCall, AgentHarness, AgentHarnessConfig, BeforeToolCall
from brain.messages import AgentMessage
from brain.provider import ModelProvider
from providers import get_provider

# Tools and skills are baked into the image at build time, not user-editable
# at runtime, so this part of the prompt is assembled once rather than per
# request. The current date is not — see system_prompt().
STATIC_SYSTEM_PROMPT = build_system_prompt(TOOLS) + format_skills_for_prompt()


def system_prompt() -> str:
    """Recomputed per call so the date is correct across a long-lived process.

    Public - app/main.py also needs this ahead of build_harness() to
    estimate context size for auto-compaction, before a harness exists.
    """
    return f"{STATIC_SYSTEM_PROMPT}\n\nCurrent date: {date.today().isoformat()}"


def build_harness(
    *,
    session_id: str | None = None,
    provider: ModelProvider | None = None,
    messages: Sequence[AgentMessage] = (),
    before_tool_call: BeforeToolCall | None = None,
    after_tool_call: AfterToolCall | None = None,
) -> AgentHarness:
    """Construct a harness with process defaults (provider factory + tools)."""
    config = AgentHarnessConfig(
        provider=provider or get_provider(),
        system=system_prompt(),
        tools=list(TOOLS),
        session_id=session_id,
        before_tool_call=before_tool_call,
        after_tool_call=after_tool_call,
    )
    return AgentHarness(config, messages=messages)
