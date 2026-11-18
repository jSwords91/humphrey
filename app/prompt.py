"""System prompt assembly — tool-contributed guidance, tau_coding.system_prompt style.

Each tool describes its own prompt line and any usage guidelines
(``AgentTool.prompt_line`` / ``prompt_guidelines``); this module only
collects and formats them. That split is what stops a tool from silently
going unmentioned in the prompt the way one once did when the
tool list here was a hand-maintained paragraph.
"""

from __future__ import annotations

from collections.abc import Sequence

from brain.tools import AgentTool

PREAMBLE = (
    "You are Humphrey, a helpful assistant.\n\n"
    "Be practical: concise, accurate, and tool-first when facts matter.\n"
    "Prefer Markdown for structure (short headings, lists, fenced code when useful)."
)

GENERAL_GUIDELINES: tuple[str, ...] = (
    "After tools return, give a short useful answer. Quote names and sources when relevant.",
    "Files you create or change in the working directory are shown to the user inline as "
    "artefacts. For reports, charts, or anything visual, write a file (Markdown, HTML, CSV, "
    "an image) rather than pasting long content into chat.",
)


def format_available_tools(tools: Sequence[AgentTool]) -> str:
    """Format one line per tool from its own ``prompt_line``."""
    lines = [f"- {tool.name} — {tool.prompt_line}" for tool in tools]
    return "\n".join(lines) if lines else "(none)"


def collect_tool_guidelines(tools: Sequence[AgentTool]) -> list[str]:
    """Collect and de-duplicate every tool's contributed guidelines, in tool order."""
    seen: set[str] = set()
    guidelines: list[str] = []
    for tool in tools:
        for guideline in tool.prompt_guidelines:
            normalized = guideline.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                guidelines.append(normalized)
    return guidelines


def build_system_prompt(tools: Sequence[AgentTool]) -> str:
    """Build Humphrey's system prompt from the preamble plus tool-contributed content."""
    guidelines = [*collect_tool_guidelines(tools), *GENERAL_GUIDELINES]
    guidance = "\n".join(f"- {g}" for g in guidelines)
    return (
        f"{PREAMBLE}\n\n"
        "Tools (use them — do not invent facts a tool can check):\n"
        f"{format_available_tools(tools)}\n\n"
        f"Guidance:\n{guidance}"
    )
