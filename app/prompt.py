"""System prompt assembly — tool-contributed guidance, tau_coding.system_prompt style.

Each tool describes its own prompt line and any usage guidelines
(``AgentTool.prompt_line`` / ``prompt_guidelines``); this module only
collects and formats them. That split is what stops a tool from silently
going unmentioned in the prompt the way one once did when the
tool list here was a hand-maintained paragraph.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from xml.sax.saxutils import quoteattr

from brain.tools import AgentTool

logger = logging.getLogger(__name__)

# Standing instructions the person keeps in the workspace (see README).
AGENTS_FILE_NAME = "AGENTS.md"

# Written once, when Humphrey creates the workspace, for the person to edit.
# Shaped like tau's own AGENTS.md: a title, what this is, then short sections
# of instructions to the agent. All of it reaches the model, so it holds
# defaults that are fine as they are, not placeholders.
AGENTS_STARTER = """\
# Humphrey Instructions

This workspace is where Humphrey reads, writes and runs things for me.
These instructions apply to every chat. Keep them short.

## Working Style

- Lead with the answer, then the detail.
- Ask before deleting or overwriting a file you didn't create in this chat.

## Files

- Give files clear names that say what they are.
"""

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


def format_project_context(path: Path, content: str) -> str:
    """Wrap an instructions file in tau's (and Pi's) project-context wording."""
    return "\n".join(
        [
            "\n\n<project_context>",
            "",
            "Project-specific instructions and guidelines:",
            "",
            f"<project_instructions path={quoteattr(str(path))}>",
            content,
            "</project_instructions>",
            "",
            "</project_context>",
        ]
    )


def workspace_instructions(workspace: Path) -> str:
    """The workspace's AGENTS.md as a prompt section, or "" when there's none.

    Read on every call, so an edit applies from the next message. An empty or
    unreadable file adds nothing; it never stops a turn.
    """
    path = workspace / AGENTS_FILE_NAME
    try:
        content = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("couldn't read %s, so it was left out of the prompt: %s", path, exc)
        return ""
    return format_project_context(path, content) if content else ""


def prepare_workspace(workspace: Path) -> None:
    """Create the workspace, with a starter AGENTS.md the first time only,
    so deleting the file later keeps it deleted."""
    if workspace.exists():
        return
    workspace.mkdir(parents=True)
    (workspace / AGENTS_FILE_NAME).write_text(AGENTS_STARTER, encoding="utf-8")
