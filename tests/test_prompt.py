"""System prompt assembly — every tool must appear, guidelines de-duplicated."""

from __future__ import annotations

from app.prompt import build_system_prompt, collect_tool_guidelines, format_available_tools
from app.tools import TOOLS


def test_every_tool_appears_in_the_assembled_prompt() -> None:
    """Regression guard: read_skill once shipped with no mention in the prompt at all."""
    listing = format_available_tools(TOOLS)
    for tool in TOOLS:
        assert tool.name in listing


def test_tool_without_explicit_snippet_falls_back_to_description() -> None:
    from brain.tools import AgentTool

    tool = AgentTool(
        name="example_tool",
        label="Example",
        description="Full schema description for the model.",
        parameters={},
        execute_fn=lambda *_: None,  # type: ignore[arg-type,return-value]
    )
    assert tool.prompt_line == tool.description


def test_collect_tool_guidelines_deduplicates() -> None:
    from brain.tools import AgentTool

    shared = "Do the same thing."
    tool_a = AgentTool(
        name="a",
        label="A",
        description="d",
        parameters={},
        execute_fn=lambda *_: None,  # type: ignore[arg-type,return-value]
        prompt_guidelines=(shared,),
    )
    tool_b = AgentTool(
        name="b",
        label="B",
        description="d",
        parameters={},
        execute_fn=lambda *_: None,  # type: ignore[arg-type,return-value]
        prompt_guidelines=(shared, "Something else."),
    )
    guidelines = collect_tool_guidelines([tool_a, tool_b])
    assert guidelines == [shared, "Something else."]


def test_build_system_prompt_includes_general_guidelines_and_skills_placeholder() -> None:
    prompt = build_system_prompt(TOOLS)
    assert "Never claim you can run shell" in prompt
    assert "status" in prompt


def test_mermaid_guidance_moved_out_of_the_base_prompt() -> None:
    """Mermaid guidance now lives in the mermaid-diagrams skill, not the always-on prompt."""
    prompt = build_system_prompt(TOOLS)
    assert "```mermaid" not in prompt
