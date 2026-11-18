"""System prompt assembly — every tool must appear, guidelines de-duplicated."""

from __future__ import annotations

from app.prompt import build_system_prompt, collect_tool_guidelines, format_available_tools
from app.tools import TOOLS


def test_every_tool_appears_in_the_assembled_prompt() -> None:
    """Regression guard: a tool once shipped with no mention in the prompt at all."""
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
    assert "After tools return" in prompt
    assert "bash" in prompt


def test_mermaid_guidance_moved_out_of_the_base_prompt() -> None:
    """Mermaid guidance now lives in the mermaid-diagrams skill, not the always-on prompt."""
    prompt = build_system_prompt(TOOLS)
    assert "```mermaid" not in prompt


# --- workspace AGENTS.md ------------------------------------------------------


def test_without_an_agents_md_the_prompt_has_no_project_context(tmp_path) -> None:
    from app.agent import system_prompt

    assert "<project_context>" not in system_prompt(tmp_path)


def test_agents_md_is_wrapped_in_taus_project_context(tmp_path) -> None:
    from app.agent import system_prompt

    (tmp_path / "AGENTS.md").write_text("- Lead with the answer.\n", encoding="utf-8")
    assert (
        "\n\n<project_context>\n\n"
        "Project-specific instructions and guidelines:\n\n"
        f'<project_instructions path="{tmp_path / "AGENTS.md"}">\n'
        "- Lead with the answer.\n"
        "</project_instructions>\n\n"
        "</project_context>\n\n"
        "Current date: "
    ) in system_prompt(tmp_path)


def test_an_edit_to_agents_md_applies_to_the_next_prompt(tmp_path) -> None:
    from app.agent import system_prompt

    path = tmp_path / "AGENTS.md"
    path.write_text("Use metric units.", encoding="utf-8")
    assert "Use metric units." in system_prompt(tmp_path)
    path.write_text("Use imperial units.", encoding="utf-8")
    prompt = system_prompt(tmp_path)
    assert "Use imperial units." in prompt and "metric" not in prompt


def test_an_empty_or_unreadable_agents_md_adds_nothing(tmp_path) -> None:
    from app.agent import system_prompt

    path = tmp_path / "AGENTS.md"
    path.write_text("  \n", encoding="utf-8")
    assert "<project_context>" not in system_prompt(tmp_path)
    path.write_bytes(b"\xff\xfe not utf-8 \x80")
    assert "<project_context>" not in system_prompt(tmp_path)


async def test_a_turn_sends_the_workspaces_agents_md_to_the_model(monkeypatch) -> None:
    from httpx import ASGITransport, AsyncClient

    import app.main
    from app.main import app as asgi_app
    from app.tools import WORKSPACE_DIR
    from providers.mock import MockProvider

    systems: list[str] = []

    class Recording(MockProvider):
        def stream(self, messages, tools, *, system, signal=None):
            if tools:  # the reply, not the title
                systems.append(system)
            return super().stream(messages, tools, system=system, signal=signal)

    async def build_provider(model=None, effort=None):
        return Recording()

    monkeypatch.setattr(app.main, "build_provider", build_provider)
    path = WORKSPACE_DIR / "AGENTS.md"
    path.write_text("Sign off every reply with a cat emoji.", encoding="utf-8")
    try:
        async with AsyncClient(transport=ASGITransport(app=asgi_app), base_url="http://t") as c:
            reply = await c.post("/api/agent", json={"message": {"content": "hi"}})
        assert reply.status_code == 200
    finally:
        path.unlink()
    assert systems and "Sign off every reply with a cat emoji." in systems[0]
