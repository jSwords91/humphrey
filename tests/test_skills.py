"""Skill loading, prompt indexing, and /skill: expansion."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.skills import (
    Skill,
    SkillError,
    _load_skills_uncached,
    expand_skill_command,
    format_skill_invocation,
    format_skills_for_prompt,
    load_skills,
)


def _write_skill(root: Path, name: str, body: str) -> None:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(body, encoding="utf-8")


def test_load_skills_finds_mermaid_diagrams_skill() -> None:
    skills = load_skills()
    skill = next((s for s in skills if s.name == "mermaid-diagrams"), None)
    assert skill is not None
    assert skill.description
    assert not skill.disable_model_invocation
    assert "```mermaid" in skill.content


def test_load_skills_parses_frontmatter_and_derives_description(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "with-frontmatter",
        "---\ndescription: Explicit description\n---\n\nBody text.\n",
    )
    _write_skill(tmp_path, "no-frontmatter", "# Heading as description\n\nMore body.\n")
    _write_skill(
        tmp_path,
        "hidden",
        "---\ndescription: Hidden from the model\ndisable-model-invocation: true\n---\n\nBody.\n",
    )

    skills = _load_skills_uncached(tmp_path)
    by_name = {s.name: s for s in skills}

    assert by_name["with-frontmatter"].description == "Explicit description"
    assert by_name["no-frontmatter"].description == "Heading as description"
    assert by_name["hidden"].disable_model_invocation is True


def test_load_skills_ignores_non_skill_entries(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("not a skill", encoding="utf-8")
    (tmp_path / "empty-dir").mkdir()
    _write_skill(tmp_path, "real", "Body only, no frontmatter.\n")

    skills = _load_skills_uncached(tmp_path)
    assert [s.name for s in skills] == ["real"]


def test_load_skills_missing_dir_returns_empty(tmp_path: Path) -> None:
    assert _load_skills_uncached(tmp_path / "does-not-exist") == ()


def test_format_skills_for_prompt_excludes_disabled() -> None:
    visible = Skill(name="visible", path=Path("visible/SKILL.md"), content="x", description="Visible one")
    hidden = Skill(
        name="hidden",
        path=Path("hidden/SKILL.md"),
        content="x",
        description="Hidden one",
        disable_model_invocation=True,
    )
    prompt = format_skills_for_prompt([visible, hidden])
    assert "visible" in prompt
    assert "Visible one" in prompt
    assert "hidden" not in prompt
    assert "Hidden one" not in prompt


def test_format_skills_for_prompt_empty_when_no_skills() -> None:
    assert format_skills_for_prompt([]) == ""


def test_format_skill_invocation_wraps_content_and_request() -> None:
    skill = Skill(name="demo", path=Path("demo/SKILL.md"), content="Do the thing.", description="d")
    invocation = format_skill_invocation(skill, "extra context")
    assert '<skill name="demo"' in invocation
    assert "Do the thing." in invocation
    assert invocation.endswith("extra context")


def test_expand_skill_command_round_trip() -> None:
    skills = [Skill(name="demo", path=Path("demo/SKILL.md"), content="Do the thing.", description="d")]
    expanded = expand_skill_command("/skill:demo please hurry", skills)
    assert expanded is not None
    assert "Do the thing." in expanded
    assert expanded.endswith("please hurry")


def test_expand_skill_command_non_skill_text_returns_none() -> None:
    assert expand_skill_command("just a normal message", []) is None


def test_expand_skill_command_missing_name_raises() -> None:
    with pytest.raises(SkillError):
        expand_skill_command("/skill:", [])


def test_expand_skill_command_unknown_skill_raises() -> None:
    with pytest.raises(SkillError):
        expand_skill_command("/skill:does-not-exist", [])
