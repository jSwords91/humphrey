"""Markdown skill loading and expansion — ported from tau_coding.skills.

A skill is a directory containing a ``SKILL.md`` (tau's Agent Skills spec).
Humphrey ships exactly one skills directory baked into the image (like
``knowledge/``) rather than tau's precedence-ordered user/project directories
— there is no end-user filesystem here to layer over. Skills are also
self-contained text: unlike tau, there is no generic file-read tool for the
model to follow relative references to sibling assets, so the model pulls in
a skill's full body through the dedicated ``read_skill`` tool instead of a
file path.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from xml.sax.saxutils import escape

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


class SkillError(ValueError):
    """Raised when a skill command or lookup is invalid."""


@dataclass(frozen=True, slots=True)
class Skill:
    """A markdown skill resource."""

    name: str
    path: Path
    content: str
    description: str | None = None
    disable_model_invocation: bool = False


def _parse_markdown_resource(text: str) -> tuple[dict[str, str], str]:
    """Parse minimal YAML-like frontmatter from a markdown resource.

    Only simple ``key: value`` pairs are supported — dependency-free, no
    arbitrary code evaluation. Ported from tau_coding.resources.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.startswith("---\n"):
        return {}, normalized

    end = normalized.find("\n---", 4)
    if end == -1:
        return {}, normalized

    raw_frontmatter = normalized[4:end]
    body = normalized[end + len("\n---") :]
    if body.startswith("\n"):
        body = body[1:]

    metadata: dict[str, str] = {}
    for line in raw_frontmatter.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition(":")
        if not separator:
            continue
        metadata[key.strip()] = value.strip().strip("\"'")
    return metadata, body


def _derive_description(content: str) -> str | None:
    """Derive a short description from markdown content when none is declared."""
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or None
        return stripped
    return None


def _load_skill(name: str, path: Path) -> Skill:
    raw = path.read_text(encoding="utf-8")
    metadata, content = _parse_markdown_resource(raw)
    description = metadata.get("description") or _derive_description(content)
    disable_model_invocation = (
        metadata.get("disable-model-invocation", "").strip().lower() == "true"
    )
    return Skill(
        name=name,
        path=path,
        content=content,
        description=description,
        disable_model_invocation=disable_model_invocation,
    )


def _load_skills_uncached(skills_dir: Path) -> tuple[Skill, ...]:
    if not skills_dir.exists() or not skills_dir.is_dir():
        return ()

    skills: list[Skill] = []
    seen: set[str] = set()
    for path in sorted(skills_dir.iterdir(), key=lambda item: item.name):
        if not path.is_dir():
            continue
        skill_path = path / "SKILL.md"
        if not skill_path.exists():
            continue
        name = path.name
        if name in seen:
            continue  # duplicate skill directory name — first one wins
        seen.add(name)
        skills.append(_load_skill(name, skill_path))
    return tuple(sorted(skills, key=lambda skill: skill.name))


@lru_cache(maxsize=1)
def _cached_skills() -> tuple[Skill, ...]:
    return _load_skills_uncached(SKILLS_DIR)


def load_skills() -> list[Skill]:
    """Load bundled skills. Cached — skills are baked into the image, not

    user-editable at runtime.
    """
    return list(_cached_skills())


def reset_skills_cache_for_tests() -> None:
    _cached_skills.cache_clear()


def get_skill(name: str, skills: Sequence[Skill] | None = None) -> Skill | None:
    for skill in skills if skills is not None else load_skills():
        if skill.name == name:
            return skill
    return None


def format_skills_for_prompt(skills: Sequence[Skill] | None = None) -> str:
    """Build the <available_skills> system-prompt index (name + description only).

    Skills with ``disable_model_invocation`` are excluded — they remain
    invocable explicitly via ``/skill:<name>``.
    """
    visible = [s for s in (skills if skills is not None else load_skills()) if not s.disable_model_invocation]
    if not visible:
        return ""
    lines = [
        "\n\nThe following skills provide specialized instructions for specific tasks.",
        "When a task matches a skill's description, call read_skill with its name to load",
        "the full instructions, then follow them for the rest of the turn.",
        "",
        "<available_skills>",
    ]
    for skill in sorted(visible, key=lambda item: item.name):
        description = skill.description or "No description"
        lines.extend(
            [
                "  <skill>",
                f"    <name>{escape(skill.name)}</name>",
                f"    <description>{escape(description)}</description>",
                "  </skill>",
            ]
        )
    lines.append("</available_skills>")
    return "\n".join(lines)


def format_skill_invocation(skill: Skill, additional_instructions: str | None = None) -> str:
    """Format a full skill invocation prompt (tau's ``<skill>`` block shape)."""
    skill_block = (
        f'<skill name="{skill.name}" location="{skill.path.name}">\n'
        f"{skill.content.strip()}\n"
        "</skill>"
    )
    if additional_instructions and additional_instructions.strip():
        return f"{skill_block}\n\n{additional_instructions.strip()}"
    return skill_block


def expand_skill_command(text: str, skills: Sequence[Skill] | None = None) -> str | None:
    """Expand `/skill:name` prompt text, or return None for non-skill text."""
    stripped = text.strip()
    if not stripped.startswith("/skill:"):
        return None

    command_and_request = stripped.split(maxsplit=1)
    command = command_and_request[0]
    request = command_and_request[1] if len(command_and_request) > 1 else None
    name = command.removeprefix("/skill:").strip()
    if not name:
        raise SkillError("Skill command must include a skill name")

    skill = get_skill(name, skills)
    if skill is None:
        raise SkillError(f"Unknown skill: {name}")

    additional_instructions = request.strip() if request is not None else None
    return format_skill_invocation(skill, additional_instructions)
