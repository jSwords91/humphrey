"""OpenAI tool schema shape required by Azure OpenAI."""

from __future__ import annotations

from app.tools import TOOLS
from providers.openai_adapter import tools_to_openai


def test_all_tools_have_object_parameter_schema() -> None:
    schemas = tools_to_openai(list(TOOLS))
    assert schemas
    for row in schemas:
        params = row["function"]["parameters"]
        assert params["type"] == "object", row["function"]["name"]
        assert isinstance(params.get("properties"), dict)
        assert isinstance(params.get("required"), list)


def test_tools_are_exactly_taus_four() -> None:
    assert [t.name for t in TOOLS] == ["read", "write", "edit", "bash"]


def test_read_schema_requires_path() -> None:
    schemas = {row["function"]["name"]: row["function"]["parameters"] for row in tools_to_openai(list(TOOLS))}
    read_schema = schemas["read"]
    assert read_schema["type"] == "object"
    assert "path" in read_schema["properties"]
    assert read_schema["required"] == ["path"]


def test_long_path_labels_keep_the_file_name() -> None:
    read = next(t for t in TOOLS if t.name == "read")
    label = read.label_for({"path": "/very/long/install/prefix/somewhere/skills/mermaid-diagrams/SKILL.md"})
    assert label.startswith("Reading …")
    assert label.endswith("mermaid-diagrams/SKILL.md")
    assert read.label_for({"path": "notes.md"}) == "Reading notes.md"
