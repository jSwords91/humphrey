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


def test_read_doc_schema_includes_path() -> None:
    schemas = {row["function"]["name"]: row["function"]["parameters"] for row in tools_to_openai(list(TOOLS))}
    read_schema = schemas["read_doc"]
    assert read_schema["type"] == "object"
    assert "path" in read_schema["properties"]
    assert "path" in read_schema["required"]


def test_read_skill_schema_includes_name() -> None:
    schemas = {row["function"]["name"]: row["function"]["parameters"] for row in tools_to_openai(list(TOOLS))}
    read_schema = schemas["read_skill"]
    assert read_schema["type"] == "object"
    assert "name" in read_schema["properties"]
    assert "name" in read_schema["required"]
