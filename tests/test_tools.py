"""Direct unit tests for tool handlers that warrant precise, non-end-to-end
coverage: read_skill_asset (path containment is security-sensitive), plus the
example tools in app.tools.examples (network calls are faked).

Uses a throwaway fixture skill (not a bundled real one) so this test file
doesn't depend on which skills happen to be present in this copy.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from app.tools import TOOLS
from app.tools.examples import _calculate
from app.tools.platform import _read_skill_asset


def _parse(raw: str) -> dict:
    return json.loads(raw)


@pytest.fixture()
def fixture_skill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """A throwaway skill with a bundled sibling asset, for read_skill_asset tests."""
    import app.skills as skills_module

    skill_dir = tmp_path / "demo-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\ndescription: demo\n---\n\nDemo skill body.\n")
    data_dir = skill_dir / "data"
    data_dir.mkdir()
    (data_dir / "table.md").write_text(
        "line one\nline two\nline three\nline four\nline five\nline six\n"
    )

    monkeypatch.setattr(skills_module, "SKILLS_DIR", tmp_path)
    skills_module.reset_skills_cache_for_tests()
    yield "demo-skill"
    skills_module.reset_skills_cache_for_tests()


def test_read_skill_asset_reads_a_bundled_sibling_file(fixture_skill: str) -> None:
    result = _parse(_read_skill_asset({"name": fixture_skill, "path": "data/table.md"}))
    assert "error" not in result
    assert "line one" in result["content"]
    assert result["skill"] == fixture_skill


def test_read_skill_asset_blocks_absolute_path_escape(fixture_skill: str) -> None:
    result = _parse(_read_skill_asset({"name": fixture_skill, "path": "/etc/hostname"}))
    assert "error" in result
    assert "escapes" in result["error"]


def test_read_skill_asset_blocks_parent_traversal_escape(fixture_skill: str) -> None:
    result = _parse(
        _read_skill_asset({"name": fixture_skill, "path": "../../../app/settings.py"})
    )
    assert "error" in result
    assert "escapes" in result["error"]


def test_read_skill_asset_unknown_skill(fixture_skill: str) -> None:
    result = _parse(_read_skill_asset({"name": "does-not-exist", "path": "x.md"}))
    assert "error" in result
    assert "unknown skill" in result["error"]


def test_read_skill_asset_missing_asset(fixture_skill: str) -> None:
    result = _parse(_read_skill_asset({"name": fixture_skill, "path": "no-such-file.md"}))
    assert "error" in result
    assert "not found" in result["error"]


def test_read_skill_asset_requires_name_and_path(fixture_skill: str) -> None:
    assert "error" in _parse(_read_skill_asset({"path": "x.md"}))
    assert "error" in _parse(_read_skill_asset({"name": fixture_skill}))


def test_read_skill_asset_offset_and_limit_paginate(fixture_skill: str) -> None:
    first = _parse(
        _read_skill_asset({"name": fixture_skill, "path": "data/table.md", "limit": 2})
    )
    assert "error" not in first
    assert first["end_line"] == 2
    assert first["total_lines"] == 6
    assert "more lines" in first["content"]

    second = _parse(
        _read_skill_asset({"name": fixture_skill, "path": "data/table.md", "offset": 3, "limit": 2})
    )
    assert second["start_line"] == 3


def test_read_skill_asset_offset_past_end_is_an_error(fixture_skill: str) -> None:
    result = _parse(
        _read_skill_asset({"name": fixture_skill, "path": "data/table.md", "offset": 99999})
    )
    assert "error" in result


def test_read_skill_asset_is_registered_with_correct_schema() -> None:
    tool = next(t for t in TOOLS if t.name == "read_skill_asset")
    assert set(tool.required) == {"name", "path"}
    assert "name" in tool.parameters
    assert "path" in tool.parameters


# --- example tools ------------------------------------------------------------


@pytest.mark.parametrize(
    ("expression", "expected"),
    [("(3 + 4) * 2", 14), ("2 ^ 10", 1024), ("sqrt(16) + abs(-1)", 5.0), ("-7 // 2", -4)],
)
def test_calculate_evaluates_arithmetic(expression: str, expected: float) -> None:
    assert _parse(_calculate({"expression": expression}))["result"] == expected


@pytest.mark.parametrize(
    "expression",
    ["__import__('os').system('echo hi')", "(1).__class__", "open('x')", "9 ** 99999", "1 / 0", ""],
)
def test_calculate_rejects_anything_but_arithmetic(expression: str) -> None:
    assert "error" in _parse(_calculate({"expression": expression}))


def test_get_weather_resolves_place_then_fetches_forecast(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.tools.examples as examples

    responses = {
        examples.GEOCODE_URL: {
            "results": [{"name": "London", "admin1": "England", "country": "United Kingdom",
                         "latitude": 51.5, "longitude": -0.12}]
        },
        examples.FORECAST_URL: {
            "timezone": "Europe/London",
            "current": {"temperature_2m": 12.3},
            "hourly": {"time": ["2026-01-01T10:00"], "temperature_2m": [12.0], "precipitation": [0.1]},
        },
    }

    def fake_get(url: str, **_kwargs: object) -> httpx.Response:
        return httpx.Response(200, json=responses[url], request=httpx.Request("GET", url))

    monkeypatch.setattr(examples.httpx, "get", fake_get)
    result = _parse(examples._get_weather({"location": "London"}))
    assert result["place"] == "London, England, United Kingdom"
    assert result["current"]["temperature_2m"] == 12.3
    assert result["next_hours"] == [{"time": "2026-01-01T10:00", "temperature_c": 12.0, "precipitation_mm": 0.1}]


def test_get_weather_reports_unknown_place(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.tools.examples as examples

    def fake_get(url: str, **_kwargs: object) -> httpx.Response:
        return httpx.Response(200, json={}, request=httpx.Request("GET", url))

    monkeypatch.setattr(examples.httpx, "get", fake_get)
    assert "no place found" in _parse(examples._get_weather({"location": "Nowhereville"}))["error"]
