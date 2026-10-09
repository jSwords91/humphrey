"""app/tools/tau_ported.py: ported-from-tau read/write/edit/bash tools.

These are Humphrey's tools (app.tools.TOOLS). Each test roots them in its own
tmp_path, so nothing here touches the real workspace.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.tools import TOOLS
from app.tools.tau_ported import (
    create_bash_tool,
    create_coding_tools,
    create_edit_tool,
    create_read_tool,
    create_write_tool,
)


def test_active_tools_are_the_tau_tools_rooted_in_the_workspace() -> None:
    from app.tools import WORKSPACE_DIR

    assert [t.name for t in TOOLS] == [t.name for t in create_coding_tools()]
    assert WORKSPACE_DIR.is_dir()
    assert WORKSPACE_DIR != Path(__file__).resolve().parent.parent  # never the repo root


@pytest.mark.asyncio
async def test_write_then_read_round_trips(tmp_path: Path) -> None:
    write = create_write_tool(cwd=tmp_path)
    read = create_read_tool(cwd=tmp_path)

    result = await write.execute("c1", {"path": "hello.txt", "content": "line1\nline2\nline3\n"})
    assert result.is_error is False
    assert (tmp_path / "hello.txt").read_text() == "line1\nline2\nline3\n"

    result = await read.execute("c2", {"path": "hello.txt"})
    assert result.is_error is False
    assert result.text == "line1\nline2\nline3\n"


@pytest.mark.asyncio
async def test_write_creates_parent_directories(tmp_path: Path) -> None:
    write = create_write_tool(cwd=tmp_path)
    result = await write.execute("c1", {"path": "nested/dir/file.txt", "content": "x"})
    assert result.is_error is False
    assert (tmp_path / "nested" / "dir" / "file.txt").read_text() == "x"


@pytest.mark.asyncio
async def test_read_missing_file_is_an_error(tmp_path: Path) -> None:
    read = create_read_tool(cwd=tmp_path)
    result = await read.execute("c1", {"path": "nope.txt"})
    assert result.is_error is True
    assert "not found" in result.text.lower()


@pytest.mark.asyncio
async def test_read_offset_and_limit(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("a\nb\nc\nd\ne\n")
    read = create_read_tool(cwd=tmp_path)
    result = await read.execute("c1", {"path": "f.txt", "offset": 2, "limit": 2})
    assert result.is_error is False
    assert "b\nc" in result.text
    assert "more lines in file" in result.text


@pytest.mark.asyncio
async def test_read_offset_past_end_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("a\nb\n")
    read = create_read_tool(cwd=tmp_path)
    result = await read.execute("c1", {"path": "f.txt", "offset": 99})
    assert result.is_error is True
    assert "beyond end of file" in result.text


@pytest.mark.asyncio
async def test_edit_replaces_unique_match_and_reports_a_diff(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("line1\nline2\nline3\n")
    edit = create_edit_tool(cwd=tmp_path)
    result = await edit.execute(
        "c1", {"path": "f.txt", "edits": [{"oldText": "line2", "newText": "LINE-TWO"}]}
    )
    assert result.is_error is False
    assert (tmp_path / "f.txt").read_text() == "line1\nLINE-TWO\nline3\n"
    assert "- line2" in result.details["diff"]
    assert "+ LINE-TWO" in result.details["diff"]


@pytest.mark.asyncio
async def test_edit_not_found_does_not_modify_the_file(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("line1\nline2\n")
    edit = create_edit_tool(cwd=tmp_path)
    result = await edit.execute(
        "c1", {"path": "f.txt", "edits": [{"oldText": "not-there", "newText": "x"}]}
    )
    assert result.is_error is True
    assert (tmp_path / "f.txt").read_text() == "line1\nline2\n"


@pytest.mark.asyncio
async def test_edit_ambiguous_match_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("dup\ndup\n")
    edit = create_edit_tool(cwd=tmp_path)
    result = await edit.execute("c1", {"path": "f.txt", "edits": [{"oldText": "dup", "newText": "x"}]})
    assert result.is_error is True
    assert "unique" in result.text.lower()


@pytest.mark.asyncio
async def test_edit_preserves_crlf_line_endings(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_bytes(b"line1\r\nline2\r\n")
    edit = create_edit_tool(cwd=tmp_path)
    result = await edit.execute(
        "c1", {"path": "f.txt", "edits": [{"oldText": "line2", "newText": "changed"}]}
    )
    assert result.is_error is False
    assert (tmp_path / "f.txt").read_bytes() == b"line1\r\nchanged\r\n"


@pytest.mark.asyncio
async def test_bash_returns_stdout_and_exit_code(tmp_path: Path) -> None:
    bash = create_bash_tool(cwd=tmp_path)
    result = await bash.execute("c1", {"command": "echo hi", "description": "greet"})
    assert result.is_error is False
    assert result.text.strip() == "hi"
    assert result.details["exit_code"] == 0


@pytest.mark.asyncio
async def test_bash_runs_in_cwd(tmp_path: Path) -> None:
    bash = create_bash_tool(cwd=tmp_path)
    result = await bash.execute("c1", {"command": "pwd", "description": "where"})
    assert str(tmp_path) in result.text


@pytest.mark.asyncio
async def test_bash_nonzero_exit_is_an_error(tmp_path: Path) -> None:
    bash = create_bash_tool(cwd=tmp_path)
    result = await bash.execute("c1", {"command": "exit 7", "description": "fail"})
    assert result.is_error is True
    assert result.details["exit_code"] == 7


@pytest.mark.asyncio
async def test_bash_timeout_is_not_treated_as_an_error(tmp_path: Path) -> None:
    bash = create_bash_tool(cwd=tmp_path)
    result = await bash.execute("c1", {"command": "sleep 5", "description": "slow", "timeout": 0.2})
    assert result.details["timed_out"] is True
    assert result.is_error is False
    assert "timed out" in result.text.lower()


@pytest.mark.asyncio
async def test_bash_respects_cancellation_signal(tmp_path: Path) -> None:
    class _Cancelled:
        def is_cancelled(self) -> bool:
            return True

    bash = create_bash_tool(cwd=tmp_path)
    result = await bash.execute("c1", {"command": "echo should-not-run", "description": "x"}, signal=_Cancelled())
    assert result.is_error is True
    assert "cancelled" in result.text.lower()
