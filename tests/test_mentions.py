"""@-mentions: tagged project files are attached for the model, not shown."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
from httpx import AsyncClient

from app.mentions import MAX_MENTIONS, mentioned_paths, mentions_block, visible_text
from app.sessions import get_session_store
from app.tools import WORKSPACE_DIR
from brain.messages import UserMessage


def _events(body: str) -> list[dict]:
    return [json.loads(b.strip()[6:]) for b in body.split("\n\n") if b.strip().startswith("data: ")]


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    (tmp_path / "bids").mkdir()
    (tmp_path / "bids" / "FCA.md").write_text("FCA answer\n")
    (tmp_path / "Q3 report.md").write_text("Q3 numbers\n")
    (tmp_path / "notes.md").write_text("notes\n")
    (tmp_path / "draft(1)").write_text("draft\n")
    return tmp_path


def test_tags_name_files_in_the_folder(folder: Path) -> None:
    text = 'Compare @bids/FCA.md with @"Q3 report.md", then see @notes.md. And @draft(1)'
    assert mentioned_paths(text, folder) == ["bids/FCA.md", "Q3 report.md", "notes.md", "draft(1)"]


@pytest.mark.parametrize(
    "text",
    [
        "mail josh@notes.md about it",  # an @ inside a word: an email, not a tag
        "ask @Josh first",  # no such file
        "@../secret.md",  # outside the folder
        "@bids",  # a folder, not a file
    ],
)
def test_other_ats_are_left_alone(folder: Path, text: str) -> None:
    (folder.parent / "secret.md").write_text("secret\n")
    assert mentioned_paths(text, folder) == []


def test_repeats_count_once_and_the_count_is_capped(tmp_path: Path) -> None:
    for i in range(MAX_MENTIONS + 2):
        (tmp_path / f"f{i}.md").write_text("x\n")
    text = "@f0.md @f0.md " + " ".join(f"@f{i}.md" for i in range(MAX_MENTIONS + 2))
    assert mentioned_paths(text, tmp_path) == [f"f{i}.md" for i in range(MAX_MENTIONS)]


@pytest.mark.asyncio
async def test_the_block_holds_each_file_as_read(folder: Path) -> None:
    (folder / "long.md").write_text("".join(f"line {i}\n" for i in range(3000)))
    (folder / "photo.png").write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe")
    block = await mentions_block("@notes.md @long.md @photo.png", folder)
    assert block is not None
    assert '<file path="notes.md">\nnotes\n</file>' in block
    assert "Use offset=2001 to continue." in block  # read's own truncation
    assert "Not a UTF-8 text file" in block  # read's own refusal, not a crash
    assert await mentions_block("nothing tagged", folder) is None


@pytest.mark.asyncio
async def test_a_tagged_file_reaches_the_model_and_not_the_bubble(client: AsyncClient) -> None:
    name = (await client.post("/api/projects", json={"name": f"Bids {uuid.uuid4().hex[:8]}"})).json()["name"]
    (WORKSPACE_DIR / "projects" / name / "brief.md").write_text("The deadline is Friday.\n")
    session_id = uuid.uuid4().hex

    first = await client.post(
        "/api/agent",
        json={"message": {"content": "Summarise @brief.md"}, "session_id": session_id, "project": name},
    )
    assert first.status_code == 200
    [echo] = [e for e in _events(first.text) if e["type"] == "user_message"]
    assert echo["text"] == "Summarise @brief.md"  # the live bubble, too
    # A later turn finds the file in the chat's project whatever the request says.
    again = await client.post(
        "/api/agent",
        json={"message": {"content": "And @brief.md again"}, "session_id": session_id, "project": None},
    )
    assert again.status_code == 200

    session = await get_session_store().get(session_id)
    assert session is not None
    sent = [m.text for m in session.messages if isinstance(m, UserMessage)]
    assert len(sent) == 2
    assert all('<file path="brief.md">\nThe deadline is Friday.\n</file>' in text for text in sent)

    shown = (await client.get(f"/api/sessions/{session_id}")).json()["messages"]
    assert [m["content"] for m in shown if m["role"] == "user"] == ["Summarise @brief.md", "And @brief.md again"]


def test_visible_text_is_what_was_typed() -> None:
    assert visible_text("hi @a.md\n\n<mentioned-files>\n...\n</mentioned-files>") == "hi @a.md"
    assert visible_text("no files here") == "no files here"


@pytest.mark.asyncio
async def test_the_file_list_filters_by_path(client: AsyncClient) -> None:
    name = (await client.post("/api/projects", json={"name": f"Filter {uuid.uuid4().hex[:8]}"})).json()["name"]
    folder = WORKSPACE_DIR / "projects" / name
    (folder / "bids").mkdir()
    (folder / "bids" / "FCA.md").write_text("x\n")
    (folder / "notes.md").write_text("x\n")
    found = (await client.get("/api/files", params={"project": name, "q": "fca"})).json()
    assert [f["name"] for f in found] == ["bids/FCA.md"]
