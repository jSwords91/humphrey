"""Projects: a folder in the workspace, and the chats that work in it."""

from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from pathlib import Path
from typing import Any

import pytest
from httpx import AsyncClient

from app import projects, runs
from app.projects import ProjectError, ProjectExists
from app.tools import WORKSPACE_DIR


def _events(body: str) -> list[dict[str, Any]]:
    return [
        json.loads(block.strip()[6:])
        for block in body.split("\n\n")
        if block.strip().startswith("data: ")
    ]


def _unique(title: str = "Board pack") -> str:
    return f"{title} {uuid.uuid4().hex[:8]}"


def test_a_project_is_made_from_the_name_typed(tmp_path: Path) -> None:
    assert projects.create(tmp_path, "  Q4 Board Pack! ") == "q4-board-pack"
    assert (tmp_path / "projects" / "q4-board-pack").is_dir()
    with pytest.raises(ProjectExists, match="already exists"):
        projects.create(tmp_path, "q4 board-pack")
    with pytest.raises(ProjectError, match="letter or a number"):
        projects.create(tmp_path, "!!!")


def test_every_folder_under_projects_is_a_project(tmp_path: Path) -> None:
    (tmp_path / "projects" / "Made by hand").mkdir(parents=True)
    (tmp_path / "projects" / ".hidden").mkdir()
    (tmp_path / "projects" / "notes.txt").write_text("not a folder")
    assert projects.list_projects(tmp_path) == ["Made by hand"]
    assert projects.folder(tmp_path, "Made by hand") == tmp_path / "projects" / "Made by hand"
    assert projects.folder(tmp_path, None) == tmp_path


@pytest.mark.parametrize("name", ["missing", "..", "../..", "Made by hand/..", "/etc", ".hidden"])
def test_only_an_existing_project_folder_can_be_named(tmp_path: Path, name: str) -> None:
    (tmp_path / "projects" / "Made by hand").mkdir(parents=True)
    (tmp_path / "projects" / ".hidden").mkdir()
    with pytest.raises(ProjectError):
        projects.folder(tmp_path, name)


@pytest.mark.asyncio
async def test_projects_are_made_and_listed_over_http(client: AsyncClient) -> None:
    title = _unique()
    made = await client.post("/api/projects", json={"name": title})
    assert made.status_code == 201
    name = made.json()["name"]
    assert made.json()["folder"] == str(WORKSPACE_DIR / "projects" / name)

    listed = (await client.get("/api/projects")).json()
    assert listed["root"] == str(WORKSPACE_DIR)
    assert name in [p["name"] for p in listed["projects"]]

    again = await client.post("/api/projects", json={"name": title})
    assert again.status_code == 409
    plain = await client.post("/api/projects", content='{"name": "x"}', headers={"content-type": "text/plain"})
    assert plain.status_code == 415


@pytest.mark.asyncio
async def test_a_chat_in_a_project_works_in_its_folder(client: AsyncClient) -> None:
    name = (await client.post("/api/projects", json={"name": _unique()})).json()["name"]
    session_id = uuid.uuid4().hex

    wrote = await client.post(
        "/api/agent",
        json={"message": {"content": "write a note"}, "session_id": session_id, "project": name},
    )
    [artifact] = [e for e in _events(wrote.text) if e["type"] == "artifact"]
    assert artifact["path"] == f"projects/{name}/notes.md"  # how it's served
    assert artifact["name"] == "notes.md"  # how it's shown
    assert (WORKSPACE_DIR / "projects" / name / "notes.md").is_file()
    assert (await client.get(f"/api/files/projects/{name}/notes.md")).status_code == 200

    # bash runs in the project's folder too, and a later turn stays there
    # whatever the request says.
    listed = await client.post(
        "/api/agent",
        json={"message": {"content": "list files"}, "session_id": session_id, "project": None},
    )
    [end] = [e for e in _events(listed.text) if e["type"] == "tool_end"]
    assert "notes.md" in end["result"]
    stored = (await client.get(f"/api/sessions/{session_id}")).json()
    assert stored["project"] == name

    in_project = (await client.get("/api/sessions", params={"project": name})).json()
    in_default = (await client.get("/api/sessions")).json()
    assert [r["id"] for r in in_project] == [session_id]
    assert session_id not in [r["id"] for r in in_default]

    files = (await client.get("/api/files", params={"project": name})).json()
    assert [(f["name"], f["path"]) for f in files] == [("notes.md", f"projects/{name}/notes.md")]
    default_files = (await client.get("/api/files")).json()
    assert not [f for f in default_files if f["path"].startswith("projects/")]


@pytest.mark.asyncio
async def test_a_project_chat_mid_first_turn_is_listed_in_its_project(client: AsyncClient) -> None:
    name = (await client.post("/api/projects", json={"name": _unique()})).json()["name"]
    session_id = uuid.uuid4().hex
    request = asyncio.create_task(
        client.post(
            "/api/agent",
            json={"message": {"content": "ask me"}, "session_id": session_id, "project": name},
        )
    )
    for _ in range(500):
        run = runs.get(session_id)
        if run is not None and run.bridge.waiting:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("the run never asked")

    assert str(WORKSPACE_DIR / "projects" / name) in run.harness.config.system  # its prompt
    in_project = (await client.get("/api/sessions", params={"project": name})).json()
    assert [r["id"] for r in in_project] == [session_id]
    assert session_id not in [r["id"] for r in (await client.get("/api/sessions")).json()]

    await client.post(f"/api/sessions/{session_id}/stop", json={})
    await request


@pytest.mark.asyncio
async def test_a_new_chat_in_a_missing_project_is_refused(client: AsyncClient) -> None:
    response = await client.post(
        "/api/agent", json={"message": {"content": "hello"}, "project": "no-such-project"}
    )
    assert response.status_code == 422
    assert "no-such-project" in response.json()["detail"]


@pytest.mark.asyncio
async def test_a_chat_whose_folder_went_says_so(client: AsyncClient) -> None:
    name = (await client.post("/api/projects", json={"name": _unique()})).json()["name"]
    session_id = uuid.uuid4().hex
    await client.post(
        "/api/agent", json={"message": {"content": "hello"}, "session_id": session_id, "project": name}
    )
    shutil.rmtree(WORKSPACE_DIR / "projects" / name)

    again = await client.post(
        "/api/agent", json={"message": {"content": "hello again"}, "session_id": session_id}
    )
    assert again.status_code == 409
    assert "new chat" in again.json()["detail"]
    reopened = await client.get(f"/api/sessions/{session_id}")
    assert reopened.status_code == 200  # still readable
