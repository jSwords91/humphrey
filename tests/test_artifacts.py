"""Artefacts: workspace change tracking, safe file serving, and the SSE/session wiring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.artifacts import (
    EMBED_SCRIPT,
    ArtifactTracker,
    artifact_kind,
    inject_embed_script,
    resolve_workspace_file,
)


def test_write_and_edit_claim_the_file_they_name(tmp_path: Path) -> None:
    tracker = ArtifactTracker(tmp_path)
    (tmp_path / "charts").mkdir()
    (tmp_path / "charts" / "sales.html").write_text("<p>chart</p>")
    (tmp_path / "other.md").write_text("changed by someone else")

    made = tracker.tool_finished("write", {"path": str(tmp_path / "charts" / "sales.html")}, False)
    assert [(a["path"], a["kind"]) for a in made] == [("charts/sales.html", "html")]
    assert tracker.tool_finished("edit", {"path": str(tmp_path / "missing.md")}, False) == []
    assert tracker.tool_finished("write", {"path": str(tmp_path / "charts" / "sales.html")}, True) == []
    (tmp_path / ".env").write_text("x")
    assert tracker.tool_finished("write", {"path": str(tmp_path / ".env")}, False) == []


def test_bash_claims_only_what_changes_while_it_runs(tmp_path: Path) -> None:
    """Another chat, or the person, may change files between this run's tool
    calls; only changes during its own command are its."""
    (tmp_path / "old.txt").write_text("same")
    tracker = ArtifactTracker(tmp_path)
    (tmp_path / "before.md").write_text("made before the command, by someone else")

    tracker.tool_started("bash")
    (tmp_path / "report.md").write_text("# hi")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "noise.js").write_text("x")
    made = tracker.tool_finished("bash", {"exit_code": 0}, False)
    assert [a["path"] for a in made] == ["report.md"]

    (tmp_path / "after.md").write_text("made after it, by someone else")
    tracker.tool_started("read")
    assert tracker.tool_finished("read", {"path": str(tmp_path / "after.md")}, False) == []


@pytest.mark.parametrize(
    ("name", "kind"),
    [("a.HTML", "html"), ("b.md", "markdown"), ("c.svg", "image"), ("d.csv", "table"),
     ("e.py", "code"), ("f.pdf", "pdf"), ("g.mmd", "diagram"), ("h.bin", "file")],
)  # fmt: skip
def test_artifact_kind(name: str, kind: str) -> None:
    assert artifact_kind(name) == kind


def test_resolve_workspace_file_never_escapes(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    (root / "ok.md").write_text("ok")
    (tmp_path / "secret.txt").write_text("no")
    assert resolve_workspace_file(root, "ok.md") == (root / "ok.md").resolve()
    assert resolve_workspace_file(root, "../secret.txt") is None
    assert resolve_workspace_file(root, str(tmp_path / "secret.txt")) is None
    assert resolve_workspace_file(root, "missing.md") is None


def test_embed_script_goes_inside_head_after_the_doctype() -> None:
    html = '<!doctype html><html lang="en"><head><title>x</title></head><body></body></html>'
    out = inject_embed_script(html)
    assert out.startswith("<!doctype html>")
    assert out.index(EMBED_SCRIPT) > out.index("<head>")
    assert inject_embed_script("<p>bare</p>").startswith(EMBED_SCRIPT)


@pytest.fixture()
async def client() -> AsyncClient:
    from app.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


async def test_file_endpoint_sandboxes_and_contains(client: AsyncClient) -> None:
    from app.tools import WORKSPACE_DIR

    (WORKSPACE_DIR / "page.html").write_text("<html><head></head><body>hi</body></html>")
    (WORKSPACE_DIR / "pic.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")

    page = await client.get("/api/files/page.html?embed=true")
    assert page.headers["content-security-policy"] == "sandbox allow-scripts allow-popups"
    assert "artifact-size" in page.text

    plain = await client.get("/api/files/page.html")
    assert "artifact-size" not in plain.text

    svg = await client.get("/api/files/pic.svg")
    assert svg.headers["content-security-policy"] == "sandbox"
    assert svg.headers["x-content-type-options"] == "nosniff"

    download = await client.get("/api/files/pic.svg?download=true")
    assert "attachment" in download.headers["content-disposition"]

    assert (await client.get("/api/files/..%2F..%2Fetc%2Fpasswd")).status_code == 404
    listing = (await client.get("/api/files")).json()
    assert {"page.html", "pic.svg"} <= {row["path"] for row in listing}


async def test_written_file_streams_as_a_versioned_artifact_and_persists(client: AsyncClient) -> None:
    def artifacts(body: str) -> list[dict]:
        events = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]
        return [e for e in events if e["type"] == "artifact"]

    first = await client.post(
        "/api/agent", json={"message": {"content": "please write a note"}}
    )
    session_id = first.headers["x-session-id"]
    [art] = artifacts(first.text)
    assert (art["path"], art["kind"], art["version"]) == ("notes.md", "markdown", 1)

    second = await client.post(
        "/api/agent",
        json={
            "session_id": session_id,
            "message": {"content": "please write a note again"},
        },
    )
    assert [a["version"] for a in artifacts(second.text)] == [2]

    stored = (await client.get(f"/api/sessions/{session_id}")).json()
    assert [a["version"] for a in stored["artifacts"]] == [1, 2]
    assert all(isinstance(m["ts"], int) for m in stored["messages"])
