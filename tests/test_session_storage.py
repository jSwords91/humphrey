"""How ObjectStoreSessionStore lays sessions out in storage (app/sessions).

Images used to be stored inline as base64, so a session with a few pictures
was megabytes, rewritten in full on every turn, and the session list read
every session in full just to show its title. Now images are stored once by
content hash, and each session has a small summary object for the list.
These run against the real local-disk ObjectStore in a temp directory.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from app.sessions import MISSING_IMAGE_NOTE, ObjectStoreSessionStore, Session
from app.storage import LocalFileObjectStore
from brain.messages import AssistantMessage, ImageContent, TextContent, UserMessage


class _CountingStore(LocalFileObjectStore):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.gets: list[str] = []
        self.puts: list[str] = []
        self.content_types: dict[str, str] = {}

    def get_bytes(self, key: str) -> bytes | None:
        self.gets.append(key)
        return super().get_bytes(key)

    def put_bytes(self, key: str, data: bytes, *, content_type: str = "") -> None:
        self.puts.append(key)
        self.content_types[key] = content_type
        super().put_bytes(key, data, content_type=content_type)


_SETTINGS = type("S", (), {"sessions_prefix": "sessions"})()


def _store(root: Path) -> tuple[ObjectStoreSessionStore, _CountingStore]:
    backing = _CountingStore(root)
    return ObjectStoreSessionStore(backing, _SETTINGS), backing  # type: ignore[arg-type]


def _png(colour: str = "teal") -> bytes:
    out = BytesIO()
    Image.new("RGB", (8, 6), colour).save(out, format="PNG")
    return out.getvalue()


def _image(blob: bytes) -> ImageContent:
    return ImageContent(data=base64.b64encode(blob).decode("ascii"), mime_type="image/png")


def _session_with_image(session_id: str, blob: bytes) -> Session:
    return Session(
        id=session_id,
        title="Pictures",
        messages=[
            UserMessage(content=[TextContent(text="what is this?"), _image(blob)]),
            AssistantMessage(content="a teal rectangle", model="mock"),
        ],
    )


def _image_keys(backing: _CountingStore) -> list[str]:
    return [k.key for k in backing.list_keys("sessions-images/")]


@pytest.mark.asyncio
async def test_images_are_stored_once_by_hash_and_come_back_intact(tmp_path: Path) -> None:
    store, backing = _store(tmp_path)
    blob = _png()
    session = _session_with_image("pictures00000001", blob)
    await store.save(session)

    digest = hashlib.sha256(blob).hexdigest()
    assert backing.get_bytes(f"sessions-images/{digest}") == blob  # raw bytes, not base64
    assert backing.content_types[f"sessions-images/{digest}"] == "image/png"
    stored = (tmp_path / "sessions" / "pictures00000001.json").read_text()
    assert session.messages[0].images[0].data not in stored
    assert json.loads(stored)["messages"][0]["content"][1] == {
        "type": "image",
        "mimeType": "image/png",
        "ref": digest,
    }

    fetched = await ObjectStoreSessionStore(LocalFileObjectStore(tmp_path), _SETTINGS).get(  # type: ignore[arg-type]
        "pictures00000001"
    )
    assert fetched is not None
    assert fetched.messages == session.messages


@pytest.mark.asyncio
async def test_saving_again_does_not_rewrite_an_image(tmp_path: Path) -> None:
    store, backing = _store(tmp_path)
    session = _session_with_image("pictures00000001", _png())
    await store.save(session)
    session.messages.append(UserMessage(content="and another question"))
    await store.save(session)
    assert len([k for k in backing.puts if k.startswith("sessions-images/")]) == 1

    # A fresh process learns the image is stored when it loads the session.
    store2, backing2 = _store(tmp_path)
    reloaded = await store2.get("pictures00000001")
    assert reloaded is not None
    await store2.save(reloaded)
    assert not [k for k in backing2.puts if k.startswith("sessions-images/")]


@pytest.mark.asyncio
async def test_the_same_image_in_two_sessions_is_stored_once(tmp_path: Path) -> None:
    store, backing = _store(tmp_path)
    blob = _png()
    await store.save(_session_with_image("pictures00000001", blob))
    await store.save(_session_with_image("pictures00000002", blob))
    await store.save(_session_with_image("pictures00000003", _png("coral")))
    assert len(_image_keys(backing)) == 2


@pytest.mark.asyncio
async def test_a_missing_image_becomes_a_note_not_a_lost_session(tmp_path: Path) -> None:
    store, backing = _store(tmp_path)
    blob = _png()
    await store.save(_session_with_image("pictures00000001", blob))
    (tmp_path / "sessions-images" / hashlib.sha256(blob).hexdigest()).unlink()

    fetched = await _store(tmp_path)[0].get("pictures00000001")
    assert fetched is not None
    first = fetched.messages[0]
    assert isinstance(first, UserMessage)
    assert first.images == []
    assert first.text == "what is this?" + MISSING_IMAGE_NOTE
    assert fetched.messages[1].text == "a teal rectangle"


@pytest.mark.asyncio
async def test_a_reference_that_is_not_a_hash_is_never_used_as_a_key(tmp_path: Path) -> None:
    store, backing = _store(tmp_path)
    data = Session(id="pictures00000001", messages=[UserMessage(content="hi")]).to_dict()
    data["messages"][0]["content"] = [{"type": "image", "mimeType": "image/png", "ref": "../../etc/passwd"}]
    backing.put_bytes("sessions/pictures00000001.json", json.dumps(data).encode())

    fetched = await store.get("pictures00000001")
    assert fetched is not None
    assert fetched.messages[0].text == MISSING_IMAGE_NOTE
    assert not [k for k in backing.gets if "passwd" in k]


@pytest.mark.asyncio
async def test_a_session_with_inline_images_loads_and_converts_on_save(tmp_path: Path) -> None:
    """Sessions saved before this format kept base64 inline."""
    store, backing = _store(tmp_path)
    old = _session_with_image("pictures00000001", _png())
    backing.put_bytes("sessions/pictures00000001.json", json.dumps(old.to_dict()).encode())

    fetched = await store.get("pictures00000001")
    assert fetched is not None
    assert fetched.messages == old.messages
    await store.save(fetched)
    assert len(_image_keys(backing)) == 1
    assert old.messages[0].images[0].data not in (tmp_path / "sessions" / "pictures00000001.json").read_text()


@pytest.mark.parametrize("payload", [b"[1, 2, 3]", b'"just a string"', b"not json", b"\xff\xfe"])
@pytest.mark.asyncio
async def test_corrupt_session_data_starts_fresh_without_raising(tmp_path: Path, payload: bytes) -> None:
    """Raising here would read as a storage outage to FallbackSessionStore,
    which would then demote every session in the process to memory."""
    store, backing = _store(tmp_path)
    backing.put_bytes("sessions/pictures00000001.json", payload)
    assert await store.get("pictures00000001") is None


@pytest.mark.asyncio
async def test_the_session_list_reads_summaries_not_sessions(tmp_path: Path) -> None:
    store, backing = _store(tmp_path)
    for i in range(3):
        session = _session_with_image(f"pictures0000000{i}", _png())
        session.title = f"Chat {i}"
        await store.save(session)
        os.utime(tmp_path / "sessions" / f"pictures0000000{i}.json", (1_000 + i, 1_000 + i))

    backing.gets.clear()
    recent = await store.list_recent()
    assert [(s.id, s.title, s.message_count) for s in recent] == [
        ("pictures00000002", "Chat 2", 2),
        ("pictures00000001", "Chat 1", 2),
        ("pictures00000000", "Chat 0", 2),
    ]
    assert all(k.startswith("sessions-summaries/") for k in backing.gets)


@pytest.mark.asyncio
async def test_a_session_without_a_summary_gets_one_the_first_time_it_is_listed(tmp_path: Path) -> None:
    store, backing = _store(tmp_path)
    old = Session(id="oldsession000001", title="From before", messages=[UserMessage(content="hi")])
    backing.put_bytes("sessions/oldsession000001.json", json.dumps(old.to_dict()).encode())

    [listed] = await store.list_recent()
    assert (listed.title, listed.message_count) == ("From before", 1)
    assert "sessions-summaries/oldsession000001.json" in backing.puts

    backing.gets.clear()
    await store.list_recent()
    assert backing.gets == ["sessions-summaries/oldsession000001.json"]


@pytest.mark.asyncio
async def test_a_corrupt_summary_is_rebuilt(tmp_path: Path) -> None:
    store, backing = _store(tmp_path)
    await store.save(Session(id="pictures00000001", title="Mine", messages=[UserMessage(content="hi")]))
    backing.put_bytes("sessions-summaries/pictures00000001.json", b"{not json")
    [listed] = await store.list_recent()
    assert listed.title == "Mine"


@pytest.mark.asyncio
async def test_stray_objects_do_not_take_the_list_slots(tmp_path: Path) -> None:
    """The list used to take the newest keys first and filter after, so a
    stray object could push a real session out of it."""
    store, backing = _store(tmp_path)
    await store.save(Session(id="pictures00000001", title="Real"))
    os.utime(tmp_path / "sessions" / "pictures00000001.json", (1_000, 1_000))
    backing.put_bytes("sessions/notes.txt", b"x")
    backing.put_bytes("sessions/nested/pictures00000009.json", b"{}")

    recent = await store.list_recent(limit=1)
    assert [s.id for s in recent] == ["pictures00000001"]


def test_local_list_keys_walks_only_the_prefix_directory(tmp_path: Path) -> None:
    backing = LocalFileObjectStore(tmp_path)
    backing.put_bytes("sessions/a.json", b"1")
    backing.put_bytes("sessions/deeper/b.json", b"2")
    backing.put_bytes("sessions-images/abc", b"3")
    backing.put_bytes("top.json", b"4")

    assert sorted(m.key for m in backing.list_keys("sessions/")) == [
        "sessions/a.json",
        "sessions/deeper/b.json",
    ]
    assert sorted(m.key for m in backing.list_keys("sessions")) == [
        "sessions-images/abc",
        "sessions/a.json",
        "sessions/deeper/b.json",
    ]
    assert len(backing.list_keys("")) == 4
    assert backing.list_keys("missing/") == []
    with pytest.raises(ValueError):
        backing.list_keys("../outside/")


@pytest.mark.asyncio
async def test_end_to_end_an_image_survives_storage_and_reaches_the_model(
    client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.main
    from app.sessions import reset_session_store_for_tests
    from app.settings import reset_settings_for_tests
    from app.storage import reset_object_store_for_tests
    from providers.mock import MockProvider

    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("LOCAL_STORAGE_DIR", str(tmp_path))
    reset_settings_for_tests()
    reset_object_store_for_tests()
    reset_session_store_for_tests()

    seen: list[list] = []

    class RecordingProvider(MockProvider):
        def stream(self, messages, tools, *, system, signal=None):
            if tools:
                seen.append(list(messages))
            return super().stream(messages, tools, system=system, signal=signal)

    monkeypatch.setattr(app.main, "get_provider", lambda model=None: RecordingProvider())

    png = base64.b64encode(_png()).decode("ascii")
    first = await client.post(
        "/api/agent", json={"message": {"content": "what is this?", "images": [{"data": png}]}}
    )
    assert first.status_code == 200
    session_id = first.headers["x-session-id"]

    root = tmp_path / "ai-logs"
    session_file = root / "sessions" / f"{session_id}.json"
    [image_file] = list((root / "sessions-images").iterdir())
    assert png not in session_file.read_text()
    assert session_file.stat().st_size < 10_000
    assert json.loads((root / "sessions-summaries" / f"{session_id}.json").read_text())["title"]

    # A new process: nothing cached, everything comes back from disk.
    reset_object_store_for_tests()
    reset_session_store_for_tests()
    second = await client.post(
        "/api/agent", json={"message": {"content": "and now?"}, "session_id": session_id}
    )
    assert second.status_code == 200
    replayed = seen[-1][0]
    assert isinstance(replayed, UserMessage)
    assert base64.b64decode(replayed.images[0].data) == image_file.read_bytes()

    listed = (await client.get("/api/sessions")).json()
    assert [s["id"] for s in listed] == [session_id]
    rows = (await client.get(f"/api/sessions/{session_id}")).json()["messages"]
    assert rows[0]["content"] == "what is this?\n(+1 image attached)"
