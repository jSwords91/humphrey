"""ObjectStore: LocalFileObjectStore's file semantics, and get_object_store()'s
STORAGE_BACKEND selection (auto/local/memory - azure needs real credentials
and is covered indirectly via test_sessions.py's fakes instead).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.storage import LocalFileObjectStore, get_object_store


def test_local_store_round_trips_bytes(tmp_path: Path) -> None:
    store = LocalFileObjectStore(tmp_path)
    store.put_bytes("sessions/abc123.json", b'{"id": "abc123"}')

    assert store.get_bytes("sessions/abc123.json") == b'{"id": "abc123"}'
    assert (tmp_path / "sessions" / "abc123.json").is_file()


def test_local_store_get_missing_key_returns_none(tmp_path: Path) -> None:
    store = LocalFileObjectStore(tmp_path)
    assert store.get_bytes("sessions/does-not-exist.json") is None


def test_local_store_overwrite_replaces_content(tmp_path: Path) -> None:
    store = LocalFileObjectStore(tmp_path)
    store.put_bytes("sessions/abc123.json", b"first")
    store.put_bytes("sessions/abc123.json", b"second")
    assert store.get_bytes("sessions/abc123.json") == b"second"


def test_local_store_write_is_atomic_no_tmp_file_left_behind(tmp_path: Path) -> None:
    store = LocalFileObjectStore(tmp_path)
    store.put_bytes("sessions/abc123.json", b"data")
    leftovers = list(tmp_path.rglob("*.tmp"))
    assert leftovers == []


def test_local_store_list_keys_filters_by_prefix_and_sorts_nothing_itself(tmp_path: Path) -> None:
    store = LocalFileObjectStore(tmp_path)
    store.put_bytes("sessions/a.json", b"1")
    store.put_bytes("sessions/b.json", b"2")
    store.put_bytes("2026/01/01/000000_deadbeef.json", b"3")

    session_keys = {m.key for m in store.list_keys("sessions/")}
    assert session_keys == {"sessions/a.json", "sessions/b.json"}

    all_keys = {m.key for m in store.list_keys()}
    assert all_keys == {"sessions/a.json", "sessions/b.json", "2026/01/01/000000_deadbeef.json"}


def test_local_store_list_keys_excludes_tmp_files(tmp_path: Path) -> None:
    store = LocalFileObjectStore(tmp_path)
    store.put_bytes("sessions/a.json", b"1")
    (tmp_path / "sessions" / "stray.json.tmp").write_bytes(b"partial write")

    keys = {m.key for m in store.list_keys()}
    assert keys == {"sessions/a.json"}


def test_local_store_rejects_key_that_escapes_root(tmp_path: Path) -> None:
    store = LocalFileObjectStore(tmp_path)
    with pytest.raises(ValueError):
        store.put_bytes("../escape.json", b"data")


def test_local_store_reports_last_modified(tmp_path: Path) -> None:
    store = LocalFileObjectStore(tmp_path)
    store.put_bytes("sessions/a.json", b"1")
    [meta] = store.list_keys("sessions/")
    assert meta.last_modified is not None
    assert meta.size == 1


def test_get_object_store_memory_backend_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.settings import reset_settings_for_tests
    from app.storage import reset_object_store_for_tests

    monkeypatch.setenv("STORAGE_BACKEND", "memory")
    reset_settings_for_tests()
    reset_object_store_for_tests()
    try:
        assert get_object_store() is None
    finally:
        reset_settings_for_tests()
        reset_object_store_for_tests()


def test_get_object_store_auto_with_nothing_configured_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.settings import reset_settings_for_tests

    # conftest strips AZURE_STORAGE_ACCOUNT_URL, so "auto" has nothing to use
    # and callers fall back to memory.
    monkeypatch.setenv("STORAGE_BACKEND", "auto")
    reset_settings_for_tests()
    assert get_object_store() is None


def test_get_object_store_local_backend_returns_local_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from app.settings import reset_settings_for_tests
    from app.storage import reset_object_store_for_tests

    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("LOCAL_STORAGE_DIR", str(tmp_path))
    # Pinned rather than relying on the ambient default/`.env` value, so this
    # test doesn't depend on what happens to be configured on this machine.
    monkeypatch.setenv("AZURE_STORAGE_AI_LOGS_CONTAINER", "test-container")
    reset_settings_for_tests()
    reset_object_store_for_tests()
    try:
        store = get_object_store()
        assert isinstance(store, LocalFileObjectStore)
        store.put_bytes("sessions/x.json", b"1")
        assert (tmp_path / "test-container" / "sessions" / "x.json").is_file()
    finally:
        reset_settings_for_tests()
        reset_object_store_for_tests()


def test_get_object_store_invalid_backend_value_falls_back_to_local(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from app.settings import get_settings, reset_settings_for_tests
    from app.storage import reset_object_store_for_tests

    monkeypatch.setenv("STORAGE_BACKEND", "not-a-real-backend")
    monkeypatch.setenv("LOCAL_STORAGE_DIR", str(tmp_path))
    reset_settings_for_tests()
    reset_object_store_for_tests()
    try:
        assert get_settings().storage_backend == "local"
        assert isinstance(get_object_store(), LocalFileObjectStore)
    finally:
        reset_settings_for_tests()
        reset_object_store_for_tests()
