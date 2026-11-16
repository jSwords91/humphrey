"""In-process registry of active runs: the harness (for mid-run steer and
follow-up) and its UI bridge (for answering a question the run is waiting on)."""

from __future__ import annotations

from threading import Lock

from app.ui_bridge import WebUiBridge
from brain.harness import AgentHarness

_lock = Lock()
_active: dict[str, AgentHarness] = {}
_bridges: dict[str, WebUiBridge] = {}


def register_active(
    session_id: str, harness: AgentHarness, bridge: WebUiBridge | None = None
) -> None:
    with _lock:
        _active[session_id] = harness
        if bridge is not None:
            _bridges[session_id] = bridge
        else:
            _bridges.pop(session_id, None)


def unregister_active(session_id: str, harness: AgentHarness | None = None) -> None:
    with _lock:
        current = _active.get(session_id)
        if current is None:
            return
        if harness is None or current is harness:
            _active.pop(session_id, None)
            _bridges.pop(session_id, None)


def get_active(session_id: str) -> AgentHarness | None:
    with _lock:
        return _active.get(session_id)


def get_bridge(session_id: str) -> WebUiBridge | None:
    with _lock:
        return _bridges.get(session_id)


def clear_active_for_tests() -> None:
    with _lock:
        _active.clear()
        _bridges.clear()
