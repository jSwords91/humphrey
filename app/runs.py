"""In-process registry of active harnesses for mid-run steer / follow-up."""

from __future__ import annotations

from threading import Lock

from brain.harness import AgentHarness

_lock = Lock()
_active: dict[str, AgentHarness] = {}


def register_active(session_id: str, harness: AgentHarness) -> None:
    with _lock:
        _active[session_id] = harness


def unregister_active(session_id: str, harness: AgentHarness | None = None) -> None:
    with _lock:
        current = _active.get(session_id)
        if current is None:
            return
        if harness is None or current is harness:
            _active.pop(session_id, None)


def get_active(session_id: str) -> AgentHarness | None:
    with _lock:
        return _active.get(session_id)


def clear_active_for_tests() -> None:
    with _lock:
        _active.clear()
