"""Cooperative cancellation for provider streams and tool execution."""

from __future__ import annotations

from typing import Protocol


class CancellationToken(Protocol):
    def is_cancelled(self) -> bool:
        """Return whether the current run should stop."""
        ...


class SimpleCancellationToken:
    """Mutable cancel flag shared by harness, loop, provider, and tools."""

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def is_cancelled(self) -> bool:
        return self._cancelled
