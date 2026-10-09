"""Provider contract owned by the portable agent layer.

Renamed from app/providers/types.py during the brain/providers/app reorg.
Mirrors tau_agent/provider.py's role exactly: the loop depends on this
protocol and these chunk shapes to call *any* provider, so the contract
lives here, not in providers/ - concrete adapters in providers/ implement
it, they don't own it (tau_ai/provider.py is just a re-export shim over
tau_agent/provider.py for the same reason).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol

from brain.cancel import CancellationToken
from brain.messages import AgentMessage
from brain.tools import AgentTool


@dataclass(slots=True)
class ToolCallDelta:
    """Incremental tool-call fragment from a model stream."""

    index: int
    id: str | None = None
    name: str | None = None
    arguments: str | None = None


@dataclass(slots=True)
class ModelChunk:
    """One chunk from a provider stream.

    ``thinking`` is a reasoning-summary text delta (Responses API only,
    parallel to ``content``). ``thinking_signature`` is the opaque reasoning
    item to replay next turn — it arrives once, near the end of the turn
    (analogous to ``usage``), not as a stream of deltas.
    """

    content: str | None = None
    tool_calls: list[ToolCallDelta] = field(default_factory=list)
    usage: dict[str, int] | None = None
    thinking: str | None = None
    thinking_signature: str | None = None


class ModelProvider(Protocol):
    """Streams one assistant turn (text and/or tool calls)."""

    @property
    def deployment_name(self) -> str:
        """Label recorded on message_end (deployment id or 'mock')."""

    def stream(
        self,
        messages: list[AgentMessage],
        tools: list[AgentTool],
        *,
        system: str,
        signal: CancellationToken | None = None,
    ) -> AsyncIterator[ModelChunk]:
        """Return an async iterator of ModelChunk."""
        ...
