"""The tool contract the portable agent loop depends on.

Only the protocol/shape lives here (mirrors tau_agent/tools.py) - the
concrete Humphrey tools (read, write, edit, bash) are an
application concern and live in app/tools/, built on top of AgentTool.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import Field

from brain.messages import TextContent, WireModel

ToolHandler = Callable[[dict[str, Any]], str]
ToolUpdateCallback = Callable[["ToolResult"], None]


class ToolCancellationToken(Protocol):
    def is_cancelled(self) -> bool: ...


class ToolResult(WireModel):
    """Structured tool outcome for the model transcript and UI.

    Content-block based to match tau's ``AgentToolResult`` — ``details``
    stays a convenience for callers that want the parsed payload without
    re-parsing ``.text``.
    """

    content: list[TextContent] = Field(default_factory=list)
    details: Any = None
    is_error: bool = False

    @property
    def text(self) -> str:
        return "".join(b.text for b in self.content)

    @classmethod
    def ok(cls, payload: Any) -> ToolResult:
        if isinstance(payload, str):
            return cls(content=[TextContent(text=payload)] if payload else [], details=None)
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        return cls(content=[TextContent(text=text)], details=payload)

    @classmethod
    def error(cls, message: str) -> ToolResult:
        payload = {"error": message}
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        return cls(content=[TextContent(text=text)], details=payload, is_error=True)


ToolExecutor = Callable[
    [str, Mapping[str, Any], ToolCancellationToken | None, ToolUpdateCallback | None],
    Awaitable[ToolResult],
]


@dataclass(frozen=True, slots=True)
class AgentTool:
    """A tool exposed to the portable agent loop."""

    name: str
    label: str
    description: str
    parameters: Mapping[str, Any]
    execute_fn: ToolExecutor
    required: tuple[str, ...] = ()
    prompt_snippet: str | None = None
    prompt_guidelines: tuple[str, ...] = ()

    @property
    def prompt_line(self) -> str:
        """One line for the system prompt's tool list.

        Falls back to the OpenAI-facing ``description`` when no shorter
        ``prompt_snippet`` is given, so a tool can never be silently missing
        from the assembled prompt.
        """
        return self.prompt_snippet or self.description

    def label_for(self, args: Mapping[str, Any] | None = None) -> str:
        values = {k: _shorten(str(v)) for k, v in (args or {}).items()}
        try:
            return self.label.format_map(_SafeFormat(values))
        except Exception:  # noqa: BLE001
            return self.name

    async def execute(
        self,
        tool_call_id: str,
        arguments: Mapping[str, Any],
        signal: ToolCancellationToken | None = None,
        on_update: ToolUpdateCallback | None = None,
    ) -> ToolResult:
        return await self.execute_fn(tool_call_id, arguments, signal, on_update)

    def openai_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": dict(self.parameters),
            "required": list(self.required),
            "additionalProperties": False,
        }



_LABEL_VALUE_MAX = 48


def _shorten(value: str) -> str:
    """Trim a label value; paths keep their tail, where the file name is."""
    if len(value) <= _LABEL_VALUE_MAX:
        return value
    if "/" in value and " " not in value:
        return "…" + value[-(_LABEL_VALUE_MAX - 1) :]
    return value[:_LABEL_VALUE_MAX]


class _SafeFormat(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return "…"
