"""Humphrey's concrete tools — assembled from app/tools/platform.py and
app/tools/examples.py's handlers into AgentTool instances (schemas + async
executors). AgentTool/ToolResult themselves live in brain/tools.py - this
package only builds this application's specific tool list on top of them.

app/tools/tau_ported.py (filesystem/shell tools ported from tau, not wired
in here on purpose) is a separate, deliberately-unused module - see its own
docstring.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from typing import Any

from brain.cancel import CancellationToken
from brain.messages import TextContent
from brain.tools import AgentTool, ToolCancellationToken, ToolExecutor, ToolResult, ToolUpdateCallback
from app.tools.examples import _calculate, _get_weather
from app.tools.platform import (
    _capabilities,
    _status,
    _list_docs,
    _read_doc,
    _read_skill,
    _read_skill_asset,
    _search_docs,
)

__all__ = [
    "TOOLS",
    "OPENAI_TOOL_SCHEMAS",
    "execute_tool_async",
    "get_tool",
    "tool_label",
    "tool_names",
]

ToolHandler = Callable[[dict[str, Any]], str]


def _wrap_sync(handler: ToolHandler) -> ToolExecutor:
    async def _execute(
        tool_call_id: str,
        arguments: Mapping[str, Any],
        signal: ToolCancellationToken | None = None,
        on_update: ToolUpdateCallback | None = None,
    ) -> ToolResult:
        _ = tool_call_id, on_update
        if signal is not None and signal.is_cancelled():
            return ToolResult.error("Operation aborted")
        try:
            raw = await asyncio.to_thread(handler, dict(arguments))
        except Exception as exc:  # noqa: BLE001
            return ToolResult.error(str(exc))
        if signal is not None and signal.is_cancelled():
            return ToolResult.error("Operation aborted")
        content = [TextContent(text=raw)] if raw else []
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return ToolResult(content=content, details=None, is_error=False)
        is_error = isinstance(parsed, dict) and "error" in parsed
        return ToolResult(content=content, details=parsed, is_error=is_error)

    return _execute


def _tool(
    name: str,
    label: str,
    description: str,
    handler: ToolHandler,
    properties: dict[str, Any] | None = None,
    required: tuple[str, ...] = (),
    snippet: str | None = None,
    guidelines: tuple[str, ...] = (),
) -> AgentTool:
    return AgentTool(
        name=name,
        label=label,
        description=description,
        parameters=properties or {},
        required=required,
        execute_fn=_wrap_sync(handler),
        prompt_snippet=snippet,
        prompt_guidelines=guidelines,
    )


TOOLS: list[AgentTool] = [
    _tool(
        "capabilities",
        "Checking capabilities",
        "List tools and limits for this agent. Use when the user asks what you can do.",
        _capabilities,
        snippet="what you can do",
    ),
    _tool(
        "status",
        "Checking status",
        "Report runtime status: mock mode, OpenAI deployment, storage wiring, UTC time. "
        "Use for health, config, or environment questions.",
        _status,
        snippet="live config: mock mode, deployment, storage wiring",
        guidelines=('For "is storage configured?" or "which deployment?": call status.',),
    ),
    _tool(
        "list_docs",
        "Listing docs",
        "List bundled knowledge documents. "
        "Prefer this before answering how-to questions, then read_doc.",
        _list_docs,
        snippet="list bundled knowledge docs",
    ),
    _tool(
        "read_doc",
        "Reading {path}",
        "Read a bundled knowledge markdown file. Paths are filenames only, e.g. deploy.md.",
        _read_doc,
        {
            "path": {
                "type": "string",
                "description": "Document filename under knowledge/, e.g. humphrey.md",
            },
            "offset": {
                "type": "integer",
                "description": "1-indexed start line (0 or omit = start of file)",
            },
            "limit": {
                "type": "integer",
                "description": "Max lines to return (default 120, max 400)",
            },
        },
        required=("path",),
        snippet="read one bundled knowledge doc",
    ),
    _tool(
        "search_docs",
        "Searching docs for “{query}”",
        "Keyword search across bundled knowledge docs. Returns matching snippets with paths.",
        _search_docs,
        {
            "query": {
                "type": "string",
                "description": "Search terms, e.g. 'storage backend' or 'skills'",
            },
            "max_hits": {
                "type": "integer",
                "description": "Maximum hits to return (default 8, max 20)",
            },
        },
        required=("query",),
        snippet="keyword search across bundled knowledge docs",
        guidelines=(
            "For questions about Humphrey itself (setup, config, architecture): search or list docs, "
            "then read the relevant file.",
        ),
    ),
    _tool(
        "read_skill",
        "Loading skill {name}",
        "Load the full instructions for a specialized skill named in <available_skills>. "
        "Call this when a task matches a skill's description, then follow its instructions.",
        _read_skill,
        {
            "name": {
                "type": "string",
                "description": "Skill name as listed in <available_skills>, e.g. mermaid-diagrams",
            },
        },
        required=("name",),
        snippet="load a specialized skill's full instructions",
    ),
    _tool(
        "read_skill_asset",
        "Reading {path} ({name})",
        "Read a bundled file that a skill's own instructions reference by relative path "
        "(e.g. a reference table or worked example shipped alongside the skill). The path "
        "is relative to that skill's own directory; files outside it are not accessible.",
        _read_skill_asset,
        {
            "name": {
                "type": "string",
                "description": "The skill this asset belongs to, e.g. mermaid-diagrams",
            },
            "path": {
                "type": "string",
                "description": "Path to the asset, relative to the skill's own directory, "
                "e.g. examples/flowchart.md",
            },
            "offset": {
                "type": "integer",
                "description": "1-indexed start line (0 or omit = start of file)",
            },
            "limit": {
                "type": "integer",
                "description": "Max lines to return (default 200, max 2000)",
            },
        },
        required=("name", "path"),
        snippet="read a file bundled alongside a skill",
        guidelines=(
            "When a skill's own instructions reference a bundled file by relative path "
            "(e.g. `examples/flowchart.md`), use read_skill_asset with that skill's "
            "name and the path — read_doc only sees knowledge/, not skill directories.",
        ),
    ),
    _tool(
        "get_weather",
        "Checking weather in {location}",
        "Get current conditions and the next few hours' forecast (temperature, "
        "precipitation, wind) for a named place, via the free Open-Meteo API.",
        _get_weather,
        {
            "location": {
                "type": "string",
                "description": "Place name, e.g. 'London' or 'Paris, France'",
            },
        },
        required=("location",),
        snippet="current weather and short forecast for a place",
    ),
    _tool(
        "calculate",
        "Calculating {expression}",
        "Evaluate an arithmetic expression exactly. Supports + - * / // % ** (or ^), "
        "parentheses, pi, e, and sqrt/log/log10/exp/sin/cos/tan/floor/ceil/abs/round.",
        _calculate,
        {
            "expression": {
                "type": "string",
                "description": "Arithmetic expression, e.g. '(3 + 4) * 2' or 'sqrt(2) * pi'",
            },
        },
        required=("expression",),
        snippet="evaluate arithmetic exactly",
        guidelines=("For any non-trivial arithmetic, use calculate rather than working it out yourself.",),
    ),
]

_TOOLS_BY_NAME: dict[str, AgentTool] = {tool.name: tool for tool in TOOLS}


def get_tool(name: str) -> AgentTool | None:
    return _TOOLS_BY_NAME.get(name)


def tool_names() -> list[str]:
    return [tool.name for tool in TOOLS]


def tool_label(name: str, args: dict[str, Any] | None = None) -> str:
    tool = _TOOLS_BY_NAME.get(name)
    if tool is None:
        return name
    return tool.label_for(args)


# Back-compat for anything still importing schemas / sync execute.
OPENAI_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.openai_parameters(),
        },
    }
    for tool in TOOLS
]


async def execute_tool_async(
    name: str,
    arguments: Mapping[str, Any] | str,
    *,
    call_id: str = "",
    signal: CancellationToken | None = None,
) -> ToolResult:
    tool = _TOOLS_BY_NAME.get(name)
    if tool is None:
        return ToolResult.error(f"unknown tool: {name}")
    if isinstance(arguments, str):
        try:
            parsed: Any = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as exc:
            return ToolResult.error(f"invalid tool arguments JSON: {exc}")
        if not isinstance(parsed, dict):
            return ToolResult.error("tool arguments must be a JSON object")
        args = parsed
    else:
        args = dict(arguments)
    return await tool.execute(call_id, args, signal, None)
