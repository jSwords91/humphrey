"""Filesystem and shell tools, ported from tau_coding.tools.

DELIBERATELY NOT WIRED IN. These are not part of ``app.tools.TOOLS`` and
nothing imports this module today. Humphrey's stated design excludes shell
access and filesystem writes on purpose (see knowledge/humphrey.md,
"Intentionally not included") — this module exists so that decision can be
revisited deliberately later, with working code to evaluate, rather than
re-implemented from scratch under time pressure. Wiring any of these into
``TOOLS`` is a decision for whoever enables them, not made by porting them.

Ported from tau's actual four coding tools — read, write, edit, bash — which
is the whole of tau_coding's built-in tool set (confirmed by inspection: no
separate glob/grep/task tools exist there). Adapted to Humphrey's simpler
``AgentTool``/``ToolResult`` shapes (see brain/tools.py) rather than tau's
richer ``ToolDefinition`` wrapper, and with tau's image-reading support in
``read`` dropped entirely. Note this predates Humphrey's own chat-upload
image feature (brain/messages.py's ``ImageContent``, added later for the
web UI) — wiring image support into this ported ``read`` tool would now be
a smaller lift than "from scratch", since the content type it would attach
to already exists.

Before ever adding any of these to TOOLS, at minimum decide and review:
  - What ``cwd`` should scope them to. The factories default to
    ``Path.cwd()`` at call time (tau's own default) — inside the container
    that's wherever uvicorn was launched from (see Dockerfile: /app), which
    is NOT obviously the right scope for a multi-tenant or knowledge-only
    deployment. Nothing here picks a safer default on Humphrey's behalf.
  - Whether `write`/`edit`/`bash` are safe at all given the container's
    filesystem is ephemeral, shared across a single running replica, and
    (per settings.py) reads its own source under /app — a write/edit tool
    can modify the running app's own code path.
  - `bash` is unrestricted shell execution with no allowlist. This is the
    single biggest departure from Humphrey's "platform-safe tools only"
    posture and warrants its own explicit sign-off, separate from read/edit.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import os
import signal as _signal
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic
from typing import Any

from brain.messages import TextContent
from brain.tools import AgentTool, ToolCancellationToken, ToolResult, ToolUpdateCallback

DEFAULT_MAX_OUTPUT_BYTES = 50 * 1024
DEFAULT_MAX_OUTPUT_LINES = 2_000
UTF8_BOM = "﻿"


class ToolInputError(ValueError):
    """Raised internally for invalid arguments; always caught into ToolResult.error()."""


# --- truncation ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TruncationResult:
    content: str
    truncated: bool
    truncated_by: str | None
    total_lines: int
    total_bytes: int
    output_lines: int
    output_bytes: int
    last_line_partial: bool
    first_line_exceeds_limit: bool
    max_lines: int
    max_bytes: int

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def format_size(bytes_count: int) -> str:
    if bytes_count < 1024:
        return f"{bytes_count}B"
    if bytes_count < 1024 * 1024:
        return f"{bytes_count / 1024:.1f}KB"
    return f"{bytes_count / (1024 * 1024):.1f}MB"


def _split_lines_for_counting(content: str) -> list[str]:
    if not content:
        return []
    lines = content.split("\n")
    if content.endswith("\n"):
        lines.pop()
    return lines


def _truncate_string_to_bytes_from_end(text: str, max_bytes: int) -> str:
    encoded = text.encode()
    if len(encoded) <= max_bytes:
        return text
    clipped = encoded[-max_bytes:]
    return clipped.decode(errors="ignore")


def _truncation_result(
    content: str,
    truncated: bool,
    truncated_by: str | None,
    total_lines: int,
    total_bytes: int,
    output_lines: int,
    output_bytes: int,
    *,
    last_line_partial: bool = False,
    first_line: bool = False,
) -> TruncationResult:
    return TruncationResult(
        content=content,
        truncated=truncated,
        truncated_by=truncated_by,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=output_lines,
        output_bytes=output_bytes,
        last_line_partial=last_line_partial,
        first_line_exceeds_limit=first_line,
        max_lines=DEFAULT_MAX_OUTPUT_LINES,
        max_bytes=DEFAULT_MAX_OUTPUT_BYTES,
    )


def truncate_head(
    content: str,
    *,
    max_lines: int = DEFAULT_MAX_OUTPUT_LINES,
    max_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
) -> TruncationResult:
    """Keep the start of ``content``, dropping lines/bytes off the end (used by ``read``)."""
    lines = _split_lines_for_counting(content)
    total_lines = len(lines)
    total_bytes = len(content.encode())
    if total_lines <= max_lines and total_bytes <= max_bytes:
        return _truncation_result(
            content, False, None, total_lines, total_bytes, total_lines, total_bytes
        )

    first_line_bytes = len(lines[0].encode()) if lines else 0
    if first_line_bytes > max_bytes:
        return _truncation_result("", True, "bytes", total_lines, total_bytes, 0, 0, first_line=True)

    output_lines: list[str] = []
    output_bytes = 0
    truncated_by = "lines"
    for index, line in enumerate(lines[:max_lines]):
        line_bytes = len(line.encode()) + (1 if index > 0 else 0)
        if output_bytes + line_bytes > max_bytes:
            truncated_by = "bytes"
            break
        output_lines.append(line)
        output_bytes += line_bytes

    output = "\n".join(output_lines)
    return _truncation_result(
        output, True, truncated_by, total_lines, total_bytes, len(output_lines), len(output.encode())
    )


def truncate_tail(
    content: str,
    *,
    max_lines: int = DEFAULT_MAX_OUTPUT_LINES,
    max_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
) -> TruncationResult:
    """Keep the end of ``content``, dropping lines/bytes off the start (used by ``bash``)."""
    lines = _split_lines_for_counting(content)
    total_lines = len(lines)
    total_bytes = len(content.encode())
    if total_lines <= max_lines and total_bytes <= max_bytes:
        return _truncation_result(
            content, False, None, total_lines, total_bytes, total_lines, total_bytes
        )

    output_lines: list[str] = []
    output_bytes = 0
    truncated_by = "lines"
    last_line_partial = False
    for line in reversed(lines):
        line_bytes = len(line.encode()) + (1 if output_lines else 0)
        if len(output_lines) >= max_lines:
            truncated_by = "lines"
            break
        if output_bytes + line_bytes > max_bytes:
            truncated_by = "bytes"
            if not output_lines:
                clipped = _truncate_string_to_bytes_from_end(line, max_bytes)
                output_lines.insert(0, clipped)
                output_bytes = len(clipped.encode())
                last_line_partial = True
            break
        output_lines.insert(0, line)
        output_bytes += line_bytes

    output = "\n".join(output_lines)
    return _truncation_result(
        output,
        True,
        truncated_by,
        total_lines,
        total_bytes,
        len(output_lines),
        len(output.encode()),
        last_line_partial=last_line_partial,
    )


# --- line endings + edit application ------------------------------------


def detect_line_ending(content: str) -> str:
    crlf_index = content.find("\r\n")
    lf_index = content.find("\n")
    if lf_index == -1 or crlf_index == -1:
        return "\n"
    return "\r\n" if crlf_index < lf_index else "\n"


def normalize_to_lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def restore_line_endings(text: str, ending: str) -> str:
    return text.replace("\n", "\r\n") if ending == "\r\n" else text


def _strip_bom(content: str) -> tuple[str, str]:
    return (UTF8_BOM, content[1:]) if content.startswith(UTF8_BOM) else ("", content)


def _count_occurrences(content: str, text: str) -> int:
    count = 0
    start = 0
    while True:
        index = content.find(text, start)
        if index == -1:
            return count
        count += 1
        start = index + len(text)


def _validate_non_overlapping(spans: list[tuple[int, int, str]]) -> None:
    previous_end = -1
    for start, end, _new_text in sorted(spans):
        if start < previous_end:
            raise ToolInputError("Edits must not overlap")
        previous_end = end


def _not_found_error(path: str, edit_index: int, total_edits: int) -> str:
    if total_edits == 1:
        return (
            f"Could not find the exact text in {path}. The old text must match exactly "
            "including all whitespace and newlines."
        )
    return (
        f"Could not find edits[{edit_index}] in {path}. The oldText must match exactly "
        "including all whitespace and newlines."
    )


def _duplicate_error(path: str, edit_index: int, total_edits: int, occurrences: int) -> str:
    if total_edits == 1:
        return (
            f"Found {occurrences} occurrences of the text in {path}. The text must be unique. "
            "Please provide more context to make it unique."
        )
    return (
        f"Found {occurrences} occurrences of edits[{edit_index}] in {path}. "
        "Each oldText must be unique. Please provide more context to make it unique."
    )


def _empty_old_text_error(path: str, edit_index: int, total_edits: int) -> str:
    if total_edits == 1:
        return f"oldText must not be empty in {path}."
    return f"edits[{edit_index}].oldText must not be empty in {path}."


def _no_change_error(path: str, total_edits: int) -> str:
    if total_edits == 1:
        return (
            f"No changes made to {path}. The replacement produced identical content. "
            "This might indicate an issue with special characters or the text not existing "
            "as expected."
        )
    return f"No changes made to {path}. The replacements produced identical content."


def apply_edits_to_normalized_content(
    normalized_content: str, edits: list[dict[str, str]], path: str
) -> tuple[str, str]:
    """Validate and apply non-overlapping, unique-match edits. Raises ToolInputError."""
    normalized_edits = [
        {"oldText": normalize_to_lf(edit["oldText"]), "newText": normalize_to_lf(edit["newText"])}
        for edit in edits
    ]
    for index, edit in enumerate(normalized_edits):
        if not edit["oldText"]:
            raise ToolInputError(_empty_old_text_error(path, index, len(normalized_edits)))

    matches: list[tuple[int, int, str]] = []
    for index, edit in enumerate(normalized_edits):
        old_text = edit["oldText"]
        occurrences = _count_occurrences(normalized_content, old_text)
        if occurrences == 0:
            raise ToolInputError(_not_found_error(path, index, len(normalized_edits)))
        if occurrences > 1:
            raise ToolInputError(_duplicate_error(path, index, len(normalized_edits), occurrences))
        start = normalized_content.index(old_text)
        matches.append((start, start + len(old_text), edit["newText"]))

    _validate_non_overlapping(matches)
    new_content = normalized_content
    for start, end, new_text in sorted(matches, reverse=True):
        new_content = f"{new_content[:start]}{new_text}{new_content[end:]}"
    if new_content == normalized_content:
        raise ToolInputError(_no_change_error(path, len(normalized_edits)))
    return normalized_content, new_content


def generate_diff_string(old: str, new: str) -> tuple[str, int | None]:
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    diff = "\n".join(difflib.ndiff(old_lines, new_lines))
    first_changed_line: int | None = None
    new_line_number = 0
    for line in difflib.ndiff(old_lines, new_lines):
        if line.startswith("  "):
            new_line_number += 1
        elif line.startswith("+"):
            new_line_number += 1
            if first_changed_line is None:
                first_changed_line = new_line_number
        elif line.startswith("-") and first_changed_line is None:
            first_changed_line = max(new_line_number + 1, 1)
    return diff, first_changed_line


def generate_unified_patch(path: str, old: str, new: str) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True), new.splitlines(keepends=True), fromfile=path, tofile=path
        )
    )


# --- argument helpers -----------------------------------------------------


def _str_arg(arguments: Mapping[str, Any], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str):
        raise ToolInputError(f"{name} must be a string")
    return value


def _path_arg(arguments: Mapping[str, Any], name: str, *, cwd: Path) -> Path:
    value = _str_arg(arguments, name)
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = cwd / path
    return path


def _optional_int_arg(arguments: Mapping[str, Any], name: str) -> int | None:
    value = arguments.get(name)
    if value is None:
        return None
    if not isinstance(value, int):
        raise ToolInputError(f"{name} must be an integer")
    return value


def _optional_float_arg(arguments: Mapping[str, Any], name: str) -> float | None:
    value = arguments.get(name)
    if value is None:
        return None
    if not isinstance(value, int | float):
        raise ToolInputError(f"{name} must be a number")
    return float(value)


def _prepare_edit_arguments(arguments: Mapping[str, Any]) -> Mapping[str, Any]:
    """Accept legacy top-level oldText/newText and JSON-string edits, like tau does."""
    prepared = dict(arguments)
    edits_value = prepared.get("edits")
    if isinstance(edits_value, str):
        try:
            parsed = json.loads(edits_value)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            prepared["edits"] = parsed

    old_text = prepared.get("oldText")
    new_text = prepared.get("newText")
    if isinstance(old_text, str) and isinstance(new_text, str):
        edits = prepared.get("edits")
        edit_list = edits if isinstance(edits, list) else []
        prepared["edits"] = [*edit_list, {"oldText": old_text, "newText": new_text}]
        prepared.pop("oldText", None)
        prepared.pop("newText", None)
    return prepared


def _edits_arg(arguments: Mapping[str, Any]) -> list[dict[str, str]]:
    value = arguments.get("edits")
    if not isinstance(value, list) or not value:
        raise ToolInputError("Edit tool input is invalid. edits must contain at least one replacement.")

    edits: list[dict[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ToolInputError(f"edits[{index}] must be an object")
        old_text = item.get("oldText")
        new_text = item.get("newText")
        if not isinstance(old_text, str) or not isinstance(new_text, str):
            raise ToolInputError(f"edits[{index}].oldText and edits[{index}].newText must be strings")
        edits.append({"oldText": old_text, "newText": new_text})
    return edits


# --- file locking (per-path, in-process) -----------------------------------

_file_locks: dict[Path, asyncio.Lock] = {}


class _FileLockContext:
    def __init__(self, path: Path) -> None:
        self._path = path.resolve()
        self._lock: asyncio.Lock | None = None

    async def __aenter__(self) -> None:
        lock = _file_locks.setdefault(self._path, asyncio.Lock())
        self._lock = lock
        await lock.acquire()

    async def __aexit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        if self._lock is not None:
            self._lock.release()


def _file_lock(path: Path) -> _FileLockContext:
    return _FileLockContext(path)


# --- bash subprocess helpers ------------------------------------------------


def _prefixed_shell_command(command: str, prefix: str | None) -> str:
    if prefix is None:
        return command
    return f"{prefix}\n{command}"


def append_status_block(text: str, status: str) -> str:
    return f"{text}\n\n{status}" if text else status


def _kill_process_tree(process: asyncio.subprocess.Process) -> None:
    if os.name == "posix":
        try:
            os.killpg(process.pid, _signal.SIGKILL)
        except ProcessLookupError:
            return
    else:
        try:
            process.kill()
        except ProcessLookupError:
            return


async def _wait_for_cancel(signal: ToolCancellationToken) -> None:
    while not signal.is_cancelled():
        await asyncio.sleep(0.05)


async def _communicate_with_cancellation(
    process: asyncio.subprocess.Process,
    *,
    timeout: float | None,
    signal: ToolCancellationToken | None,
) -> tuple[bytes, bool, bool]:
    communicate = asyncio.create_task(process.communicate())
    cancel_watch: asyncio.Task[None] | None = None
    try:
        wait_for: set[asyncio.Task[Any]] = {communicate}
        if signal is not None:
            cancel_watch = asyncio.create_task(_wait_for_cancel(signal))
            wait_for.add(cancel_watch)

        done, _pending = await asyncio.wait(
            wait_for, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
        )
        if communicate in done:
            output_bytes, _stderr = communicate.result()
            return output_bytes, False, False

        cancelled = cancel_watch is not None and cancel_watch in done
        _kill_process_tree(process)
        try:
            output_bytes, _stderr = await communicate
        except asyncio.CancelledError:
            output_bytes = b""
        return output_bytes, not cancelled, cancelled
    except asyncio.CancelledError:
        _kill_process_tree(process)
        if not communicate.done():
            communicate.cancel()
        raise
    finally:
        if cancel_watch is not None:
            cancel_watch.cancel()


def _write_temp_output(output: str) -> str:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix="humphrey-bash-", suffix=".log", delete=False
    ) as handle:
        handle.write(output)
        return handle.name


# --- tool factories ---------------------------------------------------------


def create_read_tool(*, cwd: str | Path | None = None) -> AgentTool:
    """Read a UTF-8 text file, with offset/limit and head-truncation.

    No image support (unlike tau's version) — Humphrey's message model has
    no image content type to return one as.
    """
    root = Path.cwd() if cwd is None else Path(cwd)

    async def execute(
        _tool_call_id: str,
        arguments: Mapping[str, Any],
        signal: ToolCancellationToken | None = None,
        _on_update: ToolUpdateCallback | None = None,
    ) -> ToolResult:
        _ = signal
        try:
            path = _path_arg(arguments, "path", cwd=root)
            offset = _optional_int_arg(arguments, "offset")
            limit = _optional_int_arg(arguments, "limit")
            if offset is not None and offset < 0:
                raise ToolInputError("offset must be at least 0")
            if limit is not None and limit < 1:
                raise ToolInputError("limit must be at least 1")
            if not path.exists():
                raise ToolInputError(f"File not found: {path}")
            if path.is_dir():
                raise ToolInputError(f"Path is a directory: {path}")

            text = path.read_bytes().decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
            all_lines = text.split("\n")
            start_line = 0 if offset is None or offset == 0 else offset - 1
            if start_line >= len(all_lines):
                raise ToolInputError(
                    f"Offset {offset} is beyond end of file ({len(all_lines)} lines total)"
                )

            user_limited_lines: int | None = None
            if limit is not None:
                end_line = min(start_line + limit, len(all_lines))
                selected = "\n".join(all_lines[start_line:end_line])
                user_limited_lines = end_line - start_line
            else:
                selected = "\n".join(all_lines[start_line:])
        except ToolInputError as exc:
            return ToolResult.error(str(exc))
        except UnicodeDecodeError:
            return ToolResult.error(f"Not a UTF-8 text file: {path}")

        truncation = truncate_head(selected)
        start_display = start_line + 1
        details: dict[str, Any] = {"path": str(path), "truncation": truncation.to_json()}

        if truncation.first_line_exceeds_limit:
            first_line_size = format_size(len(all_lines[start_line].encode()))
            output = (
                f"[Line {start_display} is {first_line_size}, exceeds "
                f"{format_size(DEFAULT_MAX_OUTPUT_BYTES)} limit.]"
            )
        elif truncation.truncated:
            end_display = start_display + truncation.output_lines - 1
            next_offset = end_display + 1
            output = truncation.content
            if truncation.truncated_by == "lines":
                output += (
                    f"\n\n[Showing lines {start_display}-{end_display} of {len(all_lines)}. "
                    f"Use offset={next_offset} to continue.]"
                )
            else:
                output += (
                    f"\n\n[Showing lines {start_display}-{end_display} of {len(all_lines)} "
                    f"({format_size(DEFAULT_MAX_OUTPUT_BYTES)} limit). "
                    f"Use offset={next_offset} to continue.]"
                )
        elif user_limited_lines is not None and start_line + user_limited_lines < len(all_lines):
            remaining = len(all_lines) - (start_line + user_limited_lines)
            next_offset = start_line + user_limited_lines + 1
            output = (
                f"{truncation.content}\n\n"
                f"[{remaining} more lines in file. Use offset={next_offset} to continue.]"
            )
        else:
            output = truncation.content

        return ToolResult(content=[TextContent(text=output)], details=details, is_error=False)

    return AgentTool(
        name="read",
        label="Reading {path}",
        description=(
            "Read the contents of a UTF-8 text file. Output is truncated to "
            f"{DEFAULT_MAX_OUTPUT_LINES} lines or {DEFAULT_MAX_OUTPUT_BYTES // 1024}KB "
            "(whichever is hit first). Use offset/limit for large files."
        ),
        parameters={
            "path": {"type": "string", "description": "Path to the file to read"},
            "offset": {"type": "integer", "description": "1-indexed line number to start reading from"},
            "limit": {"type": "integer", "description": "Maximum number of lines to read"},
        },
        required=("path",),
        execute_fn=execute,
        prompt_snippet="Read file contents",
        prompt_guidelines=("Use read to examine files instead of a shell command.",),
    )


def create_write_tool(*, cwd: str | Path | None = None) -> AgentTool:
    """Write (create or overwrite) a UTF-8 text file. No sandboxing beyond ``cwd`` resolution."""
    root = Path.cwd() if cwd is None else Path(cwd)

    async def execute(
        _tool_call_id: str,
        arguments: Mapping[str, Any],
        signal: ToolCancellationToken | None = None,
        _on_update: ToolUpdateCallback | None = None,
    ) -> ToolResult:
        _ = signal
        try:
            path = _path_arg(arguments, "path", cwd=root)
            content = _str_arg(arguments, "content")
        except ToolInputError as exc:
            return ToolResult.error(str(exc))

        async with _file_lock(path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

        return ToolResult(
            content=[TextContent(text=f"Successfully wrote to {path}.")],
            details={"path": str(path), "characters": len(content)},
            is_error=False,
        )

    return AgentTool(
        name="write",
        label="Writing {path}",
        description=(
            "Write content to a file. Creates the file if it doesn't exist, overwrites if it "
            "does. Automatically creates parent directories."
        ),
        parameters={
            "path": {"type": "string", "description": "Path to the file to write"},
            "content": {"type": "string", "description": "Content to write to the file"},
        },
        required=("path", "content"),
        execute_fn=execute,
        prompt_snippet="Create or overwrite files",
        prompt_guidelines=("Use write only for new files or complete rewrites.",),
    )


def create_edit_tool(*, cwd: str | Path | None = None) -> AgentTool:
    """Apply one or more exact, unique, non-overlapping text replacements to one file."""
    root = Path.cwd() if cwd is None else Path(cwd)

    async def execute(
        _tool_call_id: str,
        arguments: Mapping[str, Any],
        signal: ToolCancellationToken | None = None,
        _on_update: ToolUpdateCallback | None = None,
    ) -> ToolResult:
        _ = signal
        try:
            prepared = _prepare_edit_arguments(arguments)
            path = _path_arg(prepared, "path", cwd=root)
            edits = _edits_arg(prepared)
            if not path.exists():
                raise ToolInputError(f"Could not edit file: {path}. File not found.")
            if path.is_dir():
                raise ToolInputError(f"Could not edit file: {path}. Path is a directory.")

            async with _file_lock(path):
                # NOT path.read_text(): its universal-newlines mode silently
                # rewrites \r\n to \n on the way in, before detect_line_ending
                # ever sees a \r - which defeats CRLF preservation entirely.
                # This is a real bug in tau's own identical read_text() call,
                # caught by this port's own tests; fixed here, not carried
                # over, since there's no reason to reproduce it faithfully.
                raw_content = path.read_bytes().decode("utf-8")
                bom, content = _strip_bom(raw_content)
                original_ending = detect_line_ending(content)
                normalized = normalize_to_lf(content)
                base_content, new_content = apply_edits_to_normalized_content(
                    normalized, edits, str(path)
                )
                final_content = bom + restore_line_endings(new_content, original_ending)
                path.write_text(final_content, encoding="utf-8")
        except ToolInputError as exc:
            return ToolResult.error(str(exc))

        diff_text, first_changed_line = generate_diff_string(base_content, new_content)
        patch = generate_unified_patch(str(path), base_content, new_content)
        return ToolResult(
            content=[TextContent(text=f"Successfully replaced {len(edits)} block(s) in {path}.")],
            details={
                "path": str(path),
                "edits": len(edits),
                "diff": diff_text,
                "patch": patch,
                "first_changed_line": first_changed_line,
            },
            is_error=False,
        )

    return AgentTool(
        name="edit",
        label="Editing {path}",
        description=(
            "Edit a single file using exact text replacement. Every edits[].oldText must match "
            "a unique, non-overlapping region of the original file."
        ),
        parameters={
            "path": {"type": "string", "description": "Path to the file to edit"},
            "edits": {
                "type": "array",
                "description": "One or more targeted replacements.",
                "items": {
                    "type": "object",
                    "properties": {
                        "oldText": {"type": "string"},
                        "newText": {"type": "string"},
                    },
                    "required": ["oldText", "newText"],
                    "additionalProperties": False,
                },
            },
        },
        required=("path", "edits"),
        execute_fn=execute,
        prompt_snippet="Make precise file edits with exact text replacement",
        prompt_guidelines=(
            "Use edit for precise changes (edits[].oldText must match exactly).",
            "When changing multiple separate locations in one file, use one edit call with "
            "multiple entries in edits[] instead of multiple edit calls.",
            "Keep edits[].oldText as small as possible while still being unique in the file.",
        ),
    )


def create_bash_tool(
    *, cwd: str | Path | None = None, shell_command_prefix: str | None = None
) -> AgentTool:
    """Run a shell command. Unrestricted — no allowlist, no sandbox beyond ``cwd``.

    This is the single riskiest tool in this module. See the module docstring
    before ever wiring this into TOOLS.
    """
    root = Path.cwd() if cwd is None else Path(cwd)
    prefix = shell_command_prefix.strip() if shell_command_prefix else None

    async def execute(
        _tool_call_id: str,
        arguments: Mapping[str, Any],
        signal: ToolCancellationToken | None = None,
        _on_update: ToolUpdateCallback | None = None,
    ) -> ToolResult:
        try:
            command = _str_arg(arguments, "command")
            timeout = _optional_float_arg(arguments, "timeout")
            if timeout is not None and timeout <= 0:
                raise ToolInputError("timeout must be greater than 0")
        except ToolInputError as exc:
            return ToolResult.error(str(exc))
        if signal is not None and signal.is_cancelled():
            return ToolResult.error("Command cancelled")

        shell_command = _prefixed_shell_command(command, prefix)
        start = monotonic()
        if os.name == "posix":
            process = await asyncio.create_subprocess_shell(
                shell_command,
                cwd=root,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
                executable="bash" if prefix else None,
            )
        else:
            process = await asyncio.create_subprocess_shell(
                shell_command,
                cwd=root,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        output_bytes, timed_out, cancelled = await _communicate_with_cancellation(
            process, timeout=timeout, signal=signal
        )

        output = output_bytes.decode(errors="replace")
        truncation = truncate_tail(output)
        full_output_path: str | None = None
        output_text = truncation.content or "(no output)"
        if truncation.truncated:
            full_output_path = _write_temp_output(output)
            start_line = truncation.total_lines - truncation.output_lines + 1
            end_line = truncation.total_lines
            if truncation.last_line_partial:
                output_text += (
                    f"\n\n[Showing last {format_size(truncation.output_bytes)} of line {end_line}. "
                    f"Full output: {full_output_path}]"
                )
            elif truncation.truncated_by == "lines":
                output_text += (
                    f"\n\n[Showing lines {start_line}-{end_line} of {truncation.total_lines}. "
                    f"Full output: {full_output_path}]"
                )
            else:
                output_text += (
                    f"\n\n[Showing lines {start_line}-{end_line} of {truncation.total_lines} "
                    f"({format_size(DEFAULT_MAX_OUTPUT_BYTES)} limit). "
                    f"Full output: {full_output_path}]"
                )

        exit_code = process.returncode
        status: str | None = None
        if timed_out:
            status = f"Command timed out after {timeout:g} seconds" if timeout else "Command timed out"
        elif cancelled:
            status = "Command cancelled"
        elif exit_code not in (0, None):
            status = f"Command exited with code {exit_code}"
        if status:
            output_text = append_status_block(output_text, status)

        # tau's AgentToolResult has no is_error field at all - a deliberate
        # difference here, since Humphrey's UI uses it to style tool cards.
        # Only a genuine nonzero exit counts; a timeout or cancellation is a
        # reported status, not a tool malfunction.
        return ToolResult(
            content=[TextContent(text=output_text)],
            details={
                "command": command,
                "exit_code": exit_code,
                "timed_out": timed_out,
                "cancelled": cancelled,
                "duration_seconds": round(monotonic() - start, 3),
                "truncation": truncation.to_json(),
                "full_output_path": full_output_path,
                "shell_command_prefix_applied": prefix is not None,
            },
            is_error=bool(status) and not timed_out and not cancelled,
        )

    return AgentTool(
        name="bash",
        label="Running {description}",
        description=(
            "Execute a bash command. Returns combined stdout/stderr. Output is truncated to "
            f"last {DEFAULT_MAX_OUTPUT_LINES} lines or {DEFAULT_MAX_OUTPUT_BYTES // 1024}KB "
            "(whichever is hit first); full output is saved to a temp file if truncated. "
            "Optionally provide a timeout in seconds."
        ),
        parameters={
            "command": {"type": "string", "description": "Bash command to execute"},
            "description": {
                "type": "string",
                "description": (
                    "Brief present-participle summary of the command's purpose, "
                    "e.g. 'Running tests'"
                ),
            },
            "timeout": {"type": "number", "description": "Timeout in seconds (optional, no default)"},
        },
        required=("command", "description"),
        execute_fn=execute,
        prompt_snippet="Execute bash commands (ls, grep, find, etc.)",
        prompt_guidelines=(
            "Include a brief present-participle description of the command's purpose.",
        ),
    )


def create_coding_tools(
    *, cwd: str | Path | None = None, shell_command_prefix: str | None = None
) -> list[AgentTool]:
    """The full ported set, in tau's own order: read, write, edit, bash.

    Not wired into anything — see the module docstring.
    """
    root = Path.cwd() if cwd is None else Path(cwd)
    return [
        create_read_tool(cwd=root),
        create_write_tool(cwd=root),
        create_edit_tool(cwd=root),
        create_bash_tool(cwd=root, shell_command_prefix=shell_command_prefix),
    ]
