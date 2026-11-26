"""Artefacts: files the agent creates or changes in the workspace.

A file counts whether it came from ``write``/``edit`` (the file they name)
or from a ``bash`` command, such as a script that saves a chart (what
changed while the command ran). Each becomes an artefact the UI renders
inline, and files are served back through ``resolve_workspace_file``,
which never lets a path escape the workspace.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

# Directories that are tooling noise, not deliverables.
IGNORED_DIRS = frozenset({".git", "node_modules", "__pycache__", ".venv", "venv", ".cache"})
MAX_SCAN_FILES = 5_000

KINDS: dict[str, str] = {
    **dict.fromkeys(("html", "htm"), "html"),
    **dict.fromkeys(("md", "markdown"), "markdown"),
    **dict.fromkeys(("png", "jpg", "jpeg", "gif", "webp", "svg"), "image"),
    **dict.fromkeys(("csv", "tsv"), "table"),
    **dict.fromkeys(("mmd", "mermaid"), "diagram"),
    "pdf": "pdf",
    **dict.fromkeys(
        (
            "py", "js", "mjs", "ts", "tsx", "jsx", "css", "sh", "bash", "json", "yaml",
            "yml", "toml", "sql", "xml", "ini", "rs", "go", "java", "c", "cpp", "h",
            "rb", "php", "txt", "log",
        ),
        "code",
    ),
}  # fmt: skip


def artifact_kind(path: str) -> str:
    return KINDS.get(Path(path).suffix.lower().lstrip("."), "file")


@dataclass(frozen=True, slots=True)
class FileStamp:
    mtime_ns: int
    size: int


def scan_workspace(root: Path) -> dict[str, FileStamp]:
    """Relative path (POSIX) -> stamp for every file worth tracking."""
    stamps: dict[str, FileStamp] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS and not d.startswith(".")]
        for name in filenames:
            if name.startswith("."):
                continue
            full = Path(dirpath) / name
            try:
                stat = full.stat()
            except OSError:
                continue
            stamps[full.relative_to(root).as_posix()] = FileStamp(stat.st_mtime_ns, stat.st_size)
            if len(stamps) >= MAX_SCAN_FILES:
                return stamps
    return stamps


# How each tool changes the workspace. write and edit change the one file
# they name; a bash command may change anything; the rest change nothing.
# A new tool that writes files belongs in one of these.
NAMING_TOOLS = frozenset({"write", "edit"})
COMMAND_TOOLS = frozenset({"bash"})


class ArtifactTracker:
    """The files one run creates or changes: the file each of its write and
    edit calls names, and what changes while each of its bash commands runs.

    Several chats can work at once, so a change between this run's tool
    calls may be another chat's (or the person's own) and isn't counted.
    A file another chat writes while this run's command runs still is: only
    separate folders tell those apart.
    """

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        self._before: dict[str, FileStamp] | None = None

    def tool_started(self, tool_name: str) -> None:
        if tool_name in COMMAND_TOOLS:
            self._before = scan_workspace(self._root)

    def tool_finished(self, tool_name: str, details: object, is_error: bool) -> list[dict[str, object]]:
        if tool_name in NAMING_TOOLS and not is_error:
            return self._named(details)
        if tool_name in COMMAND_TOOLS and self._before is not None:
            before, self._before = self._before, None
            current = scan_workspace(self._root)
            changed = sorted(path for path, stamp in current.items() if before.get(path) != stamp)
            return [_artifact(path, current[path]) for path in changed]
        return []

    def _named(self, details: object) -> list[dict[str, object]]:
        path = details.get("path") if isinstance(details, dict) else None
        if not isinstance(path, str):
            return []
        try:
            full = Path(path).resolve()
            rel = full.relative_to(self._root).as_posix()
            stat = full.stat()
        except (OSError, ValueError):
            return []
        if any(part in IGNORED_DIRS or part.startswith(".") for part in rel.split("/")):
            return []  # scan_workspace leaves these out too
        return [_artifact(rel, FileStamp(stat.st_mtime_ns, stat.st_size))]


def _artifact(path: str, stamp: FileStamp) -> dict[str, object]:
    return {
        "path": path,
        "kind": artifact_kind(path),
        "size": stamp.size,
        "mtime": stamp.mtime_ns // 1_000_000,
        "created_ms": int(time.time() * 1000),
    }


def resolve_workspace_file(root: Path, rel_path: str) -> Path | None:
    """The file at ``rel_path`` inside ``root``, or None if missing or outside it."""
    try:
        candidate = (root / rel_path).resolve()
        candidate.relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    return candidate if candidate.is_file() else None


_HEAD_OPEN = re.compile(r"<head[^>]*>", re.IGNORECASE)
_HTML_OPEN = re.compile(r"<html[^>]*>", re.IGNORECASE)

# Injected into HTML artefacts shown inline: reports the page height so the
# iframe can size to its content, and follows Humphrey's light/dark toggle
# via data-theme on <html>. It also matches the frame's color-scheme to the
# app's, without which browsers paint the frame opaque white. postMessage
# works from the sandbox (opaque origin).
EMBED_SCRIPT = (
    "<script>(()=>{const root=document.documentElement;"
    "const setTheme=t=>{if(t==='light'||t==='dark'){root.dataset.theme=t;root.style.colorScheme=t;}};"
    "setTheme(new URLSearchParams(location.search).get('theme'));"
    "const post=()=>parent.postMessage({humphrey:'artifact-size',"
    "height:Math.ceil(Math.max(root.scrollHeight,document.body?document.body.scrollHeight:0))},'*');"
    "addEventListener('load',post);"
    "if(window.ResizeObserver)new ResizeObserver(post).observe(root);"
    "addEventListener('message',e=>{if(e.data&&e.data.humphrey==='theme')setTheme(e.data.theme);});"
    "})();</script>"
)


def inject_embed_script(html: str) -> str:
    """Insert EMBED_SCRIPT as early as possible without breaking the doctype."""
    for pattern in (_HEAD_OPEN, _HTML_OPEN):
        match = pattern.search(html)
        if match:
            return html[: match.end()] + EMBED_SCRIPT + html[match.end() :]
    return EMBED_SCRIPT + html
