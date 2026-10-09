"""Artefacts: files the agent creates or changes in the workspace.

After every tool call the workspace is compared with the previous scan, so a
file counts whether it came from ``write``/``edit`` or from a ``bash`` command
(e.g. a script that saves a chart). Each change becomes an artefact the UI
renders inline, and files are served back through ``resolve_workspace_file``,
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


class ArtifactTracker:
    """Reports files created or modified since the previous ``changes()`` call."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._last = scan_workspace(root)

    def changes(self) -> list[dict[str, object]]:
        current = scan_workspace(self._root)
        changed = sorted(path for path, stamp in current.items() if self._last.get(path) != stamp)
        self._last = current
        now_ms = int(time.time() * 1000)
        return [
            {
                "path": path,
                "kind": artifact_kind(path),
                "size": current[path].size,
                "mtime": current[path].mtime_ns // 1_000_000,
                "created_ms": now_ms,
            }
            for path in changed
        ]


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
