"""@-mentions: files the person tags in a message, attached for the model.

``@notes.md`` or ``@"Q3 report.md"`` names a file in the chat's project
(the composer offers them as the person types ``@``). Each tag that names a
file is read with the read tool, so the same confinement and truncation
apply, and attached after the message in a ``<mentioned-files>`` block.
A tag that names no file stays plain text, as do emails (the ``@`` must
start a word). The history view shows what the person typed: it leaves
the block out (``visible_text``).
"""

from __future__ import annotations

import re
from pathlib import Path

from app.artifacts import resolve_workspace_file
from app.tools.tau_ported import create_read_tool

MAX_MENTIONS = 10
OPEN = "<mentioned-files>"
_SEPARATOR = f"\n\n{OPEN}"
_MENTION = re.compile(r'(?<!\S)@(?:"([^"\n]+)"|([^\s"]+))')
# A bare tag may end a sentence or a bracket: "see @notes.md." Tried
# without them first: Windows finds "notes.md." too, dropping the dot.
_TRAILING = ".,;:!?)]}'"


def mentioned_paths(text: str, folder: Path) -> list[str]:
    """The files ``text`` tags, in order, without repeats, at most MAX_MENTIONS."""
    paths: list[str] = []
    for match in _MENTION.finditer(text):
        quoted, bare = match.groups()
        candidates = [quoted] if quoted else [bare.rstrip(_TRAILING), bare]
        path = next((c for c in candidates if c and resolve_workspace_file(folder, c)), None)
        if path is not None and path not in paths:
            paths.append(path)
    return paths[:MAX_MENTIONS]


async def mentions_block(text: str, folder: Path) -> str | None:
    """The ``<mentioned-files>`` block for the files ``text`` tags, or None."""
    paths = mentioned_paths(text, folder)
    if not paths:
        return None
    read = create_read_tool(cwd=folder)
    files = []
    for path in paths:
        result = await read.execute("mention", {"path": path})
        files.append(f'<file path="{path}">\n{result.text.rstrip()}\n</file>')
    intro = "Files the person tagged with @ in the message above, as they were when it was sent:"
    return "\n\n".join([OPEN, intro, *files, "</mentioned-files>"])


def visible_text(text: str) -> str:
    """A user message as the person typed it, without its attached files."""
    return text.split(_SEPARATOR, 1)[0]
