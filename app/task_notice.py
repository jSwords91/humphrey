"""The message a finished background task sends the chat that started it.

As in tau-subagents: a ``<task-notification>`` block, delivered as a
follow-up, so Humphrey replies with the result as soon as it's there. The
result is fit-or-fetch: one that fits goes in whole, and one that doesn't
goes in not at all, with a pointer to get_task_result. A cut-off preview
would let Humphrey act on half of it.

The block is what the model reads; ``parse`` turns it back into fields for
the card the person sees.
"""

from __future__ import annotations

import re
from typing import Literal, TypedDict

RESULT_CAP = 500
OPEN = "<task-notification>"

Status = Literal["completed", "failed", "stopped"]


class Notice(TypedDict):
    task_id: str
    description: str
    status: str
    result: str
    # True when the result didn't fit: Humphrey fetches it with get_task_result.
    pending: bool


def format_notice(task_id: str, description: str, status: Status, result: str) -> str:
    if len(result) <= RESULT_CAP:
        body = f"<result>\n{result}\n</result>"
    else:
        body = (
            f"<result-pending>The result is {len(result)} characters, too long to include. "
            f"Call get_task_result with task id {task_id} before acting on it.</result-pending>"
        )
    return (
        f"{OPEN}\n<task-id>{task_id}</task-id>\n<description>{_plain(description)}</description>\n"
        f"<status>{status}</status>\n{body}\n</task-notification>"
    )


def parse(text: str) -> Notice | None:
    """The notice's fields, or None if ``text`` isn't one."""
    if not text.startswith(OPEN):
        return None

    def field(tag: str) -> str:
        match = re.search(rf"<{tag}>\n?(.*?)\n?</{tag}>", text, re.DOTALL)
        return match.group(1) if match else ""

    return Notice(
        task_id=field("task-id"),
        description=field("description"),
        status=field("status"),
        result=field("result"),
        pending="<result-pending>" in text,
    )


def _plain(text: str) -> str:
    """One line, with no tag of its own to break the block."""
    return " ".join(text.replace("<", "‹").replace(">", "›").split())
