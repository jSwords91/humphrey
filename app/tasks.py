"""Background tasks: Humphrey hands a job to a chat of its own and carries on.

Modelled on tau-subagents (github.com/rian-dolphin/tau-subagents), in the
background only:

- A task is a real chat (``Session.parent_id`` names the chat that started
  it), working in the same project. It starts fresh with the brief, or as
  a fork: with a copy of the conversation so far, like tau's ``fork`` type,
  so the model provider can reuse the parent's cached prompt.
- At most ``MAX_RUNNING`` work at once; the rest wait their turn, first
  come first served, and show as queued.
- When one finishes, a ``<task-notification>`` (app/task_notice.py) goes
  to the chat that started it as a follow-up: into the turn under way
  there, or as a turn of its own, so Humphrey replies with the result
  whether or not anyone is watching.

Tasks live in this process, as runs do (app/runs.py): a restart loses one
under way, though its chat, once its first turn is saved, survives.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from app import runs
from app.sessions import get_session_store, new_session_id
from app.task_notice import Status, format_notice
from brain.messages import (
    AgentMessage,
    AssistantMessage,
    TextContent,
    ToolResultMessage,
)
from brain.tools import ToolResult

logger = logging.getLogger(__name__)

MAX_RUNNING = 4

FORK_FRAME = (
    "You are a background task forked from the conversation above. Do only what's asked "
    "below, then reply with the result: it goes back to that conversation."
)
FRESH_CLOSE = "When you're done, reply with the result: it goes back to the chat that asked for it."
# What a fork's copy of the start_task call returned: neutral, so the fork
# doesn't begin by reading its own start as a failure.
FORK_ORIGIN = "Started this task, as a fork of this conversation."


@dataclass(slots=True)
class Task:
    id: str  # the task's own chat
    parent_id: str
    description: str
    project: str | None
    state: Literal["queued", "working", "done"] = "queued"
    status: Status | None = None
    result: str = ""


_tasks: dict[str, Task] = {}
# asyncio keeps only weak references to tasks; these keep the jobs alive.
_jobs: set[asyncio.Task[None]] = set()
_slots: tuple[asyncio.AbstractEventLoop, asyncio.Semaphore] | None = None


def queued() -> list[Task]:
    """Tasks waiting for a slot: they have no run yet to show."""
    return [t for t in _tasks.values() if t.state == "queued"]


class ChatTasks:
    """Starts tasks for one chat's turn (app/tools/tasks.py's TaskStarter)."""

    def __init__(self, chat_id: str, *, project: str | None, model: str | None, effort: str | None) -> None:
        self.chat_id = chat_id
        self.project = project
        self.model = model
        self.effort = effort
        # The conversation as the model sees it right now, for a fork; set
        # once the turn's harness exists (app/turns.py).
        self.context: Callable[[], Sequence[AgentMessage]] = tuple

    async def start(self, description: str, prompt: str, fork: bool) -> ToolResult:
        task = Task(new_session_id(), self.chat_id, description, self.project)
        _tasks[task.id] = task
        seed = _fork_seed(self.context()) if fork else []
        waiting = _slot().locked()
        job = asyncio.create_task(_run(task, prompt, seed, self.model, self.effort))
        _jobs.add(job)
        job.add_done_callback(_jobs.discard)
        where = (
            f"queued: {MAX_RUNNING} tasks are already working, so it starts when one finishes"
            if waiting
            else "started"
        )
        text = (
            f"Task {task.id} ({description}) {where}. Its result will arrive as a "
            "<task-notification>; carry on with the user until then."
        )
        return ToolResult(content=[TextContent(text=text)], details={"task_id": task.id, "description": description})

    async def result(self, task_id: str) -> ToolResult:
        task = _tasks.get(task_id)
        if task is not None and task.parent_id == self.chat_id:
            if task.state != "done":
                run = runs.get(task_id)
                how = "waiting on the user's answer" if run and run.state == "needs_you" else task.state
                return ToolResult.ok(f"Task {task_id} ({task.description}) isn't finished: it's {how}.")
            return ToolResult.ok(f"Status: {task.status}\n\n{task.result}")
        # One from before a restart: its chat is still there.
        session = await get_session_store().get(task_id)
        if session is None or session.parent_id != self.chat_id:
            return ToolResult.error(f"No task {task_id} was started from this chat.")
        return ToolResult.ok(_last_reply(session.messages) or "The task finished without a reply.")


def _slot() -> asyncio.Semaphore:
    """The running limit, made for the event loop in use (tests run several)."""
    global _slots
    loop = asyncio.get_running_loop()
    if _slots is None or _slots[0] is not loop:
        _slots = (loop, asyncio.Semaphore(MAX_RUNNING))
    return _slots[1]


def _fork_seed(context: Sequence[AgentMessage]) -> list[AgentMessage]:
    """The conversation for a fork, with every call still waiting on a
    result (the start_task call itself) given a neutral one."""
    seed = list(context)
    answered = {m.tool_call_id for m in seed if isinstance(m, ToolResultMessage)}
    for message in tuple(seed):
        if isinstance(message, AssistantMessage):
            for call in message.tool_calls:
                if call.id not in answered:
                    answered.add(call.id)
                    seed.append(
                        ToolResultMessage(
                            tool_call_id=call.id,
                            tool_name=call.name,
                            content=[TextContent(text=FORK_ORIGIN)],
                        )
                    )
    return seed


async def _run(task: Task, prompt: str, seed: list[AgentMessage], model: str | None, effort: str | None) -> None:
    from app.turns import TurnRefused, start_turn  # app.turns builds the starters

    async with _slot():
        task.state = "working"
        brief = f"{FORK_FRAME}\n\n<fork_task>\n{prompt}\n</fork_task>" if seed else f"{prompt}\n\n{FRESH_CLOSE}"
        try:
            run = await start_turn(
                task.id,
                brief,
                shown=prompt,
                model=model,
                effort=effort,
                project=task.project,
                parent_id=task.parent_id,
                seed=seed,
                title=task.description,
            )
            await run.wait_done()
            task.status, task.result = await _outcome(task.id, stopped=run.stopped)
        except TurnRefused as exc:
            task.status, task.result = "failed", str(exc)
        except Exception as exc:  # noqa: BLE001 - the chat that started it must still hear
            logger.exception("task %s failed", task.id)
            task.status, task.result = "failed", f"The task broke: {exc}"
    task.state = "done"
    await _deliver(task, model, effort)


async def _outcome(task_id: str, *, stopped: bool) -> tuple[Status, str]:
    session = await get_session_store().get(task_id)
    messages = session.messages if session is not None else []
    last = next((m for m in reversed(messages) if isinstance(m, AssistantMessage)), None)
    if stopped:
        return "stopped", "Stopped before it finished."
    if last is not None and last.stop_reason == "error":
        return "failed", last.error_message or "The model call failed."
    return "completed", _last_reply(messages) or "The task finished without a reply."


def _last_reply(messages: Sequence[AgentMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AssistantMessage) and message.text.strip():
            return message.text.strip()
    return ""


async def _deliver(task: Task, model: str | None, effort: str | None) -> None:
    """Tell the chat that started the task, as a follow-up: into its turn
    under way, or as a turn of its own."""
    from app.turns import TurnRefused, start_turn

    notice = format_notice(task.id, task.description, task.status or "failed", task.result)
    for _ in range(3):  # a turn may start or end between the check and the send
        parent = runs.get(task.parent_id)
        if parent is not None and parent.harness.is_running:
            parent.harness.follow_up(notice)
            return
        try:
            await start_turn(task.parent_id, notice, shown=notice, model=model, effort=effort)
            return
        except TurnRefused as exc:
            if exc.kind != "busy":
                logger.warning("task %s couldn't report back: %s", task.id, exc)
                return
    logger.warning("task %s couldn't report back: its chat stayed busy", task.id)


def clear_for_tests() -> None:
    global _slots
    _tasks.clear()
    _slots = None
