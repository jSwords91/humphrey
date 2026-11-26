"""start_task and get_task_result: hand a job to a background task, and
read what it found.

Modelled on tau-subagents' ``agent`` tool, background only. A task is a
chat of its own, started from this one: fresh, with just the brief, or a
fork of this conversation. This turn carries on at once; when the task
finishes, its result comes back to this chat as a ``<task-notification>``
(app/task_notice.py), and Humphrey replies with it.

The tools call a ``TaskStarter``, which app/tasks.py provides for a chat
that can start tasks. Every chat gets the same two tools, so a forked
task's tool list matches its parent's and the model provider can reuse the
parent's cached prompt; where tasks can't start (a task's own chat, evals),
the starter refuses with a reason the model can act on.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from brain.tools import AgentTool, ToolCancellationToken, ToolResult, ToolUpdateCallback

START = "start_task"
RESULT = "get_task_result"


class TaskStarter(Protocol):
    async def start(self, description: str, prompt: str, fork: bool) -> ToolResult: ...

    async def result(self, task_id: str) -> ToolResult: ...


class NoTasks:
    """Where no task can start: there's no chat to report back to."""

    def __init__(self, reason: str = "Background tasks aren't available here. Do the work in this turn.") -> None:
        self._reason = reason

    async def start(self, description: str, prompt: str, fork: bool) -> ToolResult:
        return ToolResult.error(self._reason)

    async def result(self, task_id: str) -> ToolResult:
        return ToolResult.error(self._reason)


# A task's own chat: one level of tasks, as tau's ``isolated`` gives.
IN_A_TASK = NoTasks("This is a background task, and a task can't start tasks of its own. Do the work here.")


START_PARAMETERS: dict[str, Any] = {
    "description": {
        "type": "string",
        "description": "A short title for the task, 3 to 6 words, e.g. \"Churn by segment\". The user sees it.",
    },
    "prompt": {
        "type": "string",
        "description": (
            "What the task should do and what to report back. Without fork, the task knows "
            "nothing else, so include every detail it needs."
        ),
    },
    "fork": {
        "type": "boolean",
        "description": (
            "True to start the task from this conversation, so it knows everything said so far. "
            "Leave it false for a job the prompt fully describes."
        ),
    },
}

GUIDELINES: tuple[str, ...] = (
    "Use start_task when the user asks for something to happen in the background, or to be done "
    "separately while you carry on, or for a long job they don't need to watch. Don't use it for "
    "quick things you can do now.",
    "After start_task, tell the user in a sentence what's running, then carry on with them. Its "
    "result arrives later as a <task-notification>; reply to it by passing the result on and "
    "saying what it means for what you're doing together.",
)


def create_task_tools(starter: TaskStarter) -> list[AgentTool]:
    """The two tools for one run, bound to what can start tasks there."""

    async def start(
        _tool_call_id: str,
        arguments: Mapping[str, Any],
        _signal: ToolCancellationToken | None = None,
        _on_update: ToolUpdateCallback | None = None,
    ) -> ToolResult:
        description = str(arguments.get("description") or "").strip()
        prompt = str(arguments.get("prompt") or "").strip()
        if not description or not prompt:
            return ToolResult.error("start_task needs both a description and a prompt.")
        return await starter.start(description, prompt, bool(arguments.get("fork")))

    async def result(
        _tool_call_id: str,
        arguments: Mapping[str, Any],
        _signal: ToolCancellationToken | None = None,
        _on_update: ToolUpdateCallback | None = None,
    ) -> ToolResult:
        task_id = str(arguments.get("task_id") or "").strip()
        if not task_id:
            return ToolResult.error("get_task_result needs a task_id.")
        return await starter.result(task_id)

    return [
        AgentTool(
            name=START,
            label="Task: {description}",
            description=(
                "Start a background task: a separate chat that works on its own while this "
                "conversation carries on. Returns at once; the result arrives later as a "
                "<task-notification>."
            ),
            parameters=START_PARAMETERS,
            execute_fn=start,
            required=("description", "prompt"),
            prompt_snippet="Hand a job to a background task and carry on; its result comes back later.",
            prompt_guidelines=GUIDELINES,
        ),
        AgentTool(
            name=RESULT,
            label="Reading a task's result",
            description=(
                "The full result of a background task this chat started, or how it's getting on. "
                "Use it when a <task-notification> says the result was too long to include."
            ),
            parameters={"task_id": {"type": "string", "description": "The task id from start_task."}},
            execute_fn=result,
            required=("task_id",),
            prompt_snippet="Read the full result of a background task this chat started.",
        ),
    ]
