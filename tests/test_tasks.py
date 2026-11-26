"""Background tasks: Humphrey hands a job to a chat of its own, carries on,
and replies with the result when it comes back (app/tasks.py).

In mock mode "... in the background" makes Humphrey start a task with the
rest of the message, and a task's brief is handled like any other prompt:
"write a note" writes one, "ask me" holds the task open on a question.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

import pytest
from httpx import AsyncClient

from app import runs, tasks
from app.sessions import FORKED_NOTE, get_session_store
from app.task_notice import RESULT_CAP, format_notice, parse
from app.tools.tasks import IN_A_TASK
from brain.messages import ToolResultMessage

ANSWER = {"answers": {"Period": {"choices": ["Last 12 months"]}}}


def _events(body: str) -> list[dict[str, Any]]:
    return [
        json.loads(block.strip()[6:])
        for block in body.split("\n\n")
        if block.strip().startswith("data: ")
    ]


async def _settled() -> None:
    """Until every task has reported back and every turn is saved."""
    for _ in range(1000):
        if not tasks._jobs and not runs.current():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("tasks or turns never settled")


async def _asking(session_id: str) -> str:
    for _ in range(500):
        run = runs.get(session_id)
        if run is not None and run.bridge.waiting:
            [request_id] = list(run.bridge._pending)
            return request_id
        await asyncio.sleep(0.01)
    raise AssertionError("never asked")


def _send(client: AsyncClient, session_id: str, content: str):
    return client.post("/api/agent", json={"message": {"content": content}, "session_id": session_id})


def test_a_result_that_fits_goes_in_whole_and_one_that_does_not_is_never_cut() -> None:
    short = format_notice("t1", "Churn <by> segment", "completed", "Churn is 4%.")
    assert parse(short) == {
        "task_id": "t1",
        "description": "Churn ‹by› segment",
        "status": "completed",
        "result": "Churn is 4%.",
        "pending": False,
    }
    long = format_notice("t2", "Interviews", "completed", "x" * (RESULT_CAP + 1))
    assert "x" not in long  # no preview to act on
    assert parse(long)["pending"] is True
    assert "get_task_result with task id t2" in long
    assert parse("just a message") is None


@pytest.mark.asyncio
async def test_a_task_works_in_its_own_chat_and_humphrey_replies_with_its_result(client: AsyncClient) -> None:
    chat = uuid.uuid4().hex
    started = await _send(client, chat, "write a note in the background")
    [tool_end] = [e for e in _events(started.text) if e["type"] == "tool_end"]
    task_id = tool_end["task"]["task_id"]
    assert tool_end["tool"] == "start_task" and "started" in tool_end["result"]
    await _settled()

    task_chat = (await client.get(f"/api/sessions/{task_id}")).json()
    assert task_chat["title"] == "Background job"
    assert [a["name"] for a in task_chat["artifacts"]] == ["notes.md"]

    rows = (await client.get(f"/api/sessions/{chat}")).json()["messages"]
    [task_row] = [r for r in rows if r["role"] == "task"]
    assert task_row["task"]["task_id"] == task_id
    assert task_row["task"]["status"] == "completed"
    assert "write" in task_row["content"]  # the task's own reply, passed on
    assert rows[-1]["role"] == "assistant" and "reported back" in rows[-1]["content"]
    # The rail shows the task under the chat that started it.
    listed = {r["id"]: r for r in (await client.get("/api/sessions")).json()}
    assert listed[task_id]["parent_id"] == chat
    assert listed[chat]["parent_id"] is None


@pytest.mark.asyncio
async def test_a_fork_starts_from_the_conversation(client: AsyncClient) -> None:
    chat = uuid.uuid4().hex
    await _send(client, chat, "hello")
    started = await _send(client, chat, "fork this and write a note in the background")
    [tool_end] = [e for e in _events(started.text) if e["type"] == "tool_end"]
    task_id = tool_end["task"]["task_id"]
    await _settled()

    task = await get_session_store().get(task_id)
    assert task is not None and task.inherited > 0
    assert task.messages[0].text == "hello"  # the conversation it was forked from
    [origin] = [
        m for m in task.messages[: task.inherited]
        if isinstance(m, ToolResultMessage) and m.tool_name == "start_task"
    ]
    assert not origin.is_error  # its own start isn't read as a failure
    rows = (await client.get(f"/api/sessions/{task_id}")).json()["messages"]
    assert rows[0] == {**rows[0], "role": "context_summary", "content": FORKED_NOTE}
    assert "hello" not in [r["content"] for r in rows]  # one line, not a copy


@pytest.mark.asyncio
async def test_a_result_joins_the_turn_under_way(client: AsyncClient) -> None:
    """The task finishes while the chat is busy: its result is a follow-up
    to that turn, not a second turn alongside it."""
    chat = uuid.uuid4().hex
    started = await _send(client, chat, "ask me in the background")
    task_id = [e for e in _events(started.text) if e["type"] == "tool_end"][0]["task"]["task_id"]
    task_question = await _asking(task_id)
    # Opened mid-first-turn, before it's stored, it still knows what it is.
    opened = (await client.get(f"/api/sessions/{task_id}")).json()
    assert (opened["parent_id"], opened["title"]) == (chat, "Background job")
    assert opened["run"]["prompt"].startswith("ask me\n\nWhen you're done")

    busy = asyncio.create_task(_send(client, chat, "ask me"))
    chat_question = await _asking(chat)
    await client.post(f"/api/sessions/{task_id}/ui/{task_question}", json=ANSWER)
    for _ in range(500):
        if runs.get(chat).harness.queued_messages.follow_up:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("the result never reached the busy turn")
    await client.post(f"/api/sessions/{chat}/ui/{chat_question}", json=ANSWER)
    events = _events((await busy).text)
    assert [e for e in events if e["type"] == "task_notification"][0]["task_id"] == task_id
    await _settled()


@pytest.mark.asyncio
async def test_tasks_past_the_limit_wait_their_turn(client: AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tasks, "MAX_RUNNING", 1)
    chat = uuid.uuid4().hex
    first = await _send(client, chat, "ask me in the background")
    first_id = [e for e in _events(first.text) if e["type"] == "tool_end"][0]["task"]["task_id"]
    question = await _asking(first_id)

    second = await _send(client, chat, "write a note in the background")
    [end] = [e for e in _events(second.text) if e["type"] == "tool_end"]
    assert "queued" in end["result"]
    second_id = end["task"]["task_id"]
    activity = {a["session_id"]: a["state"] for a in (await client.get("/api/activity")).json()}
    assert activity[first_id] == "needs_you"
    assert activity[second_id] == "queued"

    await client.post(f"/api/sessions/{first_id}/ui/{question}", json=ANSWER)
    await _settled()
    done = (await client.get(f"/api/sessions/{second_id}")).json()
    assert [a["name"] for a in done["artifacts"]] == ["notes.md"]


@pytest.mark.asyncio
async def test_a_task_cannot_start_tasks() -> None:
    refused = await IN_A_TASK.start("More", "do more", False)
    assert refused.is_error and "can't start tasks" in refused.text


@pytest.mark.asyncio
async def test_a_chat_reads_only_its_own_tasks_results(client: AsyncClient) -> None:
    chat = uuid.uuid4().hex
    started = await _send(client, chat, "write a note in the background")
    task_id = [e for e in _events(started.text) if e["type"] == "tool_end"][0]["task"]["task_id"]
    await _settled()

    mine = await tasks.ChatTasks(chat, project=None, model=None, effort=None).result(task_id)
    assert mine.text.startswith("Status: completed")
    theirs = await tasks.ChatTasks("someone-else", project=None, model=None, effort=None).result(task_id)
    assert theirs.is_error

    tasks.clear_for_tests()  # as after a restart: the task's chat is still there
    again = await tasks.ChatTasks(chat, project=None, model=None, effort=None).result(task_id)
    assert not again.is_error and "write" in again.text


@pytest.mark.asyncio
async def test_a_follow_up_that_arrives_as_a_turn_ends_still_gets_a_reply(client: AsyncClient) -> None:
    """The turn's last check for follow-ups has passed but it isn't saved:
    one queued then starts the next turn rather than being lost."""
    store = get_session_store()
    real_save = store.save
    release = asyncio.Event()

    async def slow_save(session):  # type: ignore[no-untyped-def]
        await release.wait()
        await real_save(session)

    store.save = slow_save  # type: ignore[method-assign]
    chat = uuid.uuid4().hex
    first = asyncio.create_task(_send(client, chat, "hello"))
    for _ in range(500):
        run = runs.get(chat)
        if run is not None and not run.harness.is_running:
            break
        await asyncio.sleep(0.01)
    run.harness.follow_up("and one more thing")
    release.set()
    await first
    await _settled()
    stored = (await client.get(f"/api/sessions/{chat}")).json()["messages"]
    assert [m["content"] for m in stored if m["role"] == "user"] == ["hello", "and one more thing"]
