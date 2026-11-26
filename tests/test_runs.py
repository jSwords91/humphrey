"""A run belongs to the server, not to the browser that started it.

Leaving the page (a reload, another chat, a closed tab) used to stop the
turn. Now it carries on and is saved; a browser that comes back follows it
from the first frame, and only the stop route ends it early. "ask me"
makes the mock ask a question, which holds the run open until it's answered.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

import pytest
from httpx import AsyncClient

from app import runs
from app.runs import FrameLog

ANSWER = {"answers": {"Period": {"choices": ["Last 12 months"]}}}


def _events(body: str) -> list[dict[str, Any]]:
    return [
        json.loads(block.strip()[6:])
        for block in body.split("\n\n")
        if block.strip().startswith("data: ")
    ]


async def _asking(session_id: str) -> str:
    """Wait until the session's run is asking, and return the request id."""
    for _ in range(500):
        run = runs.get(session_id)
        if run is not None and run.bridge.waiting:
            [request_id] = list(run.bridge._pending)
            return request_id
        await asyncio.sleep(0.01)
    raise AssertionError("the run never asked")


async def _ended(session_id: str) -> None:
    for _ in range(500):
        if runs.get(session_id) is None:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("the run never ended")


def _start(client: AsyncClient, session_id: str, content: str = "ask me") -> asyncio.Task:
    return asyncio.create_task(
        client.post("/api/agent", json={"message": {"content": content}, "session_id": session_id})
    )


@pytest.mark.asyncio
async def test_leaving_mid_turn_neither_stops_nor_loses_it(client: AsyncClient) -> None:
    session_id = uuid.uuid4().hex
    request = _start(client, session_id)
    request_id = await _asking(session_id)
    request.cancel()  # the browser went away, as Starlette sees a disconnect
    await asyncio.sleep(0.05)

    answered = await client.post(f"/api/sessions/{session_id}/ui/{request_id}", json=ANSWER)
    assert answered.status_code == 204
    await _ended(session_id)

    stored = (await client.get(f"/api/sessions/{session_id}")).json()
    assert [m["role"] for m in stored["messages"]][0] == "user"
    [question] = [m for m in stored["messages"] if m["role"] == "question"]
    assert question["question"]["status"] == "answered"
    assert stored["title"]  # named, though nobody was watching
    assert stored["run"] is None


@pytest.mark.asyncio
async def test_coming_back_mid_turn_replays_it_from_the_start(client: AsyncClient) -> None:
    session_id = uuid.uuid4().hex
    first = await client.post(
        "/api/agent", json={"message": {"content": "hello"}, "session_id": session_id}
    )
    assert first.status_code == 200

    original = _start(client, session_id)
    await _asking(session_id)

    # Reopened mid-turn: the stored history stops before this turn, and
    # the run is there to follow, with what was typed to start it.
    reopened = (await client.get(f"/api/sessions/{session_id}")).json()
    assert [m["content"] for m in reopened["messages"] if m["role"] == "user"] == ["hello"]
    assert reopened["run"] == {"prompt": "ask me", "task": None}

    async def answer() -> None:
        await asyncio.sleep(0.05)  # once the follower is in
        request_id = await _asking(session_id)
        await client.post(f"/api/sessions/{session_id}/ui/{request_id}", json=ANSWER)

    followed, _ = await asyncio.gather(client.get(f"/api/sessions/{session_id}/stream"), answer())
    assert followed.status_code == 200
    assert _events(followed.text) == _events((await original).text)
    assert _events(followed.text)[-1]["type"] == "agent_end"

    await _ended(session_id)
    gone = await client.get(f"/api/sessions/{session_id}/stream")
    assert gone.status_code == 404


@pytest.mark.asyncio
async def test_a_new_chat_can_be_found_and_reopened_during_its_first_turn(client: AsyncClient) -> None:
    """Nothing is stored until the first turn is, and a reload then is the
    likeliest moment of all."""
    session_id = uuid.uuid4().hex
    request = _start(client, session_id)
    await _asking(session_id)

    [listed] = [r for r in (await client.get("/api/sessions")).json() if r["id"] == session_id]
    assert listed["title"] == "ask me"  # named by its prompt until it's named properly

    reopened = await client.get(f"/api/sessions/{session_id}")
    assert reopened.status_code == 200
    assert reopened.json()["messages"] == []
    assert reopened.json()["run"] == {"prompt": "ask me", "task": None}

    await client.post(f"/api/sessions/{session_id}/stop", json={})
    await request


@pytest.mark.asyncio
async def test_stop_ends_the_run(client: AsyncClient) -> None:
    session_id = uuid.uuid4().hex
    request = _start(client, session_id)
    await _asking(session_id)

    stopped = await client.post(f"/api/sessions/{session_id}/stop", json={})
    assert stopped.status_code == 204
    events = _events((await request).text)
    [end] = [e for e in events if e["type"] == "tool_end"]
    assert end["question"]["status"] == "stopped"
    # Every follower is told it was stopped, which is not an error.
    assert {"type": "stopped", "session_id": session_id} in events
    assert not [e for e in events if e["type"] == "error"]

    await _ended(session_id)
    again = await client.post(f"/api/sessions/{session_id}/stop", json={})
    assert again.status_code == 409


@pytest.mark.asyncio
async def test_stop_takes_json_only(client: AsyncClient) -> None:
    """A page on another site can't stop a run with a plain form POST."""
    response = await client.post(
        f"/api/sessions/{uuid.uuid4().hex}/stop", content="", headers={"content-type": "text/plain"}
    )
    assert response.status_code == 415


@pytest.mark.asyncio
async def test_a_chat_runs_one_turn_at_a_time(client: AsyncClient) -> None:
    session_id = uuid.uuid4().hex
    request = _start(client, session_id)
    await _asking(session_id)

    second = await client.post(
        "/api/agent", json={"message": {"content": "hello"}, "session_id": session_id}
    )
    assert second.status_code == 409

    await client.post(f"/api/sessions/{session_id}/stop", json={})
    await request


@pytest.mark.asyncio
async def test_the_next_turn_waits_for_the_last_one_to_be_saved(client: AsyncClient) -> None:
    """A turn whose model work is done but isn't saved yet: the next turn
    waits rather than starting from the history before it."""
    from app.sessions import get_session_store

    session_id = uuid.uuid4().hex
    store = get_session_store()
    real_save = store.save
    release = asyncio.Event()

    async def slow_save(session):  # type: ignore[no-untyped-def]
        await release.wait()
        await real_save(session)

    store.save = slow_save  # type: ignore[method-assign]
    first = _start(client, session_id, "hello")
    for _ in range(500):
        run = runs.get(session_id)
        if run is not None and not run.harness.is_running:
            break
        await asyncio.sleep(0.01)

    second = _start(client, session_id, "and again")
    await asyncio.sleep(0.05)
    assert not second.done()  # waiting for the save
    release.set()
    assert (await first).status_code == 200
    assert (await second).status_code == 200

    await _ended(session_id)
    stored = (await client.get(f"/api/sessions/{session_id}")).json()
    assert [m["content"] for m in stored["messages"] if m["role"] == "user"] == [
        "hello",
        "and again",
    ]


@pytest.mark.asyncio
async def test_reopening_while_the_turn_saves_counts_its_usage_once(client: AsyncClient) -> None:
    """The stored session may already hold the turn while it's being saved,
    and the replay adds the turn's usage, so the stored totals leave it out."""
    from app.sessions import get_session_store

    session_id = uuid.uuid4().hex
    await client.post("/api/agent", json={"message": {"content": "hello"}, "session_id": session_id})
    await _ended(session_id)
    before = (await client.get(f"/api/sessions/{session_id}")).json()
    assert before["usage_totals"]["output"] > 0

    store = get_session_store()
    real_save = store.save
    release = asyncio.Event()

    async def slow_save(session):  # type: ignore[no-untyped-def]
        await release.wait()
        await real_save(session)

    store.save = slow_save  # type: ignore[method-assign]
    second = _start(client, session_id, "hello again")
    for _ in range(500):
        run = runs.get(session_id)
        if run is not None and not run.harness.is_running:
            break  # the turn is done and being saved: the memory store already holds it
        await asyncio.sleep(0.01)

    during = (await client.get(f"/api/sessions/{session_id}")).json()
    assert during["run"] is not None
    assert during["usage_totals"] == before["usage_totals"]
    assert during["last_usage"] == before["last_usage"]
    release.set()
    await second


@pytest.mark.asyncio
async def test_a_file_another_chat_writes_is_not_this_chats_artefact(client: AsyncClient) -> None:
    """Found once runs outlived the page: two chats can work at once, and a
    chat's artefacts came from comparing the whole workspace after each of
    its tools, so a file the other chat wrote while this one waited on a
    question became this chat's too."""
    waiting = uuid.uuid4().hex
    request = _start(client, waiting)
    request_id = await _asking(waiting)

    writer = uuid.uuid4().hex
    await client.post("/api/agent", json={"message": {"content": "write a note"}, "session_id": writer})
    await _ended(writer)

    await client.post(f"/api/sessions/{waiting}/ui/{request_id}", json=ANSWER)
    await request
    await _ended(waiting)
    wrote = (await client.get(f"/api/sessions/{writer}")).json()["artifacts"]
    asked = (await client.get(f"/api/sessions/{waiting}")).json()["artifacts"]
    assert [a["path"] for a in wrote] == ["notes.md"]
    assert asked == []


@pytest.mark.asyncio
async def test_activity_says_which_chats_are_working_or_waiting(client: AsyncClient) -> None:
    session_id = uuid.uuid4().hex
    assert session_id not in [a["session_id"] for a in (await client.get("/api/activity")).json()]

    request = _start(client, session_id)
    await _asking(session_id)
    [waiting] = [a for a in (await client.get("/api/activity")).json() if a["session_id"] == session_id]
    assert waiting == {"session_id": session_id, "project": None, "state": "needs_you"}

    await client.post(f"/api/sessions/{session_id}/stop", json={})
    await request
    await _ended(session_id)
    assert session_id not in [a["session_id"] for a in (await client.get("/api/activity")).json()]


def test_a_run_is_working_until_its_model_is_done() -> None:
    from app.agent import build_harness
    from app.runs import Run
    from app.ui_bridge import WebUiBridge
    from brain.messages import UserMessage

    harness = build_harness()
    bridge = WebUiBridge(lambda payload: None)
    run = Run(
        "s",
        harness=harness,
        bridge=bridge,
        log=FrameLog(),
        prompt="hi",
        started_ms=0,
        new_chat=True,
        project=None,
    )
    assert run.state is None
    harness.prompt_message(UserMessage(content="hi"))  # started, as /api/agent does
    assert run.state == "working"


@pytest.mark.asyncio
async def test_a_late_follower_gets_every_frame_then_the_live_ones() -> None:
    log = FrameLog()
    log.publish("one")
    seen: list[str | None] = []

    async def follow() -> None:
        async for frame in log.follow(idle=0.05):
            seen.append(frame)

    task = asyncio.create_task(follow())
    await asyncio.sleep(0.08)  # quiet long enough for one keepalive
    log.publish("two")
    log.finish()
    log.publish("after the end")  # ignored
    await task
    assert seen[0] == "one"
    assert None in seen
    assert seen[-1] == "two"
