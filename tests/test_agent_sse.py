"""ASGI E2E: mock SSE contract for plain chat and tool turns."""

from __future__ import annotations

import base64
import json
from io import BytesIO

import pytest
from httpx import AsyncClient
from PIL import Image


def _png_base64(size: tuple[int, int] = (8, 6)) -> str:
    output = BytesIO()
    Image.new("RGB", size, "teal").save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


def _parse_sse(body: str) -> list[dict]:
    events: list[dict] = []
    for block in body.split("\n\n"):
        line = block.strip()
        if not line.startswith("data: "):
            continue
        events.append(json.loads(line[6:]))
    return events


@pytest.mark.asyncio
async def test_healthz_mock(client: AsyncClient) -> None:
    response = await client.get("/healthz")
    assert response.status_code == 200
    payload = response.json()
    assert payload["service"] == "humphrey"
    assert payload["mock_mode"] is True
    assert payload["deployment"] == "mock"


@pytest.mark.asyncio
async def test_frontend_assets_bust_and_revalidate(client: AsyncClient) -> None:
    from app.main import APP_VERSION

    html = await client.get("/")
    assert html.status_code == 200
    assert "no-store" in html.headers.get("cache-control", "").lower()
    assert f"app.js?v={APP_VERSION}" in html.text
    assert f"app.css?v={APP_VERSION}" in html.text

    css = await client.get("/app.css")
    assert css.status_code == 200
    assert "no-cache" in css.headers.get("cache-control", "").lower()

    mermaid = await client.get("/vendor/mermaid.min.js")
    assert mermaid.status_code == 200
    assert mermaid.headers.get("content-type", "").startswith("text/javascript") or "javascript" in mermaid.headers.get(
        "content-type", ""
    )
    assert "no-cache" in mermaid.headers.get("cache-control", "").lower()


@pytest.mark.asyncio
async def test_plain_chat_sse_sequence(client: AsyncClient) -> None:
    response = await client.post(
        "/api/agent",
        json={"messages": [{"role": "user", "content": "hello there"}]},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    session_id = response.headers.get("x-session-id")
    assert session_id

    events = _parse_sse(response.text)
    types = [e["type"] for e in events]
    assert types[0] == "agent_start"
    assert types[-1] == "agent_end"
    assert "message_start" in types
    assert "message_delta" in types
    assert "message_end" in types
    assert "tool_start" not in types
    assert any(e["type"] == "message_delta" and "mock" in (e.get("text") or "").lower() for e in events)


@pytest.mark.asyncio
async def test_resending_a_long_past_assistant_reply_does_not_422(client: AsyncClient) -> None:
    """Regression test: the frontend resends the whole transcript every turn
    (frontend/app.js), including past ASSISTANT replies - which can run much
    longer than a typed human message (e.g. a detailed bid-response skill
    answer). ChatMessage.content used to cap at 8000 chars, sized for human
    input, and a longer assistant reply then 422'd every subsequent turn in
    that session since the oversized message gets resent every time.
    """
    long_reply = "x" * 20_000
    response = await client.post(
        "/api/agent",
        json={
            "messages": [
                {"role": "user", "content": "give me a detailed answer"},
                {"role": "assistant", "content": long_reply},
                {"role": "user", "content": "thanks, one more question"},
            ]
        },
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_tool_turn_bash(client: AsyncClient) -> None:
    response = await client.post(
        "/api/agent",
        json={"messages": [{"role": "user", "content": "list files please"}]},
    )
    assert response.status_code == 200
    events = _parse_sse(response.text)
    types = [e["type"] for e in events]

    assert types[0] == "agent_start"
    assert types[-1] == "agent_end"
    assert "tool_start" in types
    assert "tool_end" in types

    starts = [e for e in events if e["type"] == "tool_start"]
    assert starts[0]["tool"] == "bash"
    assert starts[0]["label"] == "Listing workspace files"
    ends = [e for e in events if e["type"] == "tool_end"]
    assert ends[0]["call_id"] == starts[0]["call_id"]
    assert ends[0]["exit_code"] == 0

    # Final assistant prose after tools
    assert types.index("tool_end") < max(i for i, t in enumerate(types) if t == "message_delta")


@pytest.mark.asyncio
async def test_write_tool_writes_into_the_workspace(client: AsyncClient) -> None:
    from app.tools import WORKSPACE_DIR

    response = await client.post(
        "/api/agent",
        json={"messages": [{"role": "user", "content": "please write a note"}]},
    )
    assert response.status_code == 200
    events = _parse_sse(response.text)
    types = [e["type"] for e in events]

    assert "tool_start" in types
    starts = [e for e in events if e["type"] == "tool_start"]
    assert starts[0]["tool"] == "write"

    ends = [e for e in events if e["type"] == "tool_end"]
    assert ends[0]["is_error"] is False
    assert "please write a note" in (WORKSPACE_DIR / "notes.md").read_text()


@pytest.mark.asyncio
async def test_mermaid_skill_triggers_via_tool_call(client: AsyncClient) -> None:
    response = await client.post(
        "/api/agent",
        json={"messages": [{"role": "user", "content": "show me the architecture"}]},
    )
    assert response.status_code == 200
    events = _parse_sse(response.text)

    starts = [e for e in events if e["type"] == "tool_start"]
    assert starts
    assert starts[0]["tool"] == "read"
    assert starts[0]["args"]["path"].endswith("mermaid-diagrams/SKILL.md")

    ends = [e for e in events if e["type"] == "tool_end"]
    assert ends[0]["is_error"] is False
    assert "```mermaid" in (ends[0].get("result") or "")


@pytest.mark.asyncio
async def test_explicit_skill_command_expands_into_prompt(client: AsyncClient) -> None:
    response = await client.post(
        "/api/agent",
        json={"messages": [{"role": "user", "content": "/skill:mermaid-diagrams check it out"}]},
    )
    assert response.status_code == 200
    session_id = response.headers["x-session-id"]

    from app.sessions import get_session_store

    session = await get_session_store().get(session_id)
    assert session is not None
    from brain.messages import UserMessage

    first_user = next(m for m in session.messages if isinstance(m, UserMessage))
    assert '<skill name="mermaid-diagrams"' in first_user.content
    assert "check it out" in first_user.content


@pytest.mark.asyncio
async def test_explicit_skill_command_unknown_skill_returns_422(client: AsyncClient) -> None:
    response = await client.post(
        "/api/agent",
        json={"messages": [{"role": "user", "content": "/skill:does-not-exist"}]},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_session_persists_full_transcript(client: AsyncClient) -> None:
    response = await client.post(
        "/api/agent",
        json={"messages": [{"role": "user", "content": "list files please"}]},
    )
    session_id = response.headers["x-session-id"]
    events = _parse_sse(response.text)
    assert any(e["type"] == "tool_end" for e in events)

    stored = await client.get(f"/api/sessions/{session_id}")
    assert stored.status_code == 200
    body = stored.json()
    # UI projection is user/assistant only
    roles = [m["role"] for m in body["messages"]]
    assert roles[0] == "user"
    assert "assistant" in roles
    # Frontend seeds its context-budget display from this on reopen.
    assert body["last_usage"] is not None
    assert body["last_usage"]["input"] > 0

    from app.sessions import get_session_store

    session = await get_session_store().get(session_id)
    assert session is not None
    roles_full = [m.role for m in session.messages]
    assert "tool_result" in roles_full
    assert "assistant" in roles_full


@pytest.mark.asyncio
async def test_new_session_gets_auto_named(client: AsyncClient) -> None:
    response = await client.post(
        "/api/agent",
        json={"messages": [{"role": "user", "content": "my app isn't working, help me diagnose it"}]},
    )
    session_id = response.headers["x-session-id"]

    from app.sessions import get_session_store

    session = await get_session_store().get(session_id)
    assert session is not None
    assert session.title
    assert len(session.title.split()) <= 4


@pytest.mark.asyncio
async def test_session_title_is_not_overwritten_on_later_turns(client: AsyncClient) -> None:
    from app.sessions import get_session_store

    first = await client.post(
        "/api/agent",
        json={"messages": [{"role": "user", "content": "hello there"}]},
    )
    session_id = first.headers["x-session-id"]
    session = await get_session_store().get(session_id)
    assert session is not None
    original_title = session.title
    assert original_title

    await client.post(
        "/api/agent",
        json={
            "messages": [{"role": "user", "content": "a completely different follow-up question"}],
            "session_id": session_id,
        },
    )
    session = await get_session_store().get(session_id)
    assert session is not None
    assert session.title == original_title


@pytest.mark.asyncio
async def test_agent_endpoint_auto_compacts_a_large_history(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: a pre-existing session with enough history to exceed the
    (test-forced, tiny) context window must come back compacted after the
    next turn - a single summary message replacing the old prefix, not the
    original wall of history.
    """
    from brain.messages import AssistantMessage, TextContent, UserMessage
    from app.sessions import Session, get_session_store
    from app.settings import get_settings, reset_settings_for_tests

    monkeypatch.setenv("MODEL_CONTEXT_WINDOW_TOKENS", "2000")
    reset_settings_for_tests()
    assert get_settings().context_window_tokens == 2000

    # Must comfortably exceed DEFAULT_COMPACTION_KEEP_RECENT_TOKENS (20_000)
    # in total, or the "keep the recent tail" boundary search finds nowhere
    # safe to split and correctly declines to compact at all.
    filler = "word " * 2000  # ~2500 tokens per message at 4 chars/token
    big_history = []
    for i in range(20):
        big_history.append(UserMessage(content=f"turn {i} {filler}"))
        big_history.append(
            AssistantMessage(content=[TextContent(text=f"answer {i} {filler}")], model="mock")
        )

    session_id = "compactionregression1"
    store = get_session_store()
    await store.save(Session(id=session_id, messages=big_history))

    response = await client.post(
        "/api/agent",
        json={
            "messages": [{"role": "user", "content": "one more question"}],
            "session_id": session_id,
        },
    )
    assert response.status_code == 200

    session = await store.get(session_id)
    assert session is not None
    assert len(session.messages) < len(big_history)
    first = session.messages[0]
    assert isinstance(first, UserMessage)
    assert first.content.startswith("Previous conversation summary:\n")

    # The UI-facing projection must not disguise the summary as a real user
    # turn (see frontend/app.js's appendContextSummary) - it's a distinct
    # role, and the raw prefix is stripped since the frontend supplies its
    # own label.
    stored = await client.get(f"/api/sessions/{session_id}")
    assert stored.status_code == 200
    ui_messages = stored.json()["messages"]
    assert ui_messages[0]["role"] == "context_summary"
    assert not ui_messages[0]["content"].startswith("Previous conversation summary:")
    assert "user" in [m["role"] for m in ui_messages]
    assert "assistant" in [m["role"] for m in ui_messages]


@pytest.mark.asyncio
async def test_agent_endpoint_accepts_an_image_only_message(client: AsyncClient) -> None:
    response = await client.post(
        "/api/agent",
        json={
            "messages": [
                {"role": "user", "content": "", "images": [{"data": _png_base64()}]}
            ]
        },
    )
    assert response.status_code == 200
    session_id = response.headers["x-session-id"]

    from brain.messages import UserMessage
    from app.sessions import get_session_store

    session = await get_session_store().get(session_id)
    assert session is not None
    first = session.messages[0]
    assert isinstance(first, UserMessage)
    assert len(first.images) == 1
    assert first.images[0].mime_type == "image/png"

    stored = await client.get(f"/api/sessions/{session_id}")
    ui_rows = stored.json()["messages"]
    assert "1 image" in ui_rows[0]["content"]


@pytest.mark.asyncio
async def test_agent_endpoint_accepts_text_plus_image(client: AsyncClient) -> None:
    response = await client.post(
        "/api/agent",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "what is this?",
                    "images": [{"data": _png_base64()}],
                }
            ]
        },
    )
    assert response.status_code == 200
    session_id = response.headers["x-session-id"]

    from brain.messages import UserMessage
    from app.sessions import get_session_store

    session = await get_session_store().get(session_id)
    assert session is not None
    first = session.messages[0]
    assert isinstance(first, UserMessage)
    assert first.text.startswith("what is this?")
    assert len(first.images) == 1


@pytest.mark.asyncio
async def test_agent_endpoint_degrades_an_unsupported_image_to_a_visible_note(
    client: AsyncClient,
) -> None:
    """A bad attachment must not 422 the whole turn - it becomes a plain-text
    note the model (and, transitively, the user reading history) can see,
    per this app's "don't drop information" convention.
    """
    garbage = base64.b64encode(b"not an image").decode("ascii")
    response = await client.post(
        "/api/agent",
        json={
            "messages": [
                {"role": "user", "content": "check this out", "images": [{"data": garbage}]}
            ]
        },
    )
    assert response.status_code == 200
    session_id = response.headers["x-session-id"]

    from brain.messages import UserMessage
    from app.sessions import get_session_store

    session = await get_session_store().get(session_id)
    assert session is not None
    first = session.messages[0]
    assert isinstance(first, UserMessage)
    assert first.images == []
    assert "unsupported or unrecognized format" in first.text


@pytest.mark.asyncio
async def test_chat_message_requires_content_or_at_least_one_image(client: AsyncClient) -> None:
    response = await client.post("/api/agent", json={"messages": [{"role": "user", "content": ""}]})
    assert response.status_code == 422
