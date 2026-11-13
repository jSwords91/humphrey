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
        json={"message": {"content": "hello there"}},
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
async def test_a_long_session_keeps_working(client: AsyncClient) -> None:
    """Regression test: the client used to resend the whole transcript, the
    request capped that list at 40 messages, and so every session failed
    with a 422 from its 21st turn, for good. Only the new turn is sent now,
    and the history comes from the stored session.
    """
    session_id = None
    for i in range(25):
        response = await client.post(
            "/api/agent",
            json={"message": {"content": f"turn {i}"}, "session_id": session_id},
        )
        assert response.status_code == 200, f"turn {i}: {response.text[:200]}"
        session_id = response.headers["x-session-id"]

    from brain.messages import UserMessage
    from app.sessions import get_session_store

    session = await get_session_store().get(session_id)
    assert session is not None
    user_turns = [m.text for m in session.messages if isinstance(m, UserMessage)]
    assert user_turns == [f"turn {i}" for i in range(25)]


@pytest.mark.asyncio
async def test_long_past_replies_do_not_affect_later_requests(client: AsyncClient) -> None:
    """A long assistant reply lives in the stored session, never in a
    request, so it can't push a later request over any size limit."""
    from brain.messages import AssistantMessage, UserMessage
    from app.sessions import Session, get_session_store

    session_id = "longreplysession1"
    await get_session_store().save(
        Session(
            id=session_id,
            messages=[
                UserMessage(content="give me a detailed answer"),
                AssistantMessage(content="x" * 200_000, model="mock"),
            ],
        )
    )
    response = await client.post(
        "/api/agent",
        json={"message": {"content": "thanks, one more question"}, "session_id": session_id},
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_the_old_transcript_payload_is_rejected(client: AsyncClient) -> None:
    response = await client.post(
        "/api/agent", json={"messages": [{"role": "user", "content": "hello"}]}
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_tool_turn_bash(client: AsyncClient) -> None:
    response = await client.post(
        "/api/agent",
        json={"message": {"content": "list files please"}},
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
        json={"message": {"content": "please write a note"}},
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
        json={"message": {"content": "show me the architecture"}},
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
        json={"message": {"content": "/skill:mermaid-diagrams check it out"}},
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
        json={"message": {"content": "/skill:does-not-exist"}},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_session_persists_full_transcript(client: AsyncClient) -> None:
    response = await client.post(
        "/api/agent",
        json={"message": {"content": "list files please"}},
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
        json={"message": {"content": "my app isn't working, help me diagnose it"}},
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
        json={"message": {"content": "hello there"}},
    )
    session_id = first.headers["x-session-id"]
    session = await get_session_store().get(session_id)
    assert session is not None
    original_title = session.title
    assert original_title

    await client.post(
        "/api/agent",
        json={
            "message": {"content": "a completely different follow-up question"},
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
    """End-to-end: a session whose history exceeds the (test-forced, tiny)
    context window is compacted for the model, while the stored transcript
    and what the user sees keep every message.
    """
    import app.main
    from brain.messages import AssistantMessage, TextContent, UserMessage
    from app.sessions import Session, get_session_store
    from app.settings import get_settings, reset_settings_for_tests
    from providers.mock import MockProvider

    seen: list[list] = []  # the context of each agent-loop model call

    class RecordingProvider(MockProvider):
        def stream(self, messages, tools, *, system, signal=None):
            if tools:  # the summary and title calls have no tools
                seen.append(list(messages))
            return super().stream(messages, tools, system=system, signal=signal)

    async def recording_provider(model=None):
        return RecordingProvider()

    monkeypatch.setattr(app.main, "build_provider", recording_provider)

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
    await store.save(Session(id=session_id, messages=list(big_history)))

    response = await client.post(
        "/api/agent",
        json={
            "message": {"content": "one more question"},
            "session_id": session_id,
        },
    )
    assert response.status_code == 200

    session = await store.get(session_id)
    assert session is not None
    # The transcript is intact: the whole old history, then this turn.
    assert session.messages[: len(big_history)] == big_history
    assert session.messages[len(big_history)].text == "one more question"
    assert len(session.compactions) == 1

    # The model saw the summary and the kept tail, not the whole history.
    context = seen[0]
    assert context[0].text.startswith("Previous conversation summary:\n")
    assert len(context) < len(big_history)
    assert context[-1].text == "one more question"

    # The user still sees every turn, with the summary shown where the
    # compaction happened, and every row in time order.
    stored = await client.get(f"/api/sessions/{session_id}")
    assert stored.status_code == 200
    rows = stored.json()["messages"]
    user_rows = [m["content"] for m in rows if m["role"] == "user"]
    assert user_rows[: len(big_history) // 2] == [m.text.strip() for m in big_history[::2]]
    assert [m["role"] for m in rows].count("context_summary") == 1
    summary_at = next(i for i, m in enumerate(rows) if m["role"] == "context_summary")
    assert rows[summary_at + 1]["content"] == "one more question"
    assert not rows[summary_at]["content"].startswith("Previous conversation summary:")
    assert [m["ts"] for m in rows] == sorted(m["ts"] for m in rows)

    # A second turn builds on the stored record, and still loses nothing.
    response = await client.post(
        "/api/agent",
        json={"message": {"content": "and another"}, "session_id": session_id},
    )
    assert response.status_code == 200
    session = await store.get(session_id)
    assert session is not None
    assert session.messages[: len(big_history)] == big_history
    assert [m.text for m in session.messages if isinstance(m, UserMessage)][-2:] == [
        "one more question",
        "and another",
    ]
    assert seen[1][0].text.startswith("Previous conversation summary:\n")
    assert seen[1][-1].text == "and another"


@pytest.mark.asyncio
async def test_agent_endpoint_accepts_an_image_only_message(client: AsyncClient) -> None:
    response = await client.post(
        "/api/agent",
        json={
            "message": {"content": "", "images": [{"data": _png_base64()}]}
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
    [row] = [r for r in stored.json()["messages"] if r["role"] == "user"]
    assert row["content"] == ""
    [url] = row["images"]
    image = await client.get(url)
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/png"
    assert base64.b64decode(first.images[0].data) == image.content


@pytest.mark.asyncio
async def test_agent_endpoint_accepts_text_plus_image(client: AsyncClient) -> None:
    response = await client.post(
        "/api/agent",
        json={
            "message": {"content": "what is this?", "images": [{"data": _png_base64()}]}
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
            "message": {"content": "check this out", "images": [{"data": garbage}]}
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
    response = await client.post("/api/agent", json={"message": {"content": ""}})
    assert response.status_code == 422
