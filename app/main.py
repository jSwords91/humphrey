"""Humphrey API + static frontend."""

from __future__ import annotations

import base64
import binascii
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator
from starlette.responses import Response
from starlette.types import Scope

from app.agent import build_harness, system_prompt
from app.artifacts import (
    ArtifactTracker,
    artifact_kind,
    inject_embed_script,
    resolve_workspace_file,
    scan_workspace,
)
from app.context_window import maybe_compact_history
from app.image_processing import (
    ImageProcessingFailure,
    ProcessedImage,
    detect_supported_image_mime_type,
    process_image,
)
from app.rendering.sse import SseRenderer, encode_sse
from app.runs import get_active, register_active, unregister_active
from app.skills import SkillError, expand_skill_command, load_skills
from app.sessions import (
    SAFE_SESSION_ID,
    Session,
    generate_session_title,
    get_session_store,
    new_session_id,
)
from app.settings import get_settings
from app.tools import TOOLS, WORKSPACE_DIR
from brain.events import ToolExecutionEndEvent
from brain.cancel import SimpleCancellationToken
from brain.messages import AssistantMessage, ImageContent, TextContent, UserMessage
from providers import active_provider_kind, get_provider

logger = logging.getLogger(__name__)
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

APP_VERSION = "0.1.0"
app = FastAPI(title="Humphrey", version=APP_VERSION)


class RevalidatingStaticFiles(StaticFiles):
    """JS/CSS are version-queried; still revalidate so a sticky cache cannot keep a stale release."""

    async def get_response(self, path: str, scope: Scope) -> Response:
        response = await super().get_response(path, scope)
        if path.lower().endswith((".js", ".css", ".svg")):
            response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return response


class ImageAttachment(BaseModel):
    """Raw client-uploaded image, before app.image_processing validates it.

    No client-declared mime type - the actual format is always sniffed
    server-side from the bytes (app.image_processing.detect_supported_image_mime_type),
    so a mislabeled or absent client hint can't matter. ``data`` is base64.
    75_000_000 chars gives headroom above the ~69.9M chars a 50MB source
    image (image_processing.DEFAULT_MAX_SOURCE_IMAGE_BYTES) encodes to, so an
    oversized image is rejected with that module's clear message rather than
    a generic length-validation error here.
    """

    data: str = Field(..., min_length=1, max_length=75_000_000)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    # Higher than QueueRequest's 8000 below: the frontend resends the whole
    # transcript on every turn (frontend/app.js), so this bounds not just a
    # typed human message but also past ASSISTANT replies - which can run
    # much longer, e.g. a detailed multi-case-study bid answer. Found live:
    # a bid-responses skill reply exceeded 8000 chars and then failed
    # validation on every subsequent turn in that session, since the
    # oversized message gets resent every time.
    content: str = Field(default="", max_length=40_000)
    images: list[ImageAttachment] = Field(default_factory=list, max_length=4)

    @model_validator(mode="after")
    def _require_content_or_images(self) -> ChatMessage:
        if not self.content.strip() and not self.images:
            raise ValueError("content or at least one image is required")
        return self


class AgentRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1, max_length=40)
    session_id: str | None = Field(default=None, max_length=64)
    # Per-request OpenAI model; must be one of /api/models. Omit for the default.
    model: str | None = Field(default=None, max_length=100)


class QueueRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=8000)


class SessionMessageOut(BaseModel):
    # context_summary: a synthetic auto-compaction summary (app/context_window.py),
    # not something the user typed - rendered distinctly by the frontend.
    role: Literal["user", "assistant", "context_summary"]
    content: str
    ts: int | None = None


class SessionOut(BaseModel):
    id: str
    messages: list[SessionMessageOut]
    created_at: str
    updated_at: str
    title: str | None = None
    last_usage: dict[str, int] | None = None
    artifacts: list[dict[str, Any]] = Field(default_factory=list)


class SessionSummaryOut(BaseModel):
    id: str
    updated_at: str
    title: str | None
    message_count: int


class ModelsOut(BaseModel):
    provider: str
    selectable: bool
    default: str
    models: list[str]


class QueueOut(BaseModel):
    steering: int
    follow_up: int
    count: int


@app.get("/")
def index() -> HTMLResponse:
    html = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    html = html.replace("__APP_VERSION__", APP_VERSION)
    return HTMLResponse(
        content=html,
        headers={
            "Cache-Control": "no-store, max-age=0, must-revalidate",
            "Pragma": "no-cache",
        },
    )


@app.get("/healthz")
def healthz() -> dict[str, object]:
    settings = get_settings()
    provider = active_provider_kind(settings)
    deployment = {
        "mock": "mock",
        "openai": settings.openai_model,
        "azure_openai": settings.azure_openai_deployment,
        "unconfigured": "unset",
    }[provider]
    return {
        "status": "ok",
        "service": "humphrey",
        "version": APP_VERSION,
        "mock_mode": settings.mock_mode,
        "storage": settings.storage_backend,
        "provider": provider,
        "deployment": deployment,
    }


@app.get("/api/models", response_model=ModelsOut)
def list_models() -> ModelsOut:
    """Models the UI may offer. Only OpenAI supports switching; the mock and
    the Azure adapter report their single fixed model."""
    settings = get_settings()
    provider = active_provider_kind(settings)
    if provider == "openai":
        models = list(settings.selectable_models)
        return ModelsOut(provider=provider, selectable=True, default=models[0], models=models)
    fixed = {
        "mock": "mock",
        "azure_openai": settings.azure_openai_deployment,
        "unconfigured": "unset",
    }[provider]
    return ModelsOut(provider=provider, selectable=False, default=fixed, models=[fixed])


@app.get("/api/sessions", response_model=list[SessionSummaryOut])
async def list_sessions() -> list[SessionSummaryOut]:
    rows = await get_session_store().list_recent(limit=20)
    return [
        SessionSummaryOut(
            id=r.id,
            updated_at=r.updated_at,
            title=r.title,
            message_count=r.message_count,
        )
        for r in rows
    ]


@app.get("/api/sessions/{session_id}", response_model=SessionOut)
async def get_session(session_id: str) -> SessionOut:
    if not SAFE_SESSION_ID.match(session_id):
        raise HTTPException(status_code=422, detail="Invalid session id.")
    session = await get_session_store().get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    # Usage accumulates per turn (each call resends the whole transcript), so
    # the last assistant message's usage is the closest available proxy for
    # "how much context this conversation currently occupies" — used by the
    # frontend to seed its context-budget display when a past chat is reopened.
    last_usage: dict[str, int] | None = None
    for message in reversed(session.messages):
        if isinstance(message, AssistantMessage) and message.usage:
            last_usage = dict(message.usage)
            break

    return SessionOut(
        id=session.id,
        messages=[
            SessionMessageOut(role=m.role, content=m.content, ts=m.ts)  # type: ignore[arg-type]
            for m in session.ui_chat_messages()
            if m.role in {"user", "assistant", "context_summary"}
        ],
        created_at=session.created_at,
        updated_at=session.updated_at,
        title=session.title,
        last_usage=last_usage,
        artifacts=session.artifacts,
    )


def _require_active(session_id: str):
    if not SAFE_SESSION_ID.match(session_id):
        raise HTTPException(status_code=422, detail="Invalid session id.")
    harness = get_active(session_id)
    if harness is None or not harness.is_running:
        raise HTTPException(status_code=409, detail="No active run for this session.")
    return harness


@app.post("/api/sessions/{session_id}/steer", response_model=QueueOut)
async def steer_session(session_id: str, body: QueueRequest) -> QueueOut:
    """Inject a mid-run steering message (drained between tool turns)."""
    harness = _require_active(session_id)
    queued = harness.steer(body.content.strip())
    return QueueOut(
        steering=len(queued.steering),
        follow_up=len(queued.follow_up),
        count=queued.count,
    )


@app.post("/api/sessions/{session_id}/follow-up", response_model=QueueOut)
async def follow_up_session(session_id: str, body: QueueRequest) -> QueueOut:
    """Queue a follow-up for after the current run would otherwise settle."""
    harness = _require_active(session_id)
    queued = harness.follow_up(body.content.strip())
    return QueueOut(
        steering=len(queued.steering),
        follow_up=len(queued.follow_up),
        count=queued.count,
    )


def _process_uploaded_images(attachments: list[ImageAttachment]) -> tuple[list[ImageContent], list[str]]:
    """Decode, sniff, and bound each attachment (app/image_processing.py).

    The client-declared mime_type is never trusted - only the sniffed
    magic-byte format decides how bytes are decoded/re-encoded, matching
    tau's read-tool boundary. A bad attachment becomes a plain-text note
    (visible to the model, not a silent drop - see this app's "don't hide
    information" convention) rather than a 422, so one bad image in a
    multi-image message doesn't block the others.
    """
    contents: list[ImageContent] = []
    notes: list[str] = []
    for attachment in attachments:
        try:
            raw = base64.b64decode(attachment.data, validate=True)
        except (binascii.Error, ValueError):
            notes.append("One image attachment could not be decoded and was skipped.")
            continue
        mime_type = detect_supported_image_mime_type(raw)
        if mime_type is None:
            notes.append(
                "One image attachment has an unsupported or unrecognized format and was skipped."
            )
            continue
        result: ProcessedImage | ImageProcessingFailure = process_image(raw, mime_type)
        if isinstance(result, ImageProcessingFailure):
            notes.append(f"One image attachment was skipped: {result.message}.")
            continue
        contents.append(
            ImageContent(
                data=base64.b64encode(result.data).decode("ascii"),
                mime_type=result.mime_type,
            )
        )
        notes.extend(result.notes)
    return contents, notes


@app.post("/api/agent")
async def agent(body: AgentRequest, request: Request) -> StreamingResponse:
    if body.messages[-1].role != "user":
        raise HTTPException(status_code=422, detail="Last message must be from the user.")

    store = get_session_store()
    session_id = body.session_id
    if session_id is not None and not SAFE_SESSION_ID.match(session_id):
        raise HTTPException(status_code=422, detail="Invalid session id.")

    if session_id:
        existing = await store.get(session_id)
        session = existing or Session(id=session_id)
    else:
        session_id = new_session_id()
        session = Session(id=session_id)

    latest_user = body.messages[-1].content
    history = list(session.messages)
    if history and isinstance(history[-1], UserMessage) and history[-1].content == latest_user:
        history = history[:-1]

    settings = get_settings()
    model = None
    if body.model and active_provider_kind(settings) == "openai":
        if body.model not in settings.selectable_models:
            raise HTTPException(status_code=422, detail=f"Unknown model: {body.model}")
        model = body.model
    provider = get_provider(model)
    history = await maybe_compact_history(
        history,
        provider=provider,
        system=system_prompt(),
        tools=TOOLS,
        context_window_tokens=settings.context_window_tokens,
    )

    try:
        expanded = expand_skill_command(latest_user, load_skills())
    except SkillError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    prompt_text = expanded if expanded is not None else latest_user

    image_contents, image_notes = _process_uploaded_images(body.messages[-1].images)
    if image_notes:
        notes_text = "\n".join(image_notes)
        prompt_text = f"{prompt_text}\n\n{notes_text}" if prompt_text else notes_text
    if image_contents:
        text_blocks = [TextContent(text=prompt_text)] if prompt_text else []
        prompt_message = UserMessage(content=[*text_blocks, *image_contents])
    else:
        prompt_message = UserMessage(content=prompt_text)

    harness = build_harness(session_id=session_id, provider=provider, messages=history)
    cancel = SimpleCancellationToken()
    register_active(session_id, harness)

    tracker = ArtifactTracker(WORKSPACE_DIR)

    async def event_stream() -> AsyncIterator[str]:
        renderer = SseRenderer(session_id=session_id)
        stream = harness.prompt_message(prompt_message)
        try:
            async for event in stream:
                if await request.is_disconnected():
                    cancel.cancel()
                    harness.cancel()
                    break
                wire = renderer.to_sse(event)
                if wire is not None:
                    yield wire
                if isinstance(event, ToolExecutionEndEvent):
                    for artifact in tracker.changes():
                        artifact["version"] = 1 + sum(
                            1 for a in session.artifacts if a.get("path") == artifact["path"]
                        )
                        session.artifacts.append(artifact)
                        yield encode_sse({"type": "artifact", **artifact})
        finally:
            try:
                await stream.aclose()
            except Exception:  # noqa: BLE001
                logger.debug("agent stream aclose failed", exc_info=True)
            if cancel.is_cancelled():
                harness.cancel()
                harness.append_interrupted_tool_results()
            unregister_active(session_id, harness)
            session.messages = list(harness.messages)
            if session.title is None:
                session.title = await generate_session_title(harness.config.provider, latest_user)
            try:
                await store.save(session)
            except Exception:  # noqa: BLE001
                logger.exception("failed to persist session %s", session_id)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Session-Id": session_id,
        },
    )


@app.get("/api/files")
def list_files() -> list[dict[str, Any]]:
    """Workspace files, newest first (capped), for the UI's Files list."""
    stamps = scan_workspace(WORKSPACE_DIR)
    rows = [
        {"path": path, "kind": artifact_kind(path), "size": s.size, "mtime": s.mtime_ns // 1_000_000}
        for path, s in stamps.items()
    ]
    rows.sort(key=lambda r: r["mtime"], reverse=True)
    return rows[:200]


@app.get("/api/files/{rel_path:path}")
def get_file(rel_path: str, embed: bool = False, download: bool = False) -> Response:
    """Serve a workspace file. Everything is sandboxed by CSP: HTML may run
    scripts but gets an opaque origin, so it can never call this API (and so
    never reach the agent's tools); nothing else may run scripts at all."""
    path = resolve_workspace_file(WORKSPACE_DIR, rel_path)
    if path is None:
        raise HTTPException(status_code=404, detail="File not found.")
    kind = artifact_kind(path.name)
    headers = {"X-Content-Type-Options": "nosniff", "Cache-Control": "no-cache"}
    if kind == "html":
        headers["Content-Security-Policy"] = "sandbox allow-scripts allow-popups"
    elif kind != "pdf":  # Chrome's PDF viewer refuses to run in a sandbox
        headers["Content-Security-Policy"] = "sandbox"
    if download:
        return FileResponse(path, filename=path.name, headers=headers)
    if kind == "html" and embed:
        html = path.read_text(encoding="utf-8", errors="replace")
        return HTMLResponse(inject_embed_script(html), headers=headers)
    return FileResponse(path, headers=headers)


app.mount("/", RevalidatingStaticFiles(directory=FRONTEND_DIR), name="frontend")
