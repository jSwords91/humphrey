"""Turns: a message sent to a chat, run to its end and saved, whether or
not a page is watching.

``start_turn`` is the one way a turn begins. POST /api/agent starts one
for the person; a finished background task starts one in the chat that
started it, to report back (app/tasks.py). Everything from here down is
the same for both: one turn at a time per chat, the chat's project folder,
compaction, the run and its frame log (app/runs.py), artefacts, and the
save. The caller decides what's sent; nothing here knows about HTTP.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any, Literal

from app import projects, runs
from app.agent import build_harness, system_prompt
from app.artifacts import ArtifactTracker
from app.context_window import compact_if_needed
from app.model_access import UnknownModel, build_provider, context_window_for
from app.projects import ProjectError
from app.rendering.sse import SseRenderer, encode_sse
from app.runs import FrameLog, Run
from app.sessions import (
    Session,
    generate_session_title,
    get_session_store,
    new_session_id,
)
from app.tasks import ChatTasks
from app.tools import TOOLS, WORKSPACE_DIR, tools_for_run
from app.tools.tasks import IN_A_TASK
from app.ui_bridge import WebUiBridge
from brain.events import ToolExecutionEndEvent, ToolExecutionStartEvent
from brain.messages import AgentMessage, ImageContent, TextContent, UserMessage

logger = logging.getLogger(__name__)


class TurnRefused(Exception):
    """A turn that can't start. ``kind`` says why: the chat is still busy,
    the request is invalid, or the chat's folder is gone."""

    def __init__(self, kind: Literal["busy", "invalid", "gone"], message: str) -> None:
        super().__init__(message)
        self.kind = kind


async def start_turn(
    session_id: str | None,
    content: str | Sequence[TextContent | ImageContent],
    *,
    shown: str,
    model: str | None = None,
    effort: str | None = None,
    project: str | None = None,
    parent_id: str | None = None,
    seed: Sequence[AgentMessage] = (),
    title: str | None = None,
) -> Run:
    """Start a turn in ``session_id`` (a new chat if None or unknown) and
    return its run, already working in a task of its own.

    ``content`` is what the model is sent; ``shown`` is what the person
    typed, which names a new chat and is shown for a run in progress.
    ``project`` is for a new chat; an existing one keeps its own. A
    background task's chat is new too, with the chat that started it
    (``parent_id``), the conversation a fork starts from (``seed``) and
    its ``title`` (app/tasks.py).
    """
    store = get_session_store()

    # One turn at a time per chat. A turn that has finished but isn't
    # saved yet is waited for, or this one would start from the history
    # before it and then save over it.
    previous = runs.get(session_id) if session_id else None
    if previous is not None and previous.harness.is_running:
        raise TurnRefused("busy", "This chat is still working: steer it or queue a follow-up.")
    if previous is not None:
        await previous.wait_saved()

    existing = await store.get(session_id) if session_id else None
    if session_id is None:
        session_id = new_session_id()
    session = existing or Session(
        id=session_id,
        parent_id=parent_id,
        title=title,
        messages=list(seed),
        inherited=len(seed),
    )

    try:
        provider = await build_provider(model, effort)
        # Housekeeping calls (compaction summaries, the chat's title) use the
        # default effort: a "max" chat shouldn't make every title a max call.
        housekeeping = await build_provider(model)
    except UnknownModel as exc:
        raise TurnRefused("invalid", str(exc)) from exc

    project = session.project if existing is not None else project
    try:
        workdir = projects.folder(WORKSPACE_DIR, project)
    except ProjectError as exc:
        if existing is not None:  # its folder was renamed or removed on disk
            raise TurnRefused("gone", f"{exc} Start a new chat to carry on.") from exc
        raise TurnRefused("invalid", str(exc)) from exc
    session.project = project

    # Every check that can refuse the turn is above this line: the store
    # may hand back the live session, so nothing changes it before here.
    last = session.messages[-1] if session.messages else None
    if isinstance(last, UserMessage) and last.content == shown:
        session.messages.pop()  # a retried turn: don't store the prompt twice
    history = session.model_context()
    compaction = await compact_if_needed(
        history,
        provider=housekeeping,
        system=system_prompt(workdir),
        tools=TOOLS,
        context_window_tokens=await context_window_for(provider.deployment_name),
    )
    if compaction is not None:
        session.record_compaction(compaction.summary, kept_from=compaction.kept_from)
        history = session.model_context()

    prompt_message = UserMessage(content=content if isinstance(content, str) else list(content))
    log = FrameLog()
    bridge = WebUiBridge(lambda payload: log.publish(encode_sse({**payload, "session_id": session_id})))
    # A chat can start background tasks; a task's own chat can't.
    tasks = (
        ChatTasks(session_id, project=project, model=model, effort=effort)
        if session.parent_id is None
        else IN_A_TASK
    )
    harness = build_harness(
        session_id=session_id,
        provider=provider,
        messages=history,
        tools=tools_for_run(bridge, workdir, tasks),
        workspace=workdir,
    )
    if isinstance(tasks, ChatTasks):
        tasks.context = lambda: harness.messages  # what a fork starts from
    # Started here rather than in the task, so the run counts as running
    # from the moment it can be found.
    events = harness.prompt_message(prompt_message)
    run = Run(
        session_id,
        harness=harness,
        bridge=bridge,
        log=log,
        prompt=shown,
        started_ms=prompt_message.timestamp_ms,
        new_chat=existing is None,
        project=project,
        parent_id=session.parent_id,
        title=session.title,
    )
    tracker = ArtifactTracker(workdir, projects.scan_exclusions(project))

    async def run_turn() -> None:
        """Run the turn and save it. Whoever follows the log sees its
        frames; nobody needs to, and a tool can wait on the bridge while
        its question card goes out."""
        renderer = SseRenderer(session_id=session_id)
        try:
            async for event in events:
                wire = renderer.to_sse(event)
                if wire is not None:
                    log.publish(wire)
                if isinstance(event, ToolExecutionStartEvent):
                    tracker.tool_started(event.tool_name)  # before the tool runs
                if isinstance(event, ToolExecutionEndEvent):
                    made = tracker.tool_finished(event.tool_name, event.result.details, event.is_error)
                    for artifact in made:
                        # Shown by its path in the project, served by its path in the workspace.
                        artifact["name"] = artifact["path"]
                        artifact["path"] = projects.served_path(project, artifact["name"])
                        artifact["version"] = 1 + sum(
                            1 for a in session.artifacts if a.get("path") == artifact["path"]
                        )
                        session.artifacts.append(artifact)
                        log.publish(encode_sse({"type": "artifact", **artifact}))
        except Exception:  # noqa: BLE001
            logger.exception("turn failed for session %s", session_id)
        finally:
            try:
                await events.aclose()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                logger.debug("agent stream aclose failed", exc_info=True)
            bridge.close()
            if run.stopped:
                harness.append_interrupted_tool_results()
            # The harness started from model_context(); append only what this
            # turn added, so the stored transcript stays complete.
            session.messages.extend(harness.messages[len(history) :])
            await finish_turn(session, store, housekeeping, shown, on_saved=run.mark_saved)
            # A follow-up that arrived as the turn ended (a task's result, or
            # one the person queued) would otherwise be lost: it starts the
            # next turn instead.
            leftover = harness.clear_queues()
            if leftover.count and not run.stopped:
                text = "\n\n".join(m.text for m in (*leftover.steering, *leftover.follow_up))
                try:
                    await start_turn(session_id, text, shown=text, model=model, effort=effort)
                except TurnRefused as exc:
                    logger.warning("a follow-up for %s was lost: %s", session_id, exc)

    runs.start(run, run_turn())
    return run


async def finish_turn(
    session: Session,
    store: Any,
    title_provider: Any,
    first_message: str,
    *,
    on_saved: Callable[[], None] = lambda: None,
) -> None:
    """Save the turn, then name a new chat and save again.

    The transcript is saved first, so the chat is never lost to a slow or
    failing title call (if the title model fails, generate_session_title
    falls back to the start of the message). ``on_saved`` runs as soon as
    the turn is stored, which lets the chat's next turn begin while this
    one is still being named.
    """

    async def save() -> None:
        try:
            await store.save(session)
        except Exception:  # noqa: BLE001
            logger.exception("failed to persist session %s", session.id)

    await save()
    on_saved()
    if session.title is None:
        title = await generate_session_title(title_provider, first_message)
        if title:
            session.title = title
            await save()
