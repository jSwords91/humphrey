"""Provider-neutral transcript messages for the agent harness.

Typed to match Hugging Face tau's ``tau_agent.messages`` shape (``WireModel``
base, content-block lists, a role-discriminated union) so this harness stays
a portable brain in the same sense tau's does. Scoped down to the three roles
this app needs — no image/bash/compaction/branch content, which are
tau_coding-specific and unused here. ``ThinkingContent`` is kept, unlike the
others, because reasoning models (see ``app/openai_responses.py``) need it to
carry a visible summary plus the opaque item tau's ``thinking_signature``
convention replays on the next turn.
"""

from __future__ import annotations

from time import time
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _to_camel(name: str) -> str:
    parts = name.split("_")
    return parts[0] + "".join(part.title() for part in parts[1:])


def _now_ms() -> int:
    return int(time() * 1000)


class WireModel(BaseModel):
    """Strict model with Python field names and camelCase JSON aliases."""

    model_config = ConfigDict(
        extra="forbid",
        validate_by_name=True,
        validate_by_alias=True,
        serialize_by_alias=True,
        alias_generator=_to_camel,
    )


StopReason = Literal["stop", "length", "tool_use", "error", "aborted"]


class TextContent(WireModel):
    type: Literal["text"] = "text"
    text: str


class ImageContent(WireModel):
    """A processed, provider-safe image (see app/image_processing.py).

    ``data`` is base64. Only ever appears in ``UserMessage.content`` — our
    tools don't produce images, unlike tau's (which also allows image tool
    results), so ToolResultMessage.content is deliberately left untouched.
    """

    type: Literal["image"] = "image"
    data: str
    mime_type: str


class ToolCall(WireModel):
    """A tool call content block requested by the assistant."""

    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ThinkingContent(WireModel):
    """A reasoning-model's visible reasoning summary.

    ``thinking`` is the visible summary text (safe to show in the UI).
    ``thinking_signature`` is the opaque reasoning item the Responses API
    returns once per turn, JSON-serialized — it must be replayed verbatim on
    the next request or the model loses its reasoning continuity (see
    ``messages_to_responses_input`` in ``app/openai_responses.py``). It is
    never shown to a user.
    """

    type: Literal["thinking"] = "thinking"
    thinking: str
    thinking_signature: str | None = None


AssistantContent = Annotated[
    TextContent | ToolCall | ThinkingContent, Field(discriminator="type")
]

UserContentBlock = Annotated[TextContent | ImageContent, Field(discriminator="type")]


class UserMessage(WireModel):
    # A bare string remains the common case (typed text); a block list is
    # used only when images are attached (see app/image_processing.py and
    # app/main.py's /api/agent handler) — mirrors tau_agent's UserContent.
    role: Literal["user"] = "user"
    content: str | list[UserContentBlock]
    timestamp_ms: int = Field(default_factory=_now_ms)

    @property
    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return "".join(b.text for b in self.content if isinstance(b, TextContent))

    @property
    def images(self) -> list[ImageContent]:
        if isinstance(self.content, str):
            return []
        return [b for b in self.content if isinstance(b, ImageContent)]


class AssistantMessage(WireModel):
    role: Literal["assistant"] = "assistant"
    content: list[AssistantContent] = Field(default_factory=list)
    model: str = "unknown"
    stop_reason: StopReason = "stop"
    error_message: str | None = None
    timestamp_ms: int = Field(default_factory=_now_ms)
    usage: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize_convenient_content(cls, value: object) -> object:
        """Accept a bare string as a construction convenience only.

        The stored model is always block based; this keeps call sites (tests,
        provider adapters) concise without a second message representation.
        """
        if not isinstance(value, dict):
            return value
        data = dict(value)
        content = data.get("content")
        if isinstance(content, str):
            data["content"] = [TextContent(text=content)] if content else []
        return data

    @property
    def text(self) -> str:
        return "".join(b.text for b in self.content if isinstance(b, TextContent))

    @property
    def thinking_text(self) -> str:
        return "".join(b.thinking for b in self.content if isinstance(b, ThinkingContent))

    @property
    def thinking_signature(self) -> str | None:
        for block in self.content:
            if isinstance(block, ThinkingContent):
                return block.thinking_signature
        return None

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        return tuple(b for b in self.content if isinstance(b, ToolCall))


class ToolResultMessage(WireModel):
    role: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    tool_name: str
    content: list[TextContent] = Field(default_factory=list)
    details: Any = None
    is_error: bool = False
    timestamp_ms: int = Field(default_factory=_now_ms)

    @model_validator(mode="before")
    @classmethod
    def _normalize_convenient_content(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        content = data.get("content")
        if isinstance(content, str):
            data["content"] = [TextContent(text=content)] if content else []
        return data

    @property
    def text(self) -> str:
        return "".join(b.text for b in self.content)


AgentMessage = Annotated[
    UserMessage | AssistantMessage | ToolResultMessage,
    Field(discriminator="role"),
]

INTERRUPTED_TOOL_RESULT = "Tool call interrupted by user"


def message_to_dict(message: AgentMessage) -> dict[str, Any]:
    return message.model_dump(by_alias=True)


def message_from_dict(data: dict[str, Any]) -> AgentMessage | None:
    role = str(data.get("role") or "")
    if role == "user":
        return UserMessage.model_validate(data)
    if role == "assistant":
        return AssistantMessage.model_validate(data)
    if role in {"tool_result", "tool"}:
        # Accept legacy OpenAI-shaped tool rows when migrating.
        if role == "tool" and "tool_call_id" in data:
            text = str(data.get("content") or "")
            return ToolResultMessage(
                tool_call_id=str(data["tool_call_id"]),
                tool_name=str(data.get("tool_name") or "unknown"),
                content=[TextContent(text=text)] if text else [],
                is_error=bool(data.get("is_error")),
            )
        return ToolResultMessage.model_validate(data)
    return None


# Marks a synthetic UserMessage injected by app.context_window's
# auto-compaction as standing in for summarized history, not something the
# user actually typed. Defined here (not in context_window.py) so ui_messages
# can distinguish it without a circular import.
COMPACTION_SUMMARY_PREFIX = "Previous conversation summary:\n"


def ui_messages(messages: list[AgentMessage]) -> list[dict[str, Any]]:
    """Project transcript to user/assistant text pairs for the web UI.

    Each row carries its message's ``ts`` (ms) so the UI can interleave
    other timestamped items, such as artefacts, in order.
    """
    out: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, UserMessage):
            text = message.text.strip()
            image_count = len(message.images)
            if image_count and not text:
                # An image-only turn (no caption) still needs a visible row,
                # or it silently vanishes from session history/UI — the
                # actual image bytes aren't replayed here (see module
                # docstring on ImageContent: kept out of the UI projection
                # to avoid bloating every session-list/reload payload).
                noun = "image" if image_count == 1 else "images"
                out.append({"role": "user", "content": f"(sent {image_count} {noun})", "ts": message.timestamp_ms})
                continue
            if not text:
                continue
            if text.startswith(COMPACTION_SUMMARY_PREFIX):
                out.append(
                    {
                        "role": "context_summary",
                        "content": text.removeprefix(COMPACTION_SUMMARY_PREFIX),
                        "ts": message.timestamp_ms,
                    }
                )
            else:
                # Don't let a caption's text silently hide that images were
                # also attached to this turn — same "no dropped information"
                # principle as elsewhere in this app's history projections.
                if image_count:
                    noun = "image" if image_count == 1 else "images"
                    text = f"{text}\n(+{image_count} {noun} attached)"
                out.append({"role": "user", "content": text, "ts": message.timestamp_ms})
        elif isinstance(message, AssistantMessage):
            text = message.text.strip()
            if text and message.stop_reason not in {"error", "aborted"}:
                out.append({"role": "assistant", "content": text, "ts": message.timestamp_ms})
    return out


def assistant_from_stream(
    *,
    text: str,
    tool_calls: list[ToolCall],
    model: str,
    usage: dict[str, int] | None = None,
    thinking: str | None = None,
    thinking_signature: str | None = None,
) -> AssistantMessage:
    blocks: list[AssistantContent] = []
    if thinking or thinking_signature:
        blocks.append(
            ThinkingContent(thinking=thinking or "", thinking_signature=thinking_signature)
        )
    if text:
        blocks.append(TextContent(text=text))
    blocks.extend(tool_calls)
    return AssistantMessage(
        content=blocks,
        model=model,
        stop_reason="tool_use" if tool_calls else "stop",
        usage=dict(usage or {}),
    )


def error_assistant(model: str, message: str) -> AssistantMessage:
    return AssistantMessage(
        content=[],
        model=model,
        stop_reason="error",
        error_message=message,
    )
