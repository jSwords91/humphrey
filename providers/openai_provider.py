"""OpenAI provider (API key): Humphrey's primary model provider."""

from __future__ import annotations

from collections.abc import AsyncIterator
from functools import lru_cache

from openai import AsyncOpenAI

from brain.cancel import CancellationToken
from brain.messages import AgentMessage
from providers.openai_adapter import messages_to_openai, tools_to_openai
from providers.openai_responses import ResponsesStreamParser, build_responses_payload, uses_responses_api
from brain.provider import ModelChunk, ToolCallDelta
from app.settings import Settings, get_settings
from brain.tools import AgentTool


@lru_cache(maxsize=1)
def _client(api_key: str) -> AsyncOpenAI:
    return AsyncOpenAI(api_key=api_key)


class OpenAIProvider:
    """Streams from api.openai.com using an API key: Chat Completions, or
    the Responses API for reasoning models.

    Same wire shape as the optional AzureOpenAIProvider adapter, so either
    can sit behind the harness.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        if not self._settings.openai_api_configured:
            raise RuntimeError(
                "OPENAI_API_KEY must be set. For offline work set HUMPHREY_MOCK=1."
            )
        self._model = self._settings.openai_model

    @property
    def deployment_name(self) -> str:
        return self._model

    async def stream(
        self,
        messages: list[AgentMessage],
        tools: list[AgentTool],
        *,
        system: str,
        signal: CancellationToken | None = None,
    ) -> AsyncIterator[ModelChunk]:
        client = _client(self._settings.openai_api_key)
        if uses_responses_api(self._model):
            async for chunk in self._stream_responses(client, messages, tools, system, signal):
                yield chunk
            return

        openai_messages = messages_to_openai(messages, system=system)
        openai_tools = tools_to_openai(tools)
        response = await client.chat.completions.create(
            model=self._model,
            messages=openai_messages,  # type: ignore[arg-type]
            tools=openai_tools,  # type: ignore[arg-type]
            stream=True,
            stream_options={"include_usage": True},
            max_completion_tokens=8192,
        )
        async for chunk in response:
            if signal is not None and signal.is_cancelled():
                break
            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage
                yield ModelChunk(
                    usage={
                        "input": int(getattr(usage, "prompt_tokens", 0) or 0),
                        "output": int(getattr(usage, "completion_tokens", 0) or 0),
                        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
                    }
                )
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            tool_deltas: list[ToolCallDelta] = []
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    tool_deltas.append(
                        ToolCallDelta(
                            index=tc.index,
                            id=tc.id,
                            name=tc.function.name if tc.function else None,
                            arguments=tc.function.arguments if tc.function else None,
                        )
                    )
            if delta.content or tool_deltas:
                yield ModelChunk(content=delta.content, tool_calls=tool_deltas)

    async def _stream_responses(
        self,
        client: AsyncOpenAI,
        messages: list[AgentMessage],
        tools: list[AgentTool],
        system: str,
        signal: CancellationToken | None,
    ) -> AsyncIterator[ModelChunk]:
        payload = build_responses_payload(
            model=self._model,
            system=system,
            messages=messages,
            tools=tools,
            reasoning_effort=self._settings.reasoning_effort,
        )
        parser = ResponsesStreamParser()
        stream = await client.responses.create(**payload)
        async for event in stream:
            if signal is not None and signal.is_cancelled():
                break
            chunk = parser.feed(event.model_dump())
            if chunk is not None:
                yield chunk
