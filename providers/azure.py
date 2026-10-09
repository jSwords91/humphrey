"""Optional Azure OpenAI adapter (Entra ID auth via DefaultAzureCredential).

Needs the `azure` dependency group: `uv sync --group azure`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from functools import lru_cache

import httpx
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AsyncAzureOpenAI, AsyncOpenAI

from brain.cancel import CancellationToken
from brain.messages import AgentMessage
from providers.openai_adapter import messages_to_openai, tools_to_openai
from providers.openai_responses import ResponsesStreamParser, build_responses_payload, uses_responses_api
from brain.provider import ModelChunk, ToolCallDelta
from app.settings import Settings, get_settings
from brain.tools import AgentTool

_COGNITIVE_SCOPE = "https://cognitiveservices.azure.com/.default"

# The v1/Responses surface is not "the existing AzureOpenAI client with a
# different api_version" — per Microsoft's own migration guide it is a
# different client entirely (the plain OpenAI SDK class, not AzureOpenAI),
# pointed at a /openai/v1/ base_url, authenticated under a different token
# scope. Live-verified against a real Azure deployment.
_AI_FOUNDRY_SCOPE = "https://ai.azure.com/.default"


@lru_cache(maxsize=1)
def _client(endpoint: str, api_version: str) -> AsyncAzureOpenAI:
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        _COGNITIVE_SCOPE,
    )
    return AsyncAzureOpenAI(
        azure_endpoint=endpoint,
        api_version=api_version,
        azure_ad_token_provider=token_provider,
    )


class _AzureADTokenAuth(httpx.Auth):
    """Adapts a per-call azure-identity token provider into httpx auth.

    Unlike ``AsyncAzureOpenAI``, the plain ``AsyncOpenAI`` class has no
    ``azure_ad_token_provider`` hook — it bakes ``api_key`` into the
    Authorization header once, as a literal string, at construction. Passing
    the callable there directly (this session's actual bug, live-verified
    against a real Azure deployment: a 401 "invalid subscription key" caused
    by the callable being stringified into the header) never invokes it.
    ``auth_flow`` runs per request, after the SDK builds its own headers, so
    it can correctly overwrite that placeholder with a real, freshly-fetched
    token on every call.
    """

    def __init__(self, token_provider: Callable[[], str]) -> None:
        self._token_provider = token_provider

    def auth_flow(self, request: httpx.Request) -> Iterator[httpx.Request]:
        request.headers["Authorization"] = f"Bearer {self._token_provider()}"
        yield request


@lru_cache(maxsize=1)
def _v1_client(endpoint: str) -> AsyncOpenAI:
    """Client for the Responses API only — see the module docstring above."""
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        _AI_FOUNDRY_SCOPE,
    )
    return AsyncOpenAI(
        base_url=f"{endpoint.rstrip('/')}/openai/v1/",
        api_key="unused-real-token-injected-by-http-client-auth",
        http_client=httpx.AsyncClient(auth=_AzureADTokenAuth(token_provider)),
    )


class AzureOpenAIProvider:
    """Streams chat completions from Azure AI Foundry / OpenAI."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        if not self._settings.openai_configured:
            raise RuntimeError(
                "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT must be set. "
                "For offline work set HUMPHREY_MOCK=1."
            )
        self._deployment = self._settings.azure_openai_deployment

    @property
    def deployment_name(self) -> str:
        return self._deployment

    async def stream(
        self,
        messages: list[AgentMessage],
        tools: list[AgentTool],
        *,
        system: str,
        signal: CancellationToken | None = None,
    ) -> AsyncIterator[ModelChunk]:
        if uses_responses_api(self._deployment):
            async for chunk in self._stream_responses(messages, tools, system, signal):
                yield chunk
            return

        client = _client(
            self._settings.azure_openai_endpoint,
            self._settings.azure_openai_api_version,
        )
        openai_messages = messages_to_openai(messages, system=system)
        openai_tools = tools_to_openai(tools)
        response = await client.chat.completions.create(
            model=self._deployment,
            messages=openai_messages,  # type: ignore[arg-type]
            tools=openai_tools,  # type: ignore[arg-type]
            stream=True,
            stream_options={"include_usage": True},
            # gpt-5-mini shares this budget with reasoning tokens; keep headroom
            # so longer answers (and tool turns) are not truncated mid-stream.
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
        messages: list[AgentMessage],
        tools: list[AgentTool],
        system: str,
        signal: CancellationToken | None,
    ) -> AsyncIterator[ModelChunk]:
        client = _v1_client(self._settings.azure_openai_endpoint)
        payload = build_responses_payload(
            model=self._deployment,
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
