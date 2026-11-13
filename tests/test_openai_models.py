"""providers/openai_models.py: reading a credential's model list."""

from __future__ import annotations

import httpx
import pytest

from providers.errors import ProviderRequestError
from providers.openai_models import ModelInfo, list_models


def _http(handler):
    return lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_plan_models_use_slugs_and_only_listed_ones_are_offered() -> None:
    """The shape a ChatGPT plan returned live: no ``id``, a ``visibility``."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["authorization"])
        return httpx.Response(
            200,
            json={
                "data": [
                    {"slug": "gpt-6.1-sol", "display_name": "GPT-6.1 Sol", "visibility": "list", "context_window": 272000},
                    {"slug": "internal-model", "visibility": "hide"},
                    {"slug": "gpt-5.6-luna", "visibility": "list"},
                ]
            },
        )

    models = await list_models("the-token", http=_http(handler))
    assert models == [
        ModelInfo("gpt-6.1-sol", "GPT-6.1 Sol", 272000),
        ModelInfo("gpt-5.6-luna", "gpt-5.6-luna", None),
    ]
    assert seen == ["Bearer the-token"]


async def test_api_key_style_entries_with_ids_also_work() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "gpt-5.6-sol", "object": "model"}]})

    assert [m.slug for m in await list_models("k", http=_http(handler))] == ["gpt-5.6-sol"]


async def test_a_refusal_keeps_its_code() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401, json={"error": {"code": "invalid_api_key", "message": "Incorrect API key provided."}}
        )

    with pytest.raises(ProviderRequestError) as caught:
        await list_models("bad", http=_http(handler))
    assert (caught.value.code, caught.value.status) == ("invalid_api_key", 401)
