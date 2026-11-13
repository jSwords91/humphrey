"""The models a credential can use, from OpenAI's GET /v1/models.

For a ChatGPT plan, this is the account's own list (it differs from the API
key's), and its entries use ``slug`` rather than ``id`` and carry a
``visibility``: only ``"list"`` entries are meant for a model picker. Read
with httpx rather than the SDK, whose model type expects an ``id``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from providers.errors import ProviderRequestError, error_from_body

MODELS_URL = "https://api.openai.com/v1/models"


@dataclass(frozen=True, slots=True)
class ModelInfo:
    slug: str
    display_name: str
    context_window: int | None = None
    # Reasoning efforts the list advertises for this model, in its order. As
    # advertised, not as checked: callers filter for what the API accepts.
    reasoning_efforts: tuple[str, ...] = ()


async def list_models(
    credential: str,
    *,
    http: Callable[[], httpx.AsyncClient] = lambda: httpx.AsyncClient(timeout=15),
) -> list[ModelInfo]:
    async with http() as client:
        try:
            response = await client.get(MODELS_URL, headers={"Authorization": f"Bearer {credential}"})
        except httpx.HTTPError as exc:
            raise ProviderRequestError("Couldn't reach OpenAI to list models.", code="connection_error") from exc
    try:
        body: Any = response.json()
    except ValueError:
        body = None
    if response.status_code != 200:
        raise error_from_body(
            body, status=response.status_code, request_id=response.headers.get("x-request-id")
        )
    entries = (body or {}).get("data") or (body or {}).get("models") or []
    models = []
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("visibility", "list") != "list":
            continue
        slug = entry.get("slug") or entry.get("id")
        if isinstance(slug, str) and slug:
            window = entry.get("context_window")
            levels = entry.get("supported_reasoning_levels") or []
            models.append(
                ModelInfo(
                    slug=slug,
                    display_name=str(entry.get("display_name") or slug),
                    context_window=window if isinstance(window, int) else None,
                    reasoning_efforts=tuple(
                        level["effort"]
                        for level in levels
                        if isinstance(level, dict) and isinstance(level.get("effort"), str)
                    ),
                )
            )
    return models
