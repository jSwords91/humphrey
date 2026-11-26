"""Each model is compacted against its own context window.

A ChatGPT plan's model list gives each model's window; an API key and Azure
list none, so their models use DEFAULT_CONTEXT_WINDOW_TOKENS. Setting
MODEL_CONTEXT_WINDOW_TOKENS replaces every model's window.
"""

from __future__ import annotations

from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from app.settings import DEFAULT_CONTEXT_WINDOW_TOKENS
from providers.openai_models import ModelInfo


@pytest.fixture
def plan(monkeypatch: pytest.MonkeyPatch) -> None:
    """A signed-in ChatGPT plan whose list gives windows for some models."""
    import app.model_access as model_access

    monkeypatch.setattr(model_access, "current_source", lambda: "chatgpt")

    async def plan_models() -> list[ModelInfo]:
        return [
            ModelInfo("gpt-big", "Big", 400_000, ("low", "medium")),
            ModelInfo("gpt-small", "Small", 64_000, ("low", "medium")),
            ModelInfo("gpt-unlisted", "Unlisted", None, ()),
        ]

    monkeypatch.setattr(model_access, "plan_models", plan_models)


@pytest.fixture
async def client() -> AsyncClient:
    from app.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost:8001") as c:
        yield c


async def test_the_plans_listed_windows_are_offered(client: AsyncClient, plan: None) -> None:
    models = (await client.get("/api/models")).json()
    assert models["context_windows"] == {
        "gpt-big": 400_000,
        "gpt-small": 64_000,
        "gpt-unlisted": DEFAULT_CONTEXT_WINDOW_TOKENS,
    }


async def test_the_setting_replaces_every_models_window(
    client: AsyncClient, plan: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.settings import reset_settings_for_tests

    monkeypatch.setenv("MODEL_CONTEXT_WINDOW_TOKENS", "50000")
    reset_settings_for_tests()
    models = (await client.get("/api/models")).json()
    assert set(models["context_windows"].values()) == {50_000}
    assert (await client.get("/healthz")).json()["context_window_tokens"] == 50_000


async def test_a_source_that_lists_no_windows_uses_the_default(client: AsyncClient) -> None:
    models = (await client.get("/api/models")).json()  # the mock, under test
    assert models["context_windows"] == {"mock": DEFAULT_CONTEXT_WINDOW_TOKENS}
    assert (await client.get("/healthz")).json()["context_window_tokens"] == (
        DEFAULT_CONTEXT_WINDOW_TOKENS
    )


@pytest.mark.parametrize(("model", "window"), [("gpt-big", 400_000), ("gpt-small", 64_000)])
async def test_a_turn_compacts_against_the_chosen_models_window(
    client: AsyncClient, plan: None, monkeypatch: pytest.MonkeyPatch, model: str, window: int
) -> None:
    import app.turns
    from providers.mock import MockProvider

    class Provider(MockProvider):
        @property
        def deployment_name(self) -> str:
            return model

    async def build_provider(requested_model: str | None = None, effort: str | None = None):
        return Provider()

    windows: list[int] = []

    async def compact_if_needed(context: Any, **kwargs: Any) -> None:
        windows.append(kwargs["context_window_tokens"])
        return None

    monkeypatch.setattr(app.turns, "build_provider", build_provider)
    monkeypatch.setattr(app.turns, "compact_if_needed", compact_if_needed)
    reply = await client.post("/api/agent", json={"message": {"content": "hi"}, "model": model})
    assert reply.status_code == 200
    assert windows == [window]
