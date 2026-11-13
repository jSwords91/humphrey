"""Choosing a reasoning effort per chat, with REASONING_EFFORT as the default.

The levels on offer were checked live on both the API key and the ChatGPT
plan: low, medium, high, xhigh and max are accepted; "minimal" and "ultra"
are refused, though the plan's model list advertises "ultra".
"""

from __future__ import annotations

from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from providers.openai_models import ModelInfo

ALL = ["low", "medium", "high", "xhigh", "max"]


@pytest.fixture
def api_key(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """The API-key route with a fake Responses endpoint that records requests."""
    import providers.openai_provider as provider_module
    from app.settings import reset_settings_for_tests

    monkeypatch.setenv("HUMPHREY_MOCK", "0")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.6-sol")
    monkeypatch.setenv("OPENAI_MODELS", "gpt-6.1-sol,gpt-4o-mini")
    monkeypatch.delenv("REASONING_EFFORT", raising=False)
    reset_settings_for_tests()
    requests: list[dict[str, Any]] = []

    class Responses:
        async def create(self, **kwargs: Any):
            requests.append(kwargs)

            class Event:
                def __init__(self, data: dict[str, Any]) -> None:
                    self._data = data

                def model_dump(self) -> dict[str, Any]:
                    return self._data

            async def events():
                yield Event({"type": "response.output_text.delta", "delta": "A title"})
                yield Event({"type": "response.completed", "response": {}})

            return events()

    class Client:
        responses = Responses()

    monkeypatch.setattr(provider_module, "_client", lambda api_key, max_retries=2: Client())
    return requests


@pytest.fixture
async def client() -> AsyncClient:
    from app.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost:8001") as c:
        yield c


async def test_api_key_models_offer_the_checked_levels_and_a_clear_default(
    client: AsyncClient, api_key: list
) -> None:
    models = (await client.get("/api/models")).json()
    assert models["default_effort"] == "medium"  # REASONING_EFFORT's default
    assert models["efforts"] == {
        "gpt-5.6-sol": ALL,
        "gpt-6.1-sol": ALL,
        "gpt-4o-mini": [],  # a Chat Completions model: no reasoning to choose
    }


async def test_an_empty_reasoning_effort_setting_turns_the_choice_off(
    client: AsyncClient, api_key: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.settings import reset_settings_for_tests

    monkeypatch.setenv("REASONING_EFFORT", "")
    reset_settings_for_tests()
    models = (await client.get("/api/models")).json()
    assert models["default_effort"] is None
    assert all(levels == [] for levels in models["efforts"].values())


async def test_the_chosen_effort_is_used_and_housekeeping_stays_at_the_default(
    client: AsyncClient, api_key: list
) -> None:
    reply = await client.post(
        "/api/agent", json={"message": {"content": "hi"}, "reasoning_effort": "max"}
    )
    assert reply.status_code == 200
    chat, title = api_key  # the reply, then the new chat's title
    assert chat["reasoning"] == {"effort": "max", "summary": "auto"}
    assert title["reasoning"] == {"effort": "medium", "summary": "auto"}


async def test_without_a_choice_the_default_applies(client: AsyncClient, api_key: list) -> None:
    await client.post("/api/agent", json={"message": {"content": "hi"}})
    assert api_key[0]["reasoning"]["effort"] == "medium"


@pytest.mark.parametrize(
    ("model", "effort"),
    [
        (None, "ultra"),  # advertised by the plan, refused by the API
        (None, "minimal"),  # refused by these models
        (None, "turbo"),
        ("gpt-4o-mini", "high"),  # no reasoning on a Chat Completions model
    ],
)
async def test_an_effort_the_model_does_not_offer_is_refused(
    client: AsyncClient, api_key: list, model: str | None, effort: str
) -> None:
    body: dict[str, Any] = {"message": {"content": "hi"}, "reasoning_effort": effort}
    if model:
        body["model"] = model
    response = await client.post("/api/agent", json=body)
    assert response.status_code == 422
    assert api_key == []  # nothing reached the model


async def test_the_plan_offers_each_models_own_levels_minus_ultra(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.model_access as model_access
    from app.settings import reset_settings_for_tests

    monkeypatch.setenv("HUMPHREY_MOCK", "0")
    monkeypatch.delenv("REASONING_EFFORT", raising=False)
    reset_settings_for_tests()
    monkeypatch.setattr(model_access, "current_source", lambda: "chatgpt")

    async def plan_models() -> list[ModelInfo]:
        return [
            ModelInfo("gpt-5.6-sol", "GPT-5.6 Sol", 272000, ("low", "medium", "high", "xhigh", "max", "ultra")),
            ModelInfo("gpt-5.6-luna", "GPT-5.6 Luna", 272000, ("low", "medium", "high", "xhigh", "max")),
            ModelInfo("gpt-mini", "Mini", 128000, ("low", "medium")),
            ModelInfo("gpt-plain", "Plain", 128000, ()),  # advertises no reasoning
        ]

    monkeypatch.setattr(model_access, "plan_models", plan_models)
    options = await model_access.model_options()
    assert options.efforts == {
        "gpt-5.6-sol": ALL,
        "gpt-5.6-luna": ALL,
        "gpt-mini": ["low", "medium"],
        "gpt-plain": [],
    }
    assert options.default_effort == "medium"
