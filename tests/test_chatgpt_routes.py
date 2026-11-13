"""The ChatGPT plan, end to end through the API: sign in, choose it, chat.

OpenAI is faked at three points: its auth service (tests/fake_openai_auth.py),
its model list, and its Responses endpoint, which records the credential and
SDK settings each request used.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

import app.auth.chatgpt as chatgpt
from app.auth.chatgpt import ChatGPTAuth
from app.auth.store import CredentialStore
from providers.openai_models import ModelInfo
from tests.fake_openai_auth import FakeOpenAIAuth

JSON = {"Content-Type": "application/json"}
PLAN_MODELS = [
    ModelInfo("gpt-6.1-sol", "GPT-6.1 Sol", 272000),
    ModelInfo("gpt-5.6-sol", "GPT-5.6 Sol", 272000),
]


class _Responses:
    def __init__(self, log: list[dict[str, Any]]) -> None:
        self._log = log

    async def create(self, **kwargs: Any):
        self._log[-1]["model"] = kwargs["model"]

        class Event:
            def __init__(self, data: dict[str, Any]) -> None:
                self._data = data

            def model_dump(self) -> dict[str, Any]:
                return self._data

        async def events():
            yield Event({"type": "response.output_text.delta", "delta": "Hello from the plan."})
            yield Event({"type": "response.completed", "response": {}})

        return events()


@pytest.fixture
def openai(monkeypatch: pytest.MonkeyPatch, tmp_path) -> dict[str, Any]:
    """Fake OpenAI, the real app: mock mode off, no API key."""
    import app.model_access as model_access
    import providers.openai_provider as provider_module
    from app.settings import reset_settings_for_tests

    monkeypatch.setenv("HUMPHREY_MOCK", "0")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.6-sol")
    reset_settings_for_tests()

    server = FakeOpenAIAuth()
    transport = httpx.MockTransport(server.handler)
    monkeypatch.setattr(
        chatgpt,
        "_instance",
        ChatGPTAuth(CredentialStore(tmp_path / "credentials"), http=lambda: httpx.AsyncClient(transport=transport)),
    )
    listed_with: list[str] = []

    async def fake_list_models(credential: str) -> list[ModelInfo]:
        listed_with.append(credential)
        return PLAN_MODELS

    monkeypatch.setattr(model_access, "list_models", fake_list_models)
    requests: list[dict[str, Any]] = []

    def fake_client(api_key: str, max_retries: int = 2):
        requests.append({"credential": api_key, "max_retries": max_retries})

        class Client:
            responses = _Responses(requests)

        return Client()

    monkeypatch.setattr(provider_module, "_client", fake_client)
    return {"server": server, "requests": requests, "listed_with": listed_with}


@pytest.fixture
async def local() -> AsyncClient:
    """Reached as the browser would: on localhost, at Humphrey's usual port."""
    from app.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost:8001") as client:
        yield client


async def _sign_in(client: AsyncClient, server: FakeOpenAIAuth) -> httpx.Response:
    url = (await client.post("/api/auth/chatgpt/start", headers=JSON, json={})).json()["url"]
    callback = server.approve(url)
    return await client.get("/api/auth/chatgpt/callback", params=callback)


async def test_status_starts_signed_out(local: AsyncClient, openai: dict) -> None:
    status = (await local.get("/api/auth/status")).json()
    assert status == {
        "signed_in": False,
        "email": None,
        "active_source": "api_key",
        "using_plan": False,
        "plan_notice_seen": False,
        "manage_usage_url": "https://chatgpt.com/settings/usage",
    }


async def test_sign_in_comes_back_to_this_server_on_127_0_0_1(local: AsyncClient, openai: dict) -> None:
    url = (await local.post("/api/auth/chatgpt/start", headers=JSON, json={})).json()["url"]
    redirect = parse_qs(urlsplit(url).query)["redirect_uri"][0]
    assert redirect == "http://127.0.0.1:8001/api/auth/chatgpt/callback"


async def test_the_callback_signs_in_and_shows_a_safe_page(local: AsyncClient, openai: dict) -> None:
    page = await _sign_in(local, openai["server"])
    assert page.status_code == 200
    assert "Signed in as user@example.com" in page.text
    assert page.headers["content-security-policy"].startswith("default-src 'none'")
    assert page.headers["referrer-policy"] == "no-referrer"
    assert "eyJ" not in page.text  # no token in the page
    status = (await local.get("/api/auth/status")).json()
    assert status["signed_in"] is True
    assert status["active_source"] == "api_key"  # signing in alone doesn't switch


async def test_a_bad_callback_explains_itself_with_escaped_text(local: AsyncClient, openai: dict) -> None:
    await local.post("/api/auth/chatgpt/start", headers=JSON, json={})
    page = await local.get("/api/auth/chatgpt/callback", params={"state": "<script>x</script>", "code": "c"})
    assert page.status_code == 400
    assert "<script>" not in page.text
    assert "expired or wasn" in page.text


@pytest.mark.parametrize(
    "path",
    [
        "/api/auth/chatgpt/start",
        "/api/auth/source",
        "/api/auth/chatgpt/plan-notice-seen",
        "/api/auth/chatgpt/sign-out",
    ],
)
async def test_auth_actions_need_a_json_request(local: AsyncClient, openai: dict, path: str) -> None:
    """A plain cross-site POST (no preflight) must not reach these."""
    response = await local.post(path, content=b"source=chatgpt", headers={"Content-Type": "text/plain"})
    assert response.status_code == 415


async def test_choosing_the_plan_needs_a_sign_in(local: AsyncClient, openai: dict) -> None:
    refused = await local.post("/api/auth/source", headers=JSON, json={"source": "chatgpt"})
    assert refused.status_code == 409
    await _sign_in(local, openai["server"])
    chosen = await local.post("/api/auth/source", headers=JSON, json={"source": "chatgpt"})
    assert chosen.json()["using_plan"] is True


async def test_with_the_plan_chosen_models_and_chats_use_it(local: AsyncClient, openai: dict) -> None:
    await _sign_in(local, openai["server"])
    await local.post("/api/auth/source", headers=JSON, json={"source": "chatgpt"})

    models = (await local.get("/api/models")).json()
    assert models == {
        "provider": "chatgpt",
        "selectable": True,
        "default": "gpt-5.6-sol",  # OPENAI_MODEL, since the plan offers it
        "models": ["gpt-6.1-sol", "gpt-5.6-sol"],
        "labels": {"gpt-6.1-sol": "GPT-6.1 Sol", "gpt-5.6-sol": "GPT-5.6 Sol"},
    }
    assert (await local.get("/healthz")).json()["provider"] == "chatgpt"

    reply = await local.post("/api/agent", json={"message": {"content": "hi"}, "model": "gpt-6.1-sol"})
    assert reply.status_code == 200
    assert "Hello from the plan." in reply.text
    # The chat, then the new chat's title: both on the plan, with its token
    # and with the SDK's own retries off.
    chat, title = openai["requests"]
    assert chat["model"] == title["model"] == "gpt-6.1-sol"
    for request in (chat, title):
        assert request["credential"] == openai["listed_with"][0]
        assert request["max_retries"] == 0

    unknown = await local.post("/api/agent", json={"message": {"content": "hi"}, "model": "gpt-4o"})
    assert unknown.status_code == 422


async def test_switching_back_to_the_api_key_stops_using_the_plan(local: AsyncClient, openai: dict) -> None:
    await _sign_in(local, openai["server"])
    await local.post("/api/auth/source", headers=JSON, json={"source": "chatgpt"})
    status = (await local.post("/api/auth/source", headers=JSON, json={"source": "api_key"})).json()
    assert status["using_plan"] is False
    assert (await local.get("/api/models")).json()["provider"] == "unconfigured"  # no key in this test


async def test_the_plan_notice_is_remembered(local: AsyncClient, openai: dict) -> None:
    await _sign_in(local, openai["server"])
    seen = await local.post("/api/auth/chatgpt/plan-notice-seen", headers=JSON, json={})
    assert seen.json()["plan_notice_seen"] is True


async def test_signing_out_revokes_and_returns_to_the_api_key(local: AsyncClient, openai: dict) -> None:
    await _sign_in(local, openai["server"])
    await local.post("/api/auth/source", headers=JSON, json={"source": "chatgpt"})
    status = (await local.post("/api/auth/chatgpt/sign-out", headers=JSON, json={})).json()
    assert openai["server"].revoked == ["rt-1"]
    assert status["signed_in"] is False
    assert status["active_source"] == "api_key"


async def test_mock_mode_still_wins(local: AsyncClient, openai: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.settings import reset_settings_for_tests

    await _sign_in(local, openai["server"])
    await local.post("/api/auth/source", headers=JSON, json={"source": "chatgpt"})
    monkeypatch.setenv("HUMPHREY_MOCK", "1")
    reset_settings_for_tests()
    assert (await local.get("/api/models")).json()["provider"] == "mock"
