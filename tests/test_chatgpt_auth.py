"""Sign in with ChatGPT (app/auth): sign-in, refresh, sign-out, and storage.

A fake of OpenAI's auth service stands in for the real one. It checks PKCE,
issues a client id at first registration, rotates refresh tokens, refuses a
reused one, and records every request. The flow and its parameters match a
live sign-in done while designing this. Fake tokens are built at runtime, so
nothing here looks like a real credential to tests/test_no_secrets.py.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import stat
import time
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

import app.auth.chatgpt as chatgpt
from app.auth.chatgpt import AuthError, ChatGPTAuth, SignInRequired, TemporaryAuthError
from app.auth.store import CredentialStore, Tokens

ISSUED_CLIENT_ID = "oaiapp" + "_" + "fake" * 3


def _jwt(claims: dict) -> str:
    def part(data: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(data).encode()).rstrip(b"=").decode()

    return ".".join([part({"alg": "none"}), part(claims), "signature"])


class FakeOpenAIAuth:
    """Just enough of auth.openai.com, with knobs for failure cases."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, str]]] = []
        self.codes: dict[str, dict[str, str]] = {}
        self.refresh_token = "rt-1"
        self.generation = 1
        self.grant_plan_scope = True
        self.nonce_override: str | None = None
        self.exchange_status = 200
        self.refresh_failure: tuple[int, str] | None = None  # (status, oauth error)
        self.refresh_unreachable = False
        self.revoke_status = 200
        self.revoked: list[str] = []

    # The browser's half: consent was given, OpenAI redirects back.
    def approve(self, authorize_url: str) -> dict[str, str]:
        params = {k: v[0] for k, v in parse_qs(urlsplit(authorize_url).query).items()}
        code = f"code-{len(self.codes)}"
        self.codes[code] = params
        callback = {"code": code, "state": params["state"], "scope": params["scope"]}
        if params["client_id"] == "dynamic_agent_client":
            callback["client_id"] = ISSUED_CLIENT_ID
        return callback

    def _tokens(self, nonce: str) -> dict:
        scope = "openid profile email offline_access resource.invoke"
        if self.grant_plan_scope:
            scope += " chatgpt.tokens.use.direct"
        return {
            "access_token": _jwt({"gen": self.generation, "exp": int(time.time()) + 3600}),
            "refresh_token": self.refresh_token,
            "id_token": _jwt(
                {"nonce": self.nonce_override or nonce, "email": "user@example.com", "sub": "user-1"}
            ),
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": scope,
        }

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("openid-configuration"):
            return httpx.Response(200, json={"revocation_endpoint": "https://auth.openai.com/oauth/revoke"})
        form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
        self.requests.append((str(request.url), form))
        if request.url.path.endswith("/revoke"):
            self.revoked.append(form.get("token", ""))
            return httpx.Response(self.revoke_status)
        if form["grant_type"] == "authorization_code":
            if self.exchange_status != 200:
                return httpx.Response(self.exchange_status, json={"error": "invalid_grant", "error_description": "Bad code."})
            issued = self.codes.pop(form["code"])  # a code works once
            challenge = base64.urlsafe_b64encode(hashlib.sha256(form["code_verifier"].encode()).digest())
            assert challenge.rstrip(b"=").decode() == issued["code_challenge"], "PKCE verifier mismatch"
            assert form["redirect_uri"] == issued["redirect_uri"]
            assert form["resource"] == "https://api.openai.com/v1"
            return httpx.Response(200, json=self._tokens(issued["nonce"]))
        if form["grant_type"] == "refresh_token":
            if self.refresh_unreachable:
                raise httpx.ConnectError("unreachable", request=request)
            if self.refresh_failure:
                status, error = self.refresh_failure
                return httpx.Response(status, json={"error": error})
            if form["refresh_token"] != self.refresh_token:
                return httpx.Response(400, json={"error": "refresh_token_reused"})
            self.generation += 1
            self.refresh_token = f"rt-{self.generation}"  # rotates every time
            body = self._tokens("unused")
            del body["id_token"]  # a refresh may omit it; the old one is kept
            return httpx.Response(200, json=body)
        return httpx.Response(400)


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> FakeOpenAIAuth:
    monkeypatch.setattr(chatgpt, "REVOKE_BACKOFF_SECONDS", 0)
    return FakeOpenAIAuth()


@pytest.fixture
def auth(tmp_path: Path, server: FakeOpenAIAuth) -> ChatGPTAuth:
    transport = httpx.MockTransport(server.handler)
    return ChatGPTAuth(
        CredentialStore(tmp_path / "credentials"),
        http=lambda: httpx.AsyncClient(transport=transport),
    )


async def _sign_in(auth: ChatGPTAuth, server: FakeOpenAIAuth, port: int = 8001) -> str:
    return await auth.complete_sign_in(server.approve(auth.start_sign_in(port)))


def _stored(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "credentials" / "chatgpt.json").read_text())


# --- sign-in -------------------------------------------------------------------


async def test_first_sign_in_registers_this_app(auth: ChatGPTAuth, server: FakeOpenAIAuth, tmp_path: Path) -> None:
    url = auth.start_sign_in(8001)
    params = {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}
    assert url.startswith("https://auth.openai.com/api/accounts/authorize?")
    assert params["client_id"] == "dynamic_agent_client"
    assert params["agent_name_hint"] == "Humphrey"
    assert params["ext_agent_host_id"].startswith("urn:uuid:")
    assert params["redirect_uri"] == "http://127.0.0.1:8001/api/auth/chatgpt/callback"
    assert params["scope"].split() == [
        "openid", "profile", "email", "offline_access", "resource.invoke", "chatgpt.tokens.use.direct"
    ]
    assert params["resource"] == "https://api.openai.com/v1"
    assert params["code_challenge_method"] == "S256"
    assert params["state"] and params["nonce"]

    assert await auth.complete_sign_in(server.approve(url)) == "user@example.com"
    stored = _stored(tmp_path)
    [registration] = stored["registrations"]
    assert registration["client_id"] == ISSUED_CLIENT_ID
    assert stored["active_client_id"] == ISSUED_CLIENT_ID
    assert stored["host_id"] == params["ext_agent_host_id"]
    assert stored["active_source"] == "api_key"  # signing in doesn't switch on its own


async def test_signing_in_again_reuses_the_issued_client(auth: ChatGPTAuth, server: FakeOpenAIAuth, tmp_path: Path) -> None:
    await _sign_in(auth, server)
    host_id = _stored(tmp_path)["host_id"]
    params = parse_qs(urlsplit(auth.start_sign_in(8123)).query)
    assert params["client_id"] == [ISSUED_CLIENT_ID]
    assert "agent_name_hint" not in params
    assert params["ext_agent_host_id"] == [host_id]
    assert params["redirect_uri"] == ["http://127.0.0.1:8123/api/auth/chatgpt/callback"]  # only the port varies


async def test_the_credential_file_is_owner_only(auth: ChatGPTAuth, server: FakeOpenAIAuth, tmp_path: Path) -> None:
    await _sign_in(auth, server)
    assert stat.S_IMODE((tmp_path / "credentials").stat().st_mode) == 0o700
    assert stat.S_IMODE((tmp_path / "credentials" / "chatgpt.json").stat().st_mode) == 0o600


@pytest.mark.parametrize(
    "tamper",
    [
        lambda q: {**q, "state": "forged"},
        lambda q: {k: v for k, v in q.items() if k != "state"},
    ],
)
async def test_a_callback_with_the_wrong_state_is_refused(auth: ChatGPTAuth, server: FakeOpenAIAuth, tamper) -> None:
    callback = server.approve(auth.start_sign_in(8001))
    with pytest.raises(AuthError, match="expired or wasn't started here"):
        await auth.complete_sign_in(tamper(callback))
    assert server.requests == []  # never reached the token endpoint


async def test_a_callback_works_once(auth: ChatGPTAuth, server: FakeOpenAIAuth) -> None:
    callback = server.approve(auth.start_sign_in(8001))
    await auth.complete_sign_in(callback)
    with pytest.raises(AuthError):
        await auth.complete_sign_in(callback)


async def test_an_old_sign_in_expires(auth: ChatGPTAuth, server: FakeOpenAIAuth, monkeypatch: pytest.MonkeyPatch) -> None:
    callback = server.approve(auth.start_sign_in(8001))
    monkeypatch.setattr(chatgpt, "SIGN_IN_TIMEOUT_SECONDS", -1)
    with pytest.raises(AuthError, match="expired"):
        await auth.complete_sign_in(callback)


async def test_cancelling_at_openai_is_reported_plainly(auth: ChatGPTAuth, server: FakeOpenAIAuth) -> None:
    callback = server.approve(auth.start_sign_in(8001))
    with pytest.raises(AuthError, match="cancelled"):
        await auth.complete_sign_in({"state": callback["state"], "error": "access_denied"})


async def test_registration_without_an_issued_client_id_is_refused(auth: ChatGPTAuth, server: FakeOpenAIAuth) -> None:
    callback = server.approve(auth.start_sign_in(8001))
    del callback["client_id"]
    with pytest.raises(AuthError, match="registering"):
        await auth.complete_sign_in(callback)


async def test_an_identity_token_for_another_sign_in_is_refused(auth: ChatGPTAuth, server: FakeOpenAIAuth, tmp_path: Path) -> None:
    server.nonce_override = "someone-elses-nonce"
    with pytest.raises(AuthError, match="didn't match"):
        await _sign_in(auth, server)
    assert not (tmp_path / "credentials" / "chatgpt.json").exists() or not _stored(tmp_path)["registrations"]


async def test_a_sign_in_without_plan_usage_is_refused(auth: ChatGPTAuth, server: FakeOpenAIAuth, tmp_path: Path) -> None:
    server.grant_plan_scope = False
    with pytest.raises(AuthError, match="isn't enabled"):
        await _sign_in(auth, server)
    assert not _stored(tmp_path)["registrations"]


async def test_a_failed_exchange_reports_the_reason_and_nothing_secret(auth: ChatGPTAuth, server: FakeOpenAIAuth) -> None:
    server.exchange_status = 400
    callback = server.approve(auth.start_sign_in(8001))
    with pytest.raises(AuthError) as caught:
        await auth.complete_sign_in(callback)
    assert str(caught.value) == "Sign-in failed: Bad code."
    assert callback["code"] not in str(caught.value)


# --- tokens --------------------------------------------------------------------


async def test_a_fresh_token_is_returned_without_a_request(auth: ChatGPTAuth, server: FakeOpenAIAuth) -> None:
    await _sign_in(auth, server)
    before = len(server.requests)
    token = await auth.access_token()
    assert chatgpt.jwt_claims(token)["gen"] == 1
    assert len(server.requests) == before


def _expire(tmp_path: Path) -> None:
    path = tmp_path / "credentials" / "chatgpt.json"
    data = json.loads(path.read_text())
    data["registrations"][0]["tokens"]["expires_at"] = int(time.time()) + 60
    path.write_text(json.dumps(data))


async def test_a_token_near_expiry_is_refreshed_and_the_rotation_kept(auth: ChatGPTAuth, server: FakeOpenAIAuth, tmp_path: Path) -> None:
    await _sign_in(auth, server)
    _expire(tmp_path)
    token = await auth.access_token()
    assert chatgpt.jwt_claims(token)["gen"] == 2
    tokens = _stored(tmp_path)["registrations"][0]["tokens"]
    assert tokens["refresh_token"] == "rt-2"
    assert tokens["id_token"]  # kept, though the refresh response omitted it
    assert "chatgpt.tokens.use.direct" in tokens["scope"]


async def test_concurrent_requests_refresh_once(auth: ChatGPTAuth, server: FakeOpenAIAuth, tmp_path: Path) -> None:
    """Two refreshes with the same rotating token would sign the user out."""
    await _sign_in(auth, server)
    _expire(tmp_path)
    tokens = await asyncio.gather(*(auth.access_token() for _ in range(5)))
    refreshes = [form for _, form in server.requests if form.get("grant_type") == "refresh_token"]
    assert len(refreshes) == 1
    assert len(set(tokens)) == 1


@pytest.mark.parametrize("error", ["invalid_grant", "refresh_token_reused", "refresh_token_invalidated"])
async def test_an_ended_sign_in_clears_tokens_but_keeps_the_client(
    auth: ChatGPTAuth, server: FakeOpenAIAuth, tmp_path: Path, error: str
) -> None:
    await _sign_in(auth, server)
    auth.set_active_source("chatgpt")
    _expire(tmp_path)
    server.refresh_failure = (400, error)
    with pytest.raises(SignInRequired):
        await auth.access_token()
    stored = _stored(tmp_path)
    assert stored["registrations"][0]["tokens"] is None
    assert stored["registrations"][0]["client_id"] == ISSUED_CLIENT_ID
    assert stored["active_source"] == "api_key"  # back to the default, not left on nothing


@pytest.mark.parametrize("failure", ["unreachable", "server_error"])
async def test_a_passing_refresh_failure_keeps_the_tokens(
    auth: ChatGPTAuth, server: FakeOpenAIAuth, tmp_path: Path, failure: str
) -> None:
    await _sign_in(auth, server)
    _expire(tmp_path)
    if failure == "unreachable":
        server.refresh_unreachable = True
    else:
        server.refresh_failure = (503, "temporarily_unavailable")
    with pytest.raises(TemporaryAuthError):
        await auth.access_token()
    assert _stored(tmp_path)["registrations"][0]["tokens"]["refresh_token"] == "rt-1"


async def test_using_the_plan_requires_a_sign_in(auth: ChatGPTAuth, server: FakeOpenAIAuth) -> None:
    with pytest.raises(SignInRequired):
        auth.set_active_source("chatgpt")
    await _sign_in(auth, server)
    auth.set_active_source("chatgpt")
    assert auth.using_plan()


# --- sign-out ------------------------------------------------------------------


async def test_sign_out_revokes_then_forgets(auth: ChatGPTAuth, server: FakeOpenAIAuth, tmp_path: Path) -> None:
    await _sign_in(auth, server)
    auth.set_active_source("chatgpt")
    await auth.sign_out()
    assert server.revoked == ["rt-1"]
    [(url, form)] = [(u, f) for u, f in server.requests if u.endswith("/revoke")]
    assert form == {"token": "rt-1", "token_type_hint": "refresh_token", "client_id": ISSUED_CLIENT_ID}
    stored = _stored(tmp_path)
    assert stored["registrations"][0]["tokens"] is None
    assert stored["active_source"] == "api_key"
    with pytest.raises(SignInRequired):
        await auth.access_token()


async def test_a_failed_revocation_keeps_the_tokens_to_retry(auth: ChatGPTAuth, server: FakeOpenAIAuth, tmp_path: Path) -> None:
    await _sign_in(auth, server)
    server.revoke_status = 503
    with pytest.raises(TemporaryAuthError):
        await auth.sign_out()
    assert len(server.revoked) == chatgpt.REVOKE_ATTEMPTS
    assert _stored(tmp_path)["registrations"][0]["tokens"] is not None


# --- secrets never leak ------------------------------------------------------------


async def test_status_and_logs_never_contain_a_token(
    auth: ChatGPTAuth, server: FakeOpenAIAuth, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Every token issued in a sign-in, a refresh and two failures stays out of
    the status the UI gets and out of the logs."""
    caplog.set_level(logging.DEBUG)
    issued: set[str] = set()

    def remember() -> None:
        tokens = _stored(tmp_path)["registrations"][0]["tokens"]
        issued.update(tokens[k] for k in ("access_token", "refresh_token", "id_token"))

    await _sign_in(auth, server)
    remember()
    _expire(tmp_path)
    await auth.access_token()
    remember()
    server.exchange_status = 400
    with pytest.raises(AuthError):
        await _sign_in(auth, server)
    server.refresh_failure = (400, "invalid_grant")
    _expire(tmp_path)
    with pytest.raises(SignInRequired):
        await auth.access_token()

    assert len(issued) == 5  # two access, two refresh, one id token (kept on refresh)
    status = json.dumps(auth.status())
    for token in issued:
        assert token not in status
        assert token not in caplog.text


def test_token_fields_are_kept_out_of_repr() -> None:
    tokens = Tokens(access_token="AAA-secret", refresh_token="RRR-secret", id_token="III-secret", expires_at=1)
    assert "secret" not in repr(tokens)


def test_an_unreadable_credential_file_means_signed_out(tmp_path: Path) -> None:
    (tmp_path / "credentials").mkdir()
    (tmp_path / "credentials" / "chatgpt.json").write_text("{not json")
    auth = ChatGPTAuth(CredentialStore(tmp_path / "credentials"))
    assert auth.status()["signed_in"] is False


async def test_two_processes_sharing_the_file_refresh_once(server: FakeOpenAIAuth, tmp_path: Path) -> None:
    """Separate ChatGPTAuth instances have separate asyncio locks, like two
    processes: only the file lock, and re-reading the file inside it, stops
    the second from refreshing with the token the first just replaced.
    The refresh takes a moment, as a real one does: answered instantly, the
    two would happen to run one after the other and prove nothing."""

    async def slow_refresh(request: httpx.Request) -> httpx.Response:
        if b"grant_type=refresh_token" in request.content:
            await asyncio.sleep(0.05)
        return server.handler(request)

    transport = httpx.MockTransport(slow_refresh)

    def instance() -> ChatGPTAuth:
        return ChatGPTAuth(
            CredentialStore(tmp_path / "credentials"),
            http=lambda: httpx.AsyncClient(transport=transport),
        )

    first, second = instance(), instance()
    await _sign_in(first, server)
    _expire(tmp_path)
    tokens = await asyncio.gather(first.access_token(), second.access_token())
    refreshes = [form for _, form in server.requests if form.get("grant_type") == "refresh_token"]
    assert len(refreshes) == 1
    assert tokens[0] == tokens[1]
