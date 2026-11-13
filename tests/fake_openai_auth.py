"""A stand-in for OpenAI's auth service (auth.openai.com), for tests.

It checks PKCE, issues a client id at first registration, rotates refresh
tokens, refuses a reused one, and records every request. The parameters
match a live sign-in. Fake tokens are built at runtime, so nothing here
looks like a real credential to tests/test_no_secrets.py.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from urllib.parse import parse_qs, urlsplit

import httpx

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
