"""Sign in with ChatGPT: use a ChatGPT plan instead of an API key.

This is OpenAI's documented flow for open-source, locally run apps
(developers.openai.com/siwc, "token sharing for open source", a preview).
It is not the Codex CLI's sign-in: this app registers its own client, and
requests go to the normal https://api.openai.com/v1/responses endpoint.

  - Sign-in is OAuth 2.0 authorization code with PKCE, state and an OIDC
    nonce. The first sign-in registers this app (``dynamic_agent_client``)
    and OpenAI returns the client id issued to it (``oaiapp_...``) on the
    callback; later sign-ins reuse that id.
  - The callback must be http://127.0.0.1:<port>/<path>: only the port may
    change between sign-ins. Humphrey serves it from its own port, so no
    extra listener is needed (Docker already publishes that port).
  - Access tokens last an hour. Refresh tokens last 30 days and are replaced
    on every refresh, so refreshes are serialized (an asyncio lock in this
    process, a file lock across processes): two refreshes racing with the
    same token would sign the user out.
  - Plan usage is only granted with the ``chatgpt.tokens.use.direct`` scope;
    a sign-in without it is refused here rather than failing at first use.

Nothing in this module logs or raises a token, code or response body: OAuth
failures carry only the server's ``error`` code and description.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import logging
import secrets
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

import httpx

from app.auth.store import CredentialFile, CredentialStore, Registration, Source, Tokens

logger = logging.getLogger(__name__)

AUTHORIZE_URL = "https://auth.openai.com/api/accounts/authorize"
TOKEN_URL = "https://auth.openai.com/api/accounts/oauth/token"
OPENID_CONFIGURATION_URL = "https://auth.openai.com/.well-known/openid-configuration"
RESOURCE = "https://api.openai.com/v1"
PLAN_SCOPE = "chatgpt.tokens.use.direct"
SCOPE = f"openid profile email offline_access resource.invoke {PLAN_SCOPE}"
REGISTRATION_CLIENT_ID = "dynamic_agent_client"  # first sign-in only, never saved
APP_NAME = "Humphrey"
CALLBACK_PATH = "/api/auth/chatgpt/callback"
MANAGE_USAGE_URL = "https://chatgpt.com/settings/usage"
SIGNED_OUT = "Your ChatGPT sign-in has ended. Sign in again, or switch to an API key."

SIGN_IN_TIMEOUT_SECONDS = 600
REFRESH_MARGIN_SECONDS = 300  # refresh this long before the access token expires
REVOKE_ATTEMPTS = 3
REVOKE_BACKOFF_SECONDS = 0.5

# Refresh failures that mean the sign-in is over (OpenAI's errors-and-recovery
# docs); anything else is treated as passing, and the tokens are kept.
TERMINAL_REFRESH_ERRORS = frozenset(
    {
        "invalid_grant",
        "invalid_refresh_token",
        "token_expired",
        "refresh_token_expired",
        "refresh_token_invalidated",
        "refresh_token_reused",
        "invalid_client",
    }
)


class AuthError(RuntimeError):
    """A sign-in problem, worded for the person using Humphrey."""


class SignInRequired(AuthError):
    """No usable tokens: the fix is to sign in again, never a silent retry."""


class TemporaryAuthError(AuthError):
    """The sign-in service couldn't be reached or failed for now; tokens kept."""


def redirect_uri(port: int) -> str:
    # 127.0.0.1, never localhost: OpenAI matches it exactly.
    return f"http://127.0.0.1:{port}{CALLBACK_PATH}"


@dataclass(slots=True)
class PendingSignIn:
    """One sign-in in progress, between opening the browser and the callback."""

    state: str
    redirect_uri: str
    client_id: str | None  # None until this app has been registered
    verifier: str = field(repr=False)
    nonce: str = field(repr=False)
    created_at: float = field(default_factory=time.monotonic)

    def expired(self) -> bool:
        return time.monotonic() - self.created_at > SIGN_IN_TIMEOUT_SECONDS


def authorize_url(pending: PendingSignIn, host_id: str) -> str:
    challenge = base64.urlsafe_b64encode(hashlib.sha256(pending.verifier.encode()).digest())
    params = {
        "client_id": pending.client_id or REGISTRATION_CLIENT_ID,
        "ext_agent_host_id": host_id,
        "response_type": "code",
        "redirect_uri": pending.redirect_uri,
        "scope": SCOPE,
        "resource": RESOURCE,
        "state": pending.state,
        "nonce": pending.nonce,
        "code_challenge": challenge.rstrip(b"=").decode("ascii"),
        "code_challenge_method": "S256",
    }
    if pending.client_id is None:
        params["agent_name_hint"] = APP_NAME  # display name, first registration only
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


def jwt_claims(token: str) -> dict[str, Any]:
    """A JWT's claims, unverified. Used only for the nonce check and the
    account's display details; OpenAI verifies the tokens it receives."""
    try:
        payload = token.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    except (IndexError, ValueError):
        return {}
    return claims if isinstance(claims, dict) else {}


def _oauth_error(response: httpx.Response) -> tuple[str | None, str | None]:
    try:
        body = response.json()
    except ValueError:
        return None, None
    if not isinstance(body, dict):
        return None, None
    error, description = body.get("error"), body.get("error_description")
    return (
        error if isinstance(error, str) else None,
        description if isinstance(description, str) else None,
    )


def _tokens_from(body: Mapping[str, Any], previous: Tokens | None = None) -> Tokens:
    """Tokens from a token response, keeping values a refresh may omit."""
    access = body.get("access_token")
    if not isinstance(access, str) or not access:
        raise AuthError("Sign-in failed: OpenAI returned no access token.")
    refresh = body.get("refresh_token") or (previous.refresh_token if previous else None)
    id_token = body.get("id_token") or (previous.id_token if previous else None)
    if not refresh or not id_token:
        raise AuthError("Sign-in failed: OpenAI's response was missing a token.")
    scope = body.get("scope")
    return Tokens(
        access_token=access,
        refresh_token=str(refresh),
        id_token=str(id_token),
        expires_at=int(time.time()) + int(body.get("expires_in") or 0),
        scope=(
            tuple(scope.split())
            if isinstance(scope, str)
            else (previous.scope if previous else ())
        ),
    )


class ChatGPTAuth:
    """Sign-in, fresh access tokens, sign-out, and which source is active.

    ``http`` is injectable so tests can stand in for OpenAI's auth service.
    """

    def __init__(
        self,
        store: CredentialStore,
        *,
        http: Callable[[], httpx.AsyncClient] = lambda: httpx.AsyncClient(timeout=30),
    ) -> None:
        self._store = store
        self._http = http
        self._lock = asyncio.Lock()
        self._pending: PendingSignIn | None = None

    # -- status -------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """What the UI may know. Never a token."""
        credentials = self._store.load()
        registration = credentials.active_registration()
        signed_in = bool(registration and registration.tokens)
        return {
            "signed_in": signed_in,
            "email": registration.email if registration else None,
            "active_source": credentials.active_source,
            "using_plan": signed_in and credentials.active_source == "chatgpt",
            "plan_notice_seen": bool(registration and registration.plan_notice_seen),
            "manage_usage_url": MANAGE_USAGE_URL,
        }

    def using_plan(self) -> bool:
        return bool(self.status()["using_plan"])

    def set_active_source(self, source: Source) -> None:
        credentials = self._store.load()
        if source == "chatgpt":
            registration = credentials.active_registration()
            if not (registration and registration.tokens):
                raise SignInRequired("Sign in with ChatGPT before using your plan.")
        credentials.active_source = source
        self._store.save(credentials)

    def mark_plan_notice_seen(self) -> None:
        credentials = self._store.load()
        registration = credentials.active_registration()
        if registration is not None:
            registration.plan_notice_seen = True
            self._store.save(credentials)

    # -- sign-in --------------------------------------------------------------

    def start_sign_in(self, port: int) -> str:
        """Begin a sign-in and return the URL to open. A new one replaces any
        sign-in still waiting (its browser tab was probably closed)."""
        credentials = self._store.load()
        self._store.save(credentials)  # persists the host id before first use
        registration = credentials.active_registration()
        self._pending = PendingSignIn(
            state=secrets.token_urlsafe(32),
            redirect_uri=redirect_uri(port),
            client_id=registration.client_id if registration else None,
            verifier=secrets.token_urlsafe(64),
            nonce=secrets.token_urlsafe(32),
        )
        return authorize_url(self._pending, credentials.host_id)

    async def complete_sign_in(self, query: Mapping[str, str]) -> str:
        """Finish the sign-in the browser came back from. Returns the email."""
        pending = self._pending
        state = query.get("state") or ""
        if pending is None or pending.expired() or not secrets.compare_digest(state, pending.state):
            raise AuthError("This sign-in has expired or wasn't started here. Start it again.")
        self._pending = None  # one use, whatever happens next
        if query.get("error"):
            if query["error"] == "access_denied":
                raise AuthError("Sign-in was cancelled.")
            raise AuthError(f"Sign-in failed: {query['error']}.")
        client_id = query.get("client_id") or pending.client_id
        if not client_id or client_id == REGISTRATION_CLIENT_ID:
            raise AuthError("Sign-in didn't finish registering this app. Start it again.")
        if not query.get("code"):
            raise AuthError("Sign-in didn't return an authorization code. Start it again.")

        async with self._http() as http:
            response = await http.post(
                TOKEN_URL,
                data={
                    "grant_type": "authorization_code",
                    "client_id": client_id,
                    "code": query["code"],
                    "code_verifier": pending.verifier,
                    "redirect_uri": pending.redirect_uri,
                    "resource": RESOURCE,
                },
                headers={"Accept": "application/json"},
            )
        if response.status_code != 200:
            error, description = _oauth_error(response)
            logger.warning(
                "ChatGPT sign-in: token exchange failed (%s, %s)", response.status_code, error
            )
            reason = (description or error or f"HTTP {response.status_code}").rstrip(".")
            raise AuthError(f"Sign-in failed: {reason}.")

        tokens = _tokens_from(response.json())
        claims = jwt_claims(tokens.id_token)
        if not secrets.compare_digest(str(claims.get("nonce", "")), pending.nonce):
            raise AuthError("Sign-in failed: the identity token didn't match this sign-in.")
        if PLAN_SCOPE not in tokens.scope:
            raise AuthError(
                "You're signed in, but using your ChatGPT plan in other apps isn't enabled "
                "for this account. You can keep using an API key."
            )

        async with self._lock, self._file_lock():
            credentials = self._store.load()
            existing = next((r for r in credentials.registrations if r.client_id == client_id), None)
            credentials.upsert(
                Registration(
                    client_id=client_id,
                    subject=str(claims.get("sub") or ""),
                    email=claims.get("email") if isinstance(claims.get("email"), str) else None,
                    tokens=tokens,
                    plan_notice_seen=bool(existing and existing.plan_notice_seen),
                )
            )
            self._store.save(credentials)
        return str(claims.get("email") or "")

    # -- tokens ---------------------------------------------------------------

    async def access_token(self) -> str:
        """A live access token for the active account, refreshed if near expiry.

        This is the credential source the OpenAI provider calls per request.
        """
        async with self._lock, self._file_lock():
            # Read inside the lock: another process may have just refreshed.
            credentials = self._store.load()
            registration = credentials.active_registration()
            if registration is None or registration.tokens is None:
                raise SignInRequired(SIGNED_OUT)
            if registration.tokens.expires_at - time.time() > REFRESH_MARGIN_SECONDS:
                return registration.tokens.access_token
            registration.tokens = await self._refresh(registration)
            self._store.save(credentials)
            return registration.tokens.access_token

    async def _refresh(self, registration: Registration) -> Tokens:
        assert registration.tokens is not None
        try:
            async with self._http() as http:
                response = await http.post(
                    TOKEN_URL,
                    data={
                        "grant_type": "refresh_token",
                        "client_id": registration.client_id,
                        "refresh_token": registration.tokens.refresh_token,
                        "resource": RESOURCE,
                    },
                    headers={"Accept": "application/json"},
                )
        except httpx.HTTPError as exc:
            raise TemporaryAuthError(
                "Couldn't reach ChatGPT to renew your sign-in. Try again shortly."
            ) from exc
        if response.status_code == 200:
            return _tokens_from(response.json(), previous=registration.tokens)
        error, _ = _oauth_error(response)
        logger.warning("ChatGPT sign-in: refresh failed (%s, %s)", response.status_code, error)
        if error in TERMINAL_REFRESH_ERRORS:
            self._clear_tokens(registration.client_id)
            raise SignInRequired(SIGNED_OUT)
        raise TemporaryAuthError("ChatGPT couldn't renew your sign-in just now. Try again shortly.")

    def _clear_tokens(self, client_id: str) -> None:
        """Forget an account's tokens but keep its registration, so signing
        in again reuses the same client id."""
        credentials = self._store.load()
        for registration in credentials.registrations:
            if registration.client_id == client_id:
                registration.tokens = None
        active = credentials.active_registration()
        if active is None or active.tokens is None:
            credentials.active_source = "api_key"  # never left pointing at nothing
        self._store.save(credentials)

    # -- sign-out -------------------------------------------------------------

    async def sign_out(self) -> None:
        """Revoke the refresh token at OpenAI, then forget the tokens. If OpenAI
        can't be reached, the tokens are kept so sign-out can be retried."""
        async with self._lock, self._file_lock():
            credentials = self._store.load()
            registration = credentials.active_registration()
            if registration is None or registration.tokens is None:
                return
            await self._revoke(registration)
            self._clear_tokens(registration.client_id)

    async def _revoke(self, registration: Registration) -> None:
        assert registration.tokens is not None
        last_error: str = "unknown"
        for attempt in range(REVOKE_ATTEMPTS):
            try:
                async with self._http() as http:
                    endpoint = (await http.get(OPENID_CONFIGURATION_URL)).json()["revocation_endpoint"]
                    response = await http.post(
                        endpoint,
                        data={
                            "token": registration.tokens.refresh_token,
                            "token_type_hint": "refresh_token",
                            "client_id": registration.client_id,
                        },
                    )
                if response.status_code < 500:
                    return  # 200, including for a token that was already invalid
                last_error = f"HTTP {response.status_code}"
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                last_error = type(exc).__name__
            await asyncio.sleep(REVOKE_BACKOFF_SECONDS * (attempt + 1))
        logger.warning("ChatGPT sign-out: revocation failed (%s)", last_error)
        raise TemporaryAuthError("Couldn't reach ChatGPT to sign out. Try again shortly.")

    # -- cross-process serialization -------------------------------------------

    @contextlib.asynccontextmanager
    async def _file_lock(self) -> Any:
        """Hold an exclusive lock on the credential file's lock file, so two
        Humphrey processes never refresh the same rotating token at once."""
        self._store.ensure_directory()
        with open(self._store.lock_path, "a+") as handle:  # noqa: ASYNC230 - tiny, local
            await asyncio.to_thread(_lock_exclusive, handle)
            try:
                yield
            finally:
                _unlock(handle)


def _lock_exclusive(handle: Any) -> None:
    with contextlib.suppress(ImportError):
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _unlock(handle: Any) -> None:
    with contextlib.suppress(ImportError):
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


_instance: ChatGPTAuth | None = None


def get_chatgpt_auth() -> ChatGPTAuth:
    """The process's one ChatGPTAuth: its lock is what serializes refreshes
    within this process, so there must be only one."""
    global _instance
    if _instance is None:
        from pathlib import Path

        from app.settings import get_settings

        _instance = ChatGPTAuth(CredentialStore(Path(get_settings().credentials_dir)))
    return _instance


def reset_chatgpt_auth_for_tests() -> None:
    global _instance
    _instance = None
