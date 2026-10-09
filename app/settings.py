"""Runtime configuration — single place for env-derived settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# The project's own .env, at the repo root.
# load_dotenv() is a no-op if the path doesn't exist and never overrides an
# already-set variable, so this is safe with no .env present too.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}


StorageBackend = Literal["auto", "azure", "local", "memory"]
_STORAGE_BACKENDS = {"auto", "azure", "local", "memory"}


def _storage_backend() -> StorageBackend:
    raw = os.environ.get("STORAGE_BACKEND", "local").strip().lower()
    return raw if raw in _STORAGE_BACKENDS else "local"  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable process settings."""

    mock_mode: bool
    azure_openai_endpoint: str
    azure_openai_deployment: str
    azure_openai_api_version: str
    openai_api_key: str
    openai_model: str
    reasoning_effort: str | None
    storage_account_url: str
    ai_logs_container: str
    sessions_prefix: str
    storage_backend: StorageBackend
    local_storage_dir: str
    context_window_tokens: int

    @property
    def openai_configured(self) -> bool:
        return bool(self.azure_openai_endpoint and self.azure_openai_deployment)

    @property
    def openai_api_configured(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def azure_storage_configured(self) -> bool:
        return bool(self.storage_account_url)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        mock_mode=_flag("HUMPHREY_MOCK"),
        azure_openai_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/"),
        azure_openai_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", ""),
        azure_openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        openai_model=os.environ.get("OPENAI_MODEL", "gpt-5.6-sol"),
        # Defaults to "medium" rather than leaving this unset: an unset
        # value doesn't just mean "the model picks its own effort level" -
        # build_responses_payload() only adds the `reasoning` field at all
        # when this is truthy, so leaving it unset also means no
        # `summary: "auto"` is ever requested, and no reasoning text streams
        # to the UI at all (not a rendering bug - nothing was ever asked
        # for). Set REASONING_EFFORT="" explicitly to opt back out of this.
        reasoning_effort=os.environ.get("REASONING_EFFORT", "medium") or None,
        storage_account_url=os.environ.get("AZURE_STORAGE_ACCOUNT_URL", "").rstrip("/"),
        ai_logs_container=os.environ.get("AZURE_STORAGE_AI_LOGS_CONTAINER", "ai-logs"),
        sessions_prefix=os.environ.get("HUMPHREY_SESSIONS_PREFIX", "sessions"),
        storage_backend=_storage_backend(),
        # Absolute by default so it doesn't depend on the process's cwd (the
        # Docker image runs uvicorn from /app, not this file's directory) —
        # data/ is gitignored.
        local_storage_dir=os.environ.get("LOCAL_STORAGE_DIR", "").strip()
        or str(Path(__file__).resolve().parent.parent / "data"),
        # One configured constant rather than tau's per-model catalog lookup
        # (RuntimeModelCatalog) - deliberately simpler, see context_window.py's
        # module docstring. 128k covers every model this app currently uses.
        context_window_tokens=int(os.environ.get("MODEL_CONTEXT_WINDOW_TOKENS", "128000")),
    )


def reset_settings_for_tests() -> None:
    """Drop cached settings (tests only)."""
    get_settings.cache_clear()


ProviderKind = Literal["mock", "openai", "azure_openai", "unconfigured"]


def active_provider_kind(settings: Settings) -> ProviderKind:
    """Single source of truth for provider precedence: mock > OpenAI > Azure OpenAI.

    Lives here (not in ``app.providers``) so callers that only need to
    *describe* the active provider — healthz, the status tool, both of
    which sit closer to app.tools than app.providers — can import it without
    pulling in the provider classes themselves, which depend on app.tools and
    would create a cycle.
    """
    if settings.mock_mode:
        return "mock"
    # OpenAI is the primary provider; Azure OpenAI is an optional adapter
    # used only when no OPENAI_API_KEY is set.
    if settings.openai_api_configured:
        return "openai"
    if settings.openai_configured:
        return "azure_openai"
    return "unconfigured"
