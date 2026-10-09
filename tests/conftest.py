"""Shared fixtures — always run under HUMPHREY_MOCK=1."""

from __future__ import annotations

import os
import tempfile

import pytest
from httpx import ASGITransport, AsyncClient

import app.settings  # noqa: F401 — import triggers the repo-root .env load;
# must happen before scrubbing env vars below, or a later first import of
# app.settings (from elsewhere) would reintroduce them via dotenv.

# Must be set before app.settings is imported in each test process.
os.environ["HUMPHREY_MOCK"] = "1"
os.environ["STORAGE_BACKEND"] = "memory"
# The tools act on files and run commands; keep that inside a throwaway dir.
os.environ["HUMPHREY_WORKSPACE"] = tempfile.mkdtemp(prefix="humphrey-test-workspace-")
os.environ.pop("AZURE_STORAGE_ACCOUNT_URL", None)
os.environ.pop("AZURE_OPENAI_ENDPOINT", None)
os.environ.pop("AZURE_OPENAI_DEPLOYMENT", None)
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("OPENAI_MODELS", None)


@pytest.fixture(autouse=True)
def _reset_singletons() -> None:
    from app.runs import clear_active_for_tests
    from app.settings import reset_settings_for_tests
    from app.sessions import reset_session_store_for_tests
    from app.storage import reset_object_store_for_tests

    reset_settings_for_tests()
    reset_object_store_for_tests()
    reset_session_store_for_tests()
    clear_active_for_tests()
    yield
    reset_settings_for_tests()
    reset_object_store_for_tests()
    reset_session_store_for_tests()
    clear_active_for_tests()


@pytest.fixture
async def client() -> AsyncClient:
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
