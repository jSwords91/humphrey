"""Provider factory: mock, OpenAI (primary), or the optional Azure OpenAI adapter.

Precedence itself lives in ``app.settings.active_provider_kind``. This is
the one deliberate exception to this package's otherwise brain-only
dependency (providers/* imports nothing else from app/) — mirrors how
tau_ai has its own tiny env.py rather than depending on tau_coding, except
we share this app's single Settings/get_settings() singleton rather than
duplicating env-var reading, since that singleton (and its
reset_settings_for_tests() test-isolation hook) is already load-bearing
everywhere else in this app. Splitting it apart is a separate, larger
decision than this reorg.

The three provider-class imports in ``get_provider()`` are deliberately
lazy (inside the function, not at module top). azure.py and
openai_provider.py both import sibling submodules of this same package
(``providers.openai_adapter`` / ``providers.openai_responses``) — an eager
top-level import of azure.py here would require this package's own
``__init__`` to be considered initialized while it's still in the middle of
running that very import, a self-referential cycle. Lazy imports sidestep it.
"""

from __future__ import annotations

from app.settings import active_provider_kind, get_settings
from brain.provider import ModelChunk, ModelProvider, ToolCallDelta

__all__ = [
    "ModelChunk",
    "ModelProvider",
    "ToolCallDelta",
    "active_provider_kind",
    "get_provider",
]


def get_provider(model: str | None = None) -> ModelProvider:
    """``model`` overrides the default OpenAI model; callers validate it
    against ``Settings.selectable_models``. Other providers ignore it."""
    settings = get_settings()
    kind = active_provider_kind(settings)
    if kind == "mock":
        from providers.mock import MockProvider

        return MockProvider()
    if kind == "openai":
        from providers.openai_provider import OpenAIProvider

        return OpenAIProvider(settings, model=model)
    try:
        from providers.azure import AzureOpenAIProvider
    except ImportError as exc:
        raise RuntimeError(
            "Azure OpenAI is configured but the Azure SDK isn't installed. "
            "Run `uv sync --group azure`, or set OPENAI_API_KEY to use OpenAI."
        ) from exc

    return AzureOpenAIProvider(settings)
