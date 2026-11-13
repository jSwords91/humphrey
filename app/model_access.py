"""How Humphrey reaches a model: which source, which models, which provider.

The one place that knows both the sign-in state (app/auth) and the providers.
providers/ never imports app/auth: a plan-backed provider is just the OpenAI
provider given the plan's access token as its credential source.

Sources, in order: the offline mock (HUMPHREY_MOCK), then the ChatGPT plan
when someone has signed in and chosen it, then an API key, then Azure. The
API key is the default; using the plan is always an explicit choice.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from app.auth.chatgpt import get_chatgpt_auth
from app.settings import active_provider_kind, get_settings
from brain.provider import ModelProvider
from providers import get_provider
from providers.openai_models import ModelInfo, list_models
from providers.openai_provider import OpenAIProvider

logger = logging.getLogger(__name__)

PLAN_MODELS_TTL_SECONDS = 600
_plan_models: tuple[str, float, list[ModelInfo]] | None = None  # (client id, fetched at, models)


class UnknownModel(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ModelOptions:
    source: str  # "mock" | "chatgpt" | "openai" | "azure_openai" | "unconfigured"
    selectable: bool
    default: str
    models: list[str]
    labels: dict[str, str]  # display names, where the source gives them


def current_source() -> str:
    kind = active_provider_kind(get_settings())
    if kind != "mock" and get_chatgpt_auth().using_plan():
        return "chatgpt"
    return kind


async def plan_models() -> list[ModelInfo]:
    """The signed-in account's models, cached briefly per account (the list
    differs between accounts, and between a plan and an API key)."""
    global _plan_models
    auth = get_chatgpt_auth()
    account = str(auth.status().get("email") or "")
    if _plan_models and _plan_models[0] == account:
        cached_for = time.monotonic() - _plan_models[1]
        if cached_for < PLAN_MODELS_TTL_SECONDS:
            return _plan_models[2]
    models = await list_models(await auth.access_token())
    _plan_models = (account, time.monotonic(), models)
    return models


async def model_options() -> ModelOptions:
    settings = get_settings()
    source = current_source()
    if source == "chatgpt":
        try:
            models = await plan_models()
        except Exception as exc:  # noqa: BLE001 - the picker degrades, the chat still works
            logger.warning("couldn't list the ChatGPT plan's models: %s", exc)
            models = []
        slugs = [m.slug for m in models] or [settings.openai_model]
        default = settings.openai_model if settings.openai_model in slugs else slugs[0]
        return ModelOptions(
            source, True, default, slugs, {m.slug: m.display_name for m in models}
        )
    if source == "openai":
        models = list(settings.selectable_models)
        return ModelOptions(source, True, models[0], models, {})
    fixed = {
        "mock": "mock",
        "azure_openai": settings.azure_openai_deployment,
        "unconfigured": "unset",
    }[source]
    return ModelOptions(source, False, fixed, [fixed], {})


async def build_provider(requested_model: str | None) -> ModelProvider:
    """The provider for the current source. ``requested_model`` must be one
    the source offers; for fixed-model sources it's ignored."""
    options = await model_options()
    model = None
    if requested_model and options.selectable:
        if requested_model not in options.models:
            raise UnknownModel(f"Unknown model: {requested_model}")
        model = requested_model
    if options.source == "chatgpt":
        return OpenAIProvider(
            get_settings(),
            model=model or options.default,
            credential=get_chatgpt_auth().access_token,
            responses_only=True,  # the plan refuses Chat Completions
        )
    return get_provider(model)


def reset_plan_models_for_tests() -> None:
    global _plan_models
    _plan_models = None
