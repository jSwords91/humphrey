# Humphrey

A streaming, tool-calling agent with skills: FastAPI + SSE on the back end, a
plain HTML/JS chat UI on the front. The core agent (messages, loop, harness,
events) is provider-agnostic and knows nothing about the app around it.

## Features

- Streamed replies, tool calls, and reasoning summaries (for reasoning models)
- Steering and follow-up messages while a turn is still running, plus stop
- Skills: markdown instructions the model loads on demand (`skills/`), or
  that you invoke directly with `/skill:<name>`
- Image attachments
- Rendered Mermaid diagrams
- Persistent, auto-named sessions with automatic context compaction
- Runs on OpenAI, with an offline mock that needs no credentials and an
  optional Azure OpenAI adapter

## Running locally

```bash
cp .env.example .env        # then set OPENAI_API_KEY
uv sync
uv run uvicorn app.main:app --reload --port 8001
```

Then open http://localhost:8001. `OPENAI_MODEL` picks the model. To try it
without a key, set `HUMPHREY_MOCK=1` instead.

Tests run offline in mock mode:

```bash
uv sync --group dev
uv run pytest
```

Or with Docker:

```bash
docker compose up --build
```

## Layout

| Path | What lives there |
| --- | --- |
| `brain/` | Portable agent core: messages, loop, harness, typed events, tool protocol |
| `providers/` | Model adapters: OpenAI (Chat Completions and Responses), mock, optional Azure OpenAI |
| `app/` | The application: API, tools, skills, prompt, sessions, storage, SSE rendering |
| `app/tools/` | Tool handlers and the registered tool list (`__init__.py`) |
| `frontend/` | Static chat UI |
| `knowledge/` | Markdown docs the model can list, search, and read |
| `skills/` | One directory per skill, each with a `SKILL.md` |

## Tools

| Tool | Purpose |
| --- | --- |
| `capabilities`, `status` | What the agent can do, and its live config |
| `list_docs`, `read_doc`, `search_docs` | Bundled knowledge docs |
| `read_skill`, `read_skill_asset` | Load a skill, or a file shipped with it |
| `get_weather` | Weather for a named place (Open-Meteo, no API key) |
| `calculate` | Safe arithmetic evaluation |

`get_weather` and `calculate` are small examples to copy when adding your own:
write a handler in `app/tools/` that takes a dict and returns a JSON string,
then register it with `_tool(...)` in `app/tools/__init__.py`.

`app/tools/tau_ported.py` contains read/write/edit/bash tools that are
deliberately not registered. Enabling them needs sandboxing first.

## Storage backend

Sessions persist through one `ObjectStore` (see `app/storage.py`), selected
by `STORAGE_BACKEND`:

| `STORAGE_BACKEND` | Behavior |
| --- | --- |
| `local` (default) | JSON files under `LOCAL_STORAGE_DIR` (default `./data`, gitignored), durable across restarts |
| `memory` | In-memory only, lost on restart (used by the tests) |
| `azure` | Azure Blob (optional adapter, see below); falls back to memory (logged loudly) if misconfigured |
| `auto` | Azure Blob if `AZURE_STORAGE_ACCOUNT_URL` is set, otherwise memory |

Under Docker, `./data` is bind-mounted to `/app/data` so sessions survive
`docker compose down`/`up`. The `status` tool reports which backend actually
ended up active (`storage.active`).

## Optional Azure adapters

Azure OpenAI (as the model provider) and Azure Blob (as session storage) are
kept as adapters, not installed by default:

```bash
uv sync --group azure
```

Azure OpenAI is used only when `OPENAI_API_KEY` is unset and
`AZURE_OPENAI_ENDPOINT` / `AZURE_OPENAI_DEPLOYMENT` are set. It authenticates
with `DefaultAzureCredential`. See `.env.example` for all settings. For
Docker, add `--group azure` to the `uv sync` line in the `Dockerfile`.
