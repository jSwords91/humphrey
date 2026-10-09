# Humphrey (this app)

A streaming, tool-calling agent with skills.

## Architecture

- FastAPI serves the static UI and `POST /api/agent` (SSE)
- `brain/` is the portable core: messages, the agent loop, the harness, typed events, and the tool protocol. It knows nothing about this app.
- `providers/` adapts model APIs to the brain: OpenAI (Chat Completions or Responses), an offline mock, or the optional Azure OpenAI adapter
- `app/` wires it together: tools, skills, system prompt, sessions, storage, and SSE rendering
- Sessions persist the full transcript (including tool results) through one storage backend: local JSON files (default), memory, or the optional Azure Blob adapter

## Tools

| Tool | Use |
|---|---|
| `capabilities` | What the agent can do |
| `status` | Runtime config (provider, storage, mock mode) |
| `list_docs` / `read_doc` / `search_docs` | Knowledge files bundled under `knowledge/` |
| `read_skill` / `read_skill_asset` | Load a skill's instructions or a file shipped with it |
| `get_weather` | Current weather and short forecast for a place (Open-Meteo) |
| `calculate` | Exact arithmetic |

New tools go in `app/tools/` and are registered in `app/tools/__init__.py`.

## Skills

A skill is a directory under `skills/` containing a `SKILL.md` with a `description` in its front matter. The model sees each skill's name and description, and loads the full instructions with `read_skill` when a task matches. Users can also invoke one directly with `/skill:<name>`.

## UI

Collapsible tool cards, streamed reasoning for reasoning models, a stop control, steering while a turn runs, image attachments, and rendered Mermaid diagrams.

## Intentionally not included

- `bash` / shell
- Arbitrary `write` / `edit` on the server filesystem

`app/tools/tau_ported.py` has working versions of these, deliberately not registered. Enabling them needs sandboxing first.
