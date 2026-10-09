"""Platform-safe tool handlers — schema-free sync functions, no shell, no FS write.

The concrete handlers for capabilities/status/knowledge-doc/skill
tools. Wrapped into AgentTool instances (with schemas) by
app/tools/__init__.py's TOOLS assembly.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from app.settings import active_provider_kind, get_settings
from app.skills import get_skill
from app.storage import get_object_store

# One level deeper than the old app/tools.py (now app/tools/platform.py),
# hence .parent.parent.parent to still land on the repo-root knowledge/ dir.
KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent.parent / "knowledge"
MAX_DOC_CHARS = 50_000
MAX_DOC_LINES = 2_000
SAFE_DOC_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,120}\.md$")


def _ok(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _err(message: str) -> str:
    return _ok({"error": message})


def _capabilities(_args: dict[str, Any]) -> str:
    # Lazy import: app.tools.__init__ builds TOOLS from this module's own
    # handlers, so importing it back at module load time would cycle -
    # same pattern providers/__init__.py uses for its own self-reference.
    from app.tools import TOOLS

    settings = get_settings()
    return _ok(
        {
            "role": "Humphrey — a general-purpose assistant with tools and skills",
            "streaming": "SSE events (agent loop)",
            "tools": [t.name for t in TOOLS],
            "sessions": f"{_active_storage_kind()}-backed (see STORAGE_BACKEND in status)",
            "provider": active_provider_kind(settings),
            "limits": {
                "shell": False,
                "filesystem_write": False,
                "unrestricted_network": False,
            },
        }
    )


def _host_only(url: str) -> str | None:
    if not url:
        return None
    parsed = urlparse(url)
    return parsed.netloc or parsed.path or None


def _active_storage_kind() -> str:
    """What get_object_store() actually resolved to, not just what
    STORAGE_BACKEND asks for — "auto" can still land on nothing configured.
    """
    from app.storage import AzureBlobObjectStore, LocalFileObjectStore

    store = get_object_store()
    if isinstance(store, AzureBlobObjectStore):
        return "azure"
    if isinstance(store, LocalFileObjectStore):
        return "local"
    return "memory-only"


def _status(_args: dict[str, Any]) -> str:
    settings = get_settings()
    provider = active_provider_kind(settings)
    # Only the active provider's detail block is included — no other provider's
    # fields are ever relevant, so leaving them out (rather than null) is what
    # stops a reader (model or human) from having to explain away unrelated
    # unset fields as if they were a problem.
    model: dict[str, Any] = {"provider": provider, "configured": provider != "unconfigured"}
    if provider == "openai":
        model["openai"] = {"model": settings.openai_model}
    elif provider == "azure_openai":
        model["azure"] = {
            "endpoint_host": _host_only(settings.azure_openai_endpoint),
            "deployment": settings.azure_openai_deployment,
            "api_version": settings.azure_openai_api_version,
        }

    storage_kind = _active_storage_kind()
    storage: dict[str, Any] = {
        "backend_setting": settings.storage_backend,
        "active": storage_kind,
        "container": settings.ai_logs_container,
        "sessions_prefix": settings.sessions_prefix,
    }
    if storage_kind == "azure":
        storage["account_host"] = _host_only(settings.storage_account_url)
    elif storage_kind == "local":
        storage["local_dir"] = settings.local_storage_dir

    return _ok(
        {
            "service": "humphrey",
            "utc_now": datetime.now(timezone.utc).isoformat(),
            "mock_mode": settings.mock_mode,
            "model": model,
            "storage": storage,
            "identity": {
                "AZURE_CLIENT_ID_set": bool(os.environ.get("AZURE_CLIENT_ID")),
            },
            "knowledge_dir_present": KNOWLEDGE_DIR.is_dir(),
        }
    )


def _resolve_doc(path: str) -> Path | str:
    name = Path(path).name
    if not SAFE_DOC_NAME.match(name):
        return "invalid document name (use a simple *.md filename)"
    candidate = (KNOWLEDGE_DIR / name).resolve()
    try:
        candidate.relative_to(KNOWLEDGE_DIR.resolve())
    except ValueError:
        return "path escapes knowledge directory"
    if not candidate.is_file():
        return f"document not found: {name}"
    return candidate


def _list_docs(_args: dict[str, Any]) -> str:
    if not KNOWLEDGE_DIR.is_dir():
        return _err("knowledge directory missing from image")
    docs = sorted(p.name for p in KNOWLEDGE_DIR.glob("*.md") if p.is_file())
    return _ok({"documents": docs, "hint": "Use read_doc with path=<filename>"})


def _read_doc(args: dict[str, Any]) -> str:
    path = str(args.get("path", "")).strip()
    if not path:
        return _err("path is required")

    resolved = _resolve_doc(path)
    if isinstance(resolved, str):
        return _err(resolved)

    text = resolved.read_text(encoding="utf-8")
    lines = text.splitlines()
    offset = int(args.get("offset") or 0)
    limit = int(args.get("limit") or 120)
    limit = max(1, min(limit, 400))

    if offset <= 0:
        start = 0
    else:
        start = offset - 1
    if start >= len(lines):
        return _err(f"offset {offset} is past end of file ({len(lines)} lines)")

    chunk = lines[start : start + limit]
    if len(chunk) > MAX_DOC_LINES:
        chunk = chunk[:MAX_DOC_LINES]
    body = "\n".join(chunk)
    if len(body) > MAX_DOC_CHARS:
        body = body[:MAX_DOC_CHARS] + "\n…[truncated]"

    end_line = start + len(chunk)
    remaining = len(lines) - end_line
    suffix = ""
    if remaining > 0:
        suffix = f"\n\n[{remaining} more lines. Use offset={end_line + 1} to continue.]"

    return _ok(
        {
            "path": resolved.name,
            "start_line": start + 1,
            "end_line": end_line,
            "total_lines": len(lines),
            "content": body + suffix,
        }
    )


def _search_docs(args: dict[str, Any]) -> str:
    query = str(args.get("query", "")).strip()
    if not query:
        return _err("query is required")
    max_hits = int(args.get("max_hits") or 8)
    max_hits = max(1, min(max_hits, 20))
    terms = [t.lower() for t in re.split(r"\s+", query) if t]
    if not terms:
        return _err("query has no searchable terms")

    if not KNOWLEDGE_DIR.is_dir():
        return _err("knowledge directory missing from image")

    hits: list[dict[str, Any]] = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines, start=1):
            lower = line.lower()
            if all(term in lower for term in terms) or any(term in lower for term in terms):
                score = sum(1 for term in terms if term in lower)
                hits.append(
                    {
                        "path": path.name,
                        "line": i,
                        "score": score,
                        "text": line.strip()[:240],
                    }
                )
    hits.sort(key=lambda h: (-int(h["score"]), str(h["path"]), int(h["line"])))
    trimmed = hits[:max_hits]
    return _ok({"query": query, "hits": trimmed, "total_matches": len(hits)})


def _read_skill(args: dict[str, Any]) -> str:
    name = str(args.get("name", "")).strip()
    if not name:
        return _err("name is required")

    skill = get_skill(name)
    if skill is None:
        return _err(f"unknown skill: {name}")
    if skill.disable_model_invocation:
        return _err(f"skill {name} is not available for autonomous invocation")

    return _ok({"name": skill.name, "instructions": skill.content.strip()})


def _resolve_skill_asset(skill_name: str, rel_path: str) -> Path | str:
    """Resolve a path a skill references relative to that skill's own directory.

    Unlike ``_resolve_doc`` (which flattens to a bare filename because
    knowledge/ is flat), skill assets can live in subdirectories — e.g.
    bid-responses/case-studies/case-studies.md — so this checks full path
    containment via ``relative_to`` rather than stripping to a basename.
    An absolute rel_path or a ``..`` escape both fail that check the same way
    (pathlib's ``/`` operator discards the left side entirely when the right
    side is absolute, so the naive-looking ``skill_dir / rel_path`` join is
    only safe *because* of the containment check that follows it).
    """
    skill = get_skill(skill_name)
    if skill is None:
        return f"unknown skill: {skill_name}"
    if not rel_path:
        return "path is required"
    skill_dir = skill.path.parent.resolve()
    candidate = (skill_dir / rel_path).resolve()
    try:
        candidate.relative_to(skill_dir)
    except ValueError:
        return "path escapes the skill's own directory"
    if not candidate.is_file():
        return f"asset not found: {rel_path}"
    return candidate


def _read_skill_asset(args: dict[str, Any]) -> str:
    name = str(args.get("name", "")).strip()
    path = str(args.get("path", "")).strip()
    if not name:
        return _err("name is required")
    if not path:
        return _err("path is required")

    resolved = _resolve_skill_asset(name, path)
    if isinstance(resolved, str):
        return _err(resolved)

    text = resolved.read_text(encoding="utf-8")
    lines = text.splitlines()
    offset = int(args.get("offset") or 0)
    limit = int(args.get("limit") or 200)
    limit = max(1, min(limit, MAX_DOC_LINES))

    start = 0 if offset <= 0 else offset - 1
    if start >= len(lines):
        return _err(f"offset {offset} is past end of file ({len(lines)} lines)")

    chunk = lines[start : start + limit]
    body = "\n".join(chunk)
    if len(body) > MAX_DOC_CHARS:
        body = body[:MAX_DOC_CHARS] + "\n…[truncated]"

    end_line = start + len(chunk)
    remaining = len(lines) - end_line
    suffix = ""
    if remaining > 0:
        suffix = f"\n\n[{remaining} more lines. Use offset={end_line + 1} to continue.]"

    return _ok(
        {
            "skill": name,
            "path": path,
            "start_line": start + 1,
            "end_line": end_line,
            "total_lines": len(lines),
            "content": body + suffix,
        }
    )
