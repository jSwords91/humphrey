"""Humphrey's tools: tau's four coding tools (read, write, edit, bash) from
app/tools/tau_ported.py, rooted in the configured workspace directory.
AgentTool/ToolResult themselves live in brain/tools.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.settings import get_settings
from app.skills import SKILLS_DIR
from app.tools.tau_ported import create_coding_tools
from brain.tools import AgentTool

__all__ = ["TOOLS", "WORKSPACE_DIR", "tool_label"]

WORKSPACE_DIR = Path(get_settings().workspace_dir).resolve()
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

# The file tools stay inside the workspace; read may also open the skills,
# which the prompt lists by absolute path.
TOOLS: list[AgentTool] = create_coding_tools(cwd=WORKSPACE_DIR, read_roots=(SKILLS_DIR,))

_TOOLS_BY_NAME: dict[str, AgentTool] = {tool.name: tool for tool in TOOLS}


def tool_label(name: str, args: dict[str, Any] | None = None) -> str:
    tool = _TOOLS_BY_NAME.get(name)
    if tool is None:
        return name
    return tool.label_for(args)
