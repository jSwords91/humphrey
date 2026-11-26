"""Humphrey's tools: tau's four coding tools (read, write, edit, bash) from
app/tools/tau_ported.py, rooted in the configured workspace directory, plus
ask_user_question unless HUMPHREY_ASK_USER_QUESTION turns it off, and
start_task and get_task_result (app/tools/tasks.py).
AgentTool/ToolResult themselves live in brain/tools.py.

ask_user_question waits on the run's UI bridge and the task tools call the
run's task starter, so each run gets its own list from ``tools_for_run``.
``TOOLS`` is the same list bound to a bridge with no UI and no tasks: it's
what the prompt, labels and context estimates read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.settings import get_settings
from app.skills import SKILLS_DIR
from app.tools.ask_user_question import create_ask_user_question_tool
from app.tools.tasks import NoTasks, TaskStarter, create_task_tools
from app.tools.tau_ported import create_coding_tools
from app.ui_bridge import NullUiBridge, UiBridge
from brain.tools import AgentTool

__all__ = ["TOOLS", "WORKSPACE_DIR", "tool_label", "tools_for_run"]

WORKSPACE_DIR = Path(get_settings().workspace_dir).resolve()
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

# The file tools stay inside the workspace; read may also open the skills,
# which the prompt lists by absolute path.
CODING_TOOLS: list[AgentTool] = create_coding_tools(cwd=WORKSPACE_DIR, read_roots=(SKILLS_DIR,))


def tools_for_run(
    bridge: UiBridge, workspace: Path | None = None, tasks: TaskStarter | None = None
) -> list[AgentTool]:
    """The tools for one run, with ask_user_question bound to its bridge and
    the task tools to its starter (none: they refuse, and say why).

    ``workspace`` roots the file tools somewhere other than WORKSPACE_DIR;
    evals give each trial its own folder this way.
    """
    if workspace is None:
        tools = list(CODING_TOOLS)
    else:
        tools = create_coding_tools(cwd=workspace, read_roots=(SKILLS_DIR,))
    if get_settings().ask_user_question:
        tools.append(create_ask_user_question_tool(bridge))
    tools.extend(create_task_tools(tasks or NoTasks()))
    return tools


TOOLS: list[AgentTool] = tools_for_run(NullUiBridge())

_TOOLS_BY_NAME: dict[str, AgentTool] = {tool.name: tool for tool in TOOLS}


def tool_label(name: str, args: dict[str, Any] | None = None) -> str:
    tool = _TOOLS_BY_NAME.get(name)
    if tool is None:
        return name
    return tool.label_for(args)
