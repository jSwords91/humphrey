"""Projects: a named folder in the workspace, and the chats that work in it.

The folder is the project. Every folder under ``workspace/projects/`` is
one, whether Humphrey made it or the person did, and there is no other
record of them. A chat with no project works in the workspace root, the
default project, whose file scans leave ``projects/`` out so that each
project's files show only in its own chats.

A run's tools, prompt and artefact scan are all rooted in its project's
folder (``folder``). Files are still served by their path in the
workspace (``served_path``), so a project needs nothing new to serve them.
"""

from __future__ import annotations

import re
from pathlib import Path

PROJECTS_DIR = "projects"
MAX_NAME_LENGTH = 64


class ProjectError(ValueError):
    """A project can't be made or found; the message says why, for the person."""


class ProjectExists(ProjectError):
    """A project with that folder name is already there."""


def list_projects(root: Path) -> list[str]:
    """Every project's folder name, sorted."""
    try:
        entries = list((root / PROJECTS_DIR).iterdir())
    except OSError:
        return []
    return sorted(e.name for e in entries if e.is_dir() and not e.name.startswith("."))


def folder(root: Path, project: str | None) -> Path:
    """The folder a chat in ``project`` works in. Only a folder that exists
    under ``projects/`` is a project, so no name reaches anywhere else."""
    if project is None:
        return root
    if project not in list_projects(root):
        raise ProjectError(f"There's no project called {project}: no folder {PROJECTS_DIR}/{project}/.")
    return root / PROJECTS_DIR / project


def create(root: Path, title: str) -> str:
    """Make a project's folder from the name the person typed, and return
    the folder's name: lowercase, with hyphens between words."""
    name = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:MAX_NAME_LENGTH].strip("-")
    if not name:
        raise ProjectError("Give the project a name with a letter or a number in it.")
    path = root / PROJECTS_DIR / name
    if path.exists():
        raise ProjectExists(f"A project called {name} already exists.")
    path.mkdir(parents=True)
    return name


def scan_exclusions(project: str | None) -> frozenset[str]:
    """Top-level folders a project's file scans leave out: the default
    project's root holds every other project."""
    return frozenset({PROJECTS_DIR}) if project is None else frozenset()


def served_path(project: str | None, path: str) -> str:
    """A path within a project, as the workspace serves it."""
    return path if project is None else f"{PROJECTS_DIR}/{project}/{path}"
