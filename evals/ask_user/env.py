"""The world for the ask_user_question evals: a copy of one seeded workspace per trial.

Every task shares the same workspace in seed/, the way a real person's workspace holds several
projects at once, and each task has its own records in it (a proposal, a team list, a folder of
reports). The clutter is on purpose: a real workspace has near-duplicate names and unrelated
files, and that's where real ambiguity comes from.

Queries, for `state` graders, return a count. Paths are relative to the workspace, with forward
slashes, and both patterns are regular expressions:

    files:<path regex>                    how many files have a matching path
    grep:<path regex>:<content regex>     how many of those files contain a match (any case)

`unchanged` is a custom grader: files matching a path pattern are exactly as seeded, none added,
removed or edited.
"""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Literal

from app.tools import tools_for_run
from app.ui_bridge import UiBridge
from brain.tools import AgentTool
from evals import Environment, Grade, Grader, Trial, register

SEED = Path(__file__).parent / "seed"
SEED_TIME = datetime(2026, 11, 13, 9, 0, tzinfo=timezone.utc).timestamp()


def read_tree(root: Path) -> dict[str, str]:
    """Every file under root, by relative path, as text."""
    return {
        p.relative_to(root).as_posix(): p.read_text(encoding="utf-8", errors="replace")
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


class Workspace(Environment):
    def setup(self, workdir: Path) -> None:
        self.workspace = workdir / "workspace"
        shutil.copytree(SEED, self.workspace)
        # Every file gets the same time, so "the newest" can't be read off whatever order the
        #   seed happened to be written or checked out in. A first live run did exactly that.
        for path in self.workspace.rglob("*"):
            os.utime(path, (SEED_TIME, SEED_TIME))

    def query(self, query: str) -> Any:
        kind, _, rest = query.partition(":")
        files = read_tree(self.workspace)
        if kind == "files":
            return sum(bool(re.search(rest, path)) for path in files)
        if kind == "grep":
            path_pattern, _, content_pattern = rest.partition(":")
            return sum(
                bool(re.search(path_pattern, path) and re.search(content_pattern, text, re.I))
                for path, text in files.items()
            )
        raise ValueError(
            f"unknown query {query!r}: use files:<path regex> or grep:<path regex>:<regex>"
        )

    def snapshot(self) -> dict[str, Any]:
        """What the agent changed in the workspace, for reading transcripts."""
        before, after = read_tree(SEED), read_tree(self.workspace)
        return {
            "added": {p: t[:2000] for p, t in after.items() if p not in before},
            "changed": {p: t[:2000] for p, t in after.items() if p in before and before[p] != t},
            "removed": sorted(p for p in before if p not in after),
        }


def tools(env: Workspace, person: UiBridge) -> Sequence[AgentTool]:
    """Humphrey's own tools, rooted in this trial's workspace and answered by its person."""
    return tools_for_run(person, workspace=env.workspace)


@register
class Unchanged(Grader):
    """E3: "a question changes nothing" is an outcome, so it's read from the workspace."""

    type: Literal["unchanged"] = "unchanged"
    name: str
    paths: str  # a regular expression over relative paths
    reads_reply: ClassVar[bool] = False

    async def grade(self, trial: Trial) -> list[Grade]:
        def matching(tree: dict[str, str]) -> dict[str, str]:
            return {p: t for p, t in tree.items() if re.search(self.paths, p)}

        before, after = matching(read_tree(SEED)), matching(read_tree(trial.env.workspace))
        problems = (
            [f"removed {p}" for p in before if p not in after]
            + [f"added {p}" for p in after if p not in before]
            + [f"changed {p}" for p in before if p in after and before[p] != after[p]]
        )
        return [Grade(self.name, not problems, "; ".join(problems))]
