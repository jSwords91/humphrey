"""The environment: the world Humphrey acts on, rebuilt from scratch for every trial.

An environment is only the world: fresh state for one trial, and a way to read the outcome
afterwards. It knows nothing about the agent under test. The agent's tools reach the trial's
world through a small function that takes the environment (and the trial's person) and returns
the tools:

    class Workspace(Environment):
        def setup(self, workdir):          # fresh state for one trial
            self.workspace = copy_seed(workdir)

        def query(self, query):            # read one value from the outcome, for state graders
            ...

    def tools(env, person):                # Humphrey's hands, pointed at this trial's world
        return tools_for_run(person, workspace=env.workspace)

Production builds the same tools with the real workspace and the browser's bridge:
`tools_for_run(WebUiBridge(...))`. So the agent and its tools are identical in both places;
only what the tools touch, and who answers, changes.

E8 [DE]: every trial gets a fresh environment, so no trial can leave state behind for the next.
  Shared state causes correlated failures, which make an agent look worse (or better) than it is.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from app.ui_bridge import UiBridge
from brain.tools import AgentTool


class Environment(ABC):
    # The folder Humphrey's file tools are rooted in for this trial, if any. The runner passes it
    #   to build_harness so the system prompt names the same folder the tools use.
    workspace: Path | None = None

    @abstractmethod
    def setup(self, workdir: Path) -> None:
        """Create fresh state for one trial. `workdir` is an empty folder, deleted afterwards.
        Keep whatever the tools will need (a folder, a fake client) as attributes."""

    def query(self, query: str) -> Any:
        """One value read from the final state. `state` graders compare it with their target."""
        raise NotImplementedError(
            f"{type(self).__name__} has no query(), so it can't run state graders."
        )

    def snapshot(self) -> dict[str, Any]:
        """Anything worth keeping in the transcript, such as the files the agent changed."""
        return {}


# Given a trial's environment and its person, the tools the agent gets.
ToolFactory = Callable[[Any, UiBridge], Sequence[AgentTool]]
