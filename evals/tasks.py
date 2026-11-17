"""Tasks and suites.

A task is one input, who answers if Humphrey asks, the graders that decide pass or fail, and a
reference solution. A suite is a set of tasks, the environment they run in, and the tools to
give Humphrey by default. A tasks file is a suite written in YAML:

    environment: evals.ask_user.env:Workspace       # the world each trial runs in
    tools: evals.ask_user.env:tools                 # the hands, given a trial's world and person
    tasks:
      - id: two_sams_asks_which
        suite: regression
        input: Add Sam to the offsite attendees.
        person: {pick: "O'Neill"}
        graders:
          - {type: tool_calls, name: asked, must_succeed: [ask_user_question]}
        reference_solution:
          - [ask_user_question, {questions: [...]}]

There's no `agent:` line, unlike agentic-evals: Humphrey has no single Agent constant. The brain
under test is Humphrey's own prompt and tools (app/agent.py), with the model and effort given
on the command line (K1).

E1 [DE]: the vocabulary used throughout:
  task        one input plus its graders and reference solution
  trial       one attempt at a task; outputs vary between runs, so each task gets several
  transcript  the full record of a trial: messages, tool calls, cards, tokens
  outcome     the final state of the environment, not what the agent says it did
  grader      code or a model that scores one aspect of a trial
E6 [DE]: two suites. Regression tasks should pass nearly every time and gate every change.
  Capability tasks start with a low pass rate, giving you something to climb. They're reported
  but never block a merge, and graduate to regression once they pass reliably.
E9 [DE]: a reference solution is a list of tool calls that passes every code grader, and a
  reference reply is a reply that passes the reply graders. Together they prove the task can be
  passed and the graders are written correctly. `python -m evals check` replays them with no
  model, for free.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .environment import Environment, ToolFactory
from .graders import Grader, parse_grader
from .person import Person


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    suite: Literal["regression", "capability"]
    input: str
    person: Person | None = None  # who answers if Humphrey asks; no person skips
    graders: list[Grader] = Field(min_length=1)
    reference_solution: list[tuple[str, dict[str, Any]]] = []
    reference_reply: str | None = None  # a correct reply, to test the reply graders offline

    @field_validator("graders", mode="before")
    @classmethod
    def _parse_graders(cls, specs: Any) -> Any:
        return [parse_grader(s) for s in specs] if isinstance(specs, list) else specs


@dataclass(frozen=True)
class Suite:
    environment: type[Environment]
    tasks: Sequence[Task] = field(default_factory=tuple)
    tools: ToolFactory | None = None  # the default tools, given a trial's environment and person

    def __post_init__(self) -> None:
        tasks = tuple(t if isinstance(t, Task) else Task.model_validate(t) for t in self.tasks)
        object.__setattr__(self, "tasks", tasks)
        if not tasks:
            raise ValueError("A suite needs at least one task.")
        ids = [t.id for t in tasks]
        if duplicates := sorted({i for i in ids if ids.count(i) > 1}):
            raise ValueError(f"duplicate task ids: {', '.join(duplicates)}")
        if not (isinstance(self.environment, type) and issubclass(self.environment, Environment)):
            raise ValueError(f"{self.environment!r} is not an Environment subclass")

    def task(self, task_id: str) -> Task:
        return next(t for t in self.tasks if t.id == task_id)


class _SuiteFile(BaseModel):
    """The YAML shape of a suite, checked before anything is imported or run."""

    model_config = ConfigDict(extra="forbid")
    environment: str
    tools: str | None = None
    tasks: list[dict[str, Any]] = Field(min_length=1)


def load_suite(path: str | Path) -> Suite:
    """Read a tasks file. Names in it are imported from the working directory."""
    spec = _SuiteFile.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    # The environment's module is imported before any task is read, so custom graders defined
    #   there are registered in time.
    environment = import_object(spec.environment)
    return Suite(
        environment=environment,
        tasks=spec.tasks,
        tools=import_object(spec.tools) if spec.tools else None,
    )


def import_object(target: str) -> Any:
    """`package.module:name` to the object it names."""
    module, _, name = target.partition(":")
    if not (module and name):
        raise ValueError(f"expected 'package.module:name', got {target!r}")
    return getattr(importlib.import_module(module), name)
