"""Task-based evals for Humphrey. Read in this order:

    environment.py  the world a trial runs in, fresh each time
    person.py       who answers Humphrey's questions in a trial
    tasks.py        tasks and suites, from a YAML file or from code
    graders.py      the built-in graders, and how to add your own
    metrics.py      pass@k, pass^k and the CI gate
    transcript.py   reading a finished run: reply, tool calls, tokens
    runner.py       evaluate(): trials, transcripts and infrastructure errors
    results.py      Results: the table, failures, comparisons and transcripts
    report.py       what those print

Nothing in app/, brain/ or providers/ imports this package (tests/test_layering.py).
"""

from .environment import Environment, ToolFactory
from .graders import Grade, Grader, Trial, register
from .person import Person, ScriptedPerson
from .results import Results
from .runner import evaluate
from .tasks import Suite, Task, load_suite

__all__ = [
    "Environment",
    "Grade",
    "Grader",
    "Person",
    "Results",
    "ScriptedPerson",
    "Suite",
    "Task",
    "ToolFactory",
    "Trial",
    "evaluate",
    "load_suite",
    "register",
]
