"""The eval harness: run every task several times, each in a fresh environment, and grade it.

    results = await evaluate(suite, provider=lambda: get_provider("gpt-6.1-sol"), trials=3)
    results.table()

The model under test is an argument, not part of the suite, so the same tasks can be run
against two models or two effort levels and the results compared task by task. Everything else
is Humphrey's own: build_harness (its prompt, skills and turn limit) and tools_for_run (its
tools), so production and evals share every line of the agent's code. The pieces are:

    evaluate    many trials, a few at a time, summarised into Results
    run_trial   one attempt at one task: fresh environment, person, harness, graders, transcript
    replay      a task's reference solution with no model, through the code graders (E9)
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import tempfile
import time
from collections.abc import Callable, Collection, Sequence
from pathlib import Path
from typing import Any, Literal

import openai

from app.agent import build_harness
from brain.provider import ModelProvider

from . import transcript
from .environment import ToolFactory
from .graders import Grade, Trial
from .metrics import pass_at_k, pass_hat_k
from .person import ScriptedPerson
from .results import Results
from .tasks import Suite, Task


class InfraError(Exception):
    """The provider failed (an outage, a rate limit), not the agent."""


async def evaluate(
    suite: Suite,
    *,
    provider: Callable[[], ModelProvider],
    tools: ToolFactory | None = None,
    trials: int = 3,
    k: int | None = None,
    include: Literal["regression", "capability", "all"] = "all",
    task_ids: Collection[str] = (),
    results_dir: str | Path = "results",
    concurrency: int = 4,
    judge_client: Any = None,
    judge_model: str = "",
    label: dict[str, Any] | None = None,
) -> Results:
    """Run the suite's tasks with `tools` (by default, the suite's own) on a fresh provider
    from `provider()` per trial. `label` describes what was tested (model, effort) in the report."""
    tools = tools or suite.tools
    if tools is None:
        raise ValueError("No tools to test: pass tools=, or name them in the suite.")
    if unknown := sorted(set(task_ids) - {t.id for t in suite.tasks}):
        raise ValueError(f"No task called {', '.join(unknown)} in this suite.")
    tasks = [
        t for t in suite.tasks if include in ("all", t.suite) and (not task_ids or t.id in task_ids)
    ]
    # Without this, every judge call would fail as an API error, be counted as infrastructure,
    #   and quietly leave each judged task with no usable trials.
    if any(g.uses_model for t in tasks for g in t.graders) and not (judge_client and judge_model):
        raise ValueError("These tasks have judge graders: pass judge_client= and judge_model=.")
    k = k or trials
    results_dir = Path(results_dir)
    semaphore = asyncio.Semaphore(concurrency)

    async def limited(task: Task, i: int) -> dict[str, Any]:
        async with semaphore:
            return await run_trial(
                suite, task, i, provider, tools, results_dir, judge_client, judge_model
            )

    runs = list(await asyncio.gather(*(limited(t, i) for t in tasks for i in range(trials))))
    return Results(agent=label or {}, k=k, tasks=summarise(tasks, runs, k), trials=runs)


async def run_trial(
    suite: Suite,
    task: Task,
    index: int,
    provider: Callable[[], ModelProvider],
    tools: ToolFactory,
    results_dir: Path,
    judge_client: Any = None,
    judge_model: str = "",
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        env = suite.environment()
        env.setup(Path(tmp))
        person = ScriptedPerson(task.person)  # E8: fresh per trial, like the environment
        start = time.monotonic()
        # E13: an infrastructure error isn't an agent failure. The trial is left out of pass^k
        #   and reported separately, so an outage can't look like the agent getting worse.
        #   Humphrey's loop doesn't raise when the provider fails: it ends the run with an
        #   "error" message (after the provider's own retries), which is what's caught here.
        #   Judge API errors count too. A bug in a grader or the runner must stop the run:
        #   counted as infrastructure, it would quietly shrink n and leave pass^k looking healthy.
        try:
            harness = build_harness(
                provider=provider(), tools=tools(env, person), workspace=env.workspace
            )
            async for _ in harness.prompt(task.input):
                pass
            messages = harness.messages
            ended = transcript.stop_reason(messages)
            if ended == "error":
                last = transcript.last_assistant(messages)
                raise InfraError(last.error_message if last else "no reply from the provider")
            seconds = time.monotonic() - start
            reply = transcript.reply_text(messages)
            tool_log = transcript.tool_log(messages)
            trial = Trial(task, reply, tool_log, env, judge_client, judge_model)
            judged = await asyncio.gather(*(g.grade(trial) for g in task.graders if g.uses_model))
        except Exception as e:
            if not _is_infra(e):
                raise
            return {"task": task.id, "trial": index, "error": repr(e)}  # no transcript for these
        grades = [
            g for spec in task.graders if not spec.uses_model for g in await spec.grade(trial)
        ]
        grades += [g for batch in judged for g in batch]
        snapshot = env.snapshot()  # read before the environment's folder is deleted

    usage = transcript.tokens(messages)
    path = results_dir / "transcripts" / f"{task.id}-{index}.json"
    summary = {
        "task": task.id,
        "trial": index,
        "suite": task.suite,
        "passed": all(g.passed for g in grades),
        # E11 [DE]: partial credit too, so progress shows before a task flips to passing.
        "score": sum(g.passed for g in grades) / len(grades),
        "failed": [g.name for g in grades if not g.passed],
        "unknown": [g.name for g in grades if g.unknown],  # for a person to review
        "grades": [dataclasses.asdict(g) for g in grades],
        "reply": reply,
        "stop_reason": ended,
        # The interruptions the person saw: what this suite is about, so it's in the table.
        "cards": len(person.cards),
        # E10/K2: the numbers that decide whether a change is worth making. Humphrey has no
        #   price table, so tokens stand in for cost.
        "model_calls": usage["model_calls"],
        "tokens": usage["input"] + usage["output"],
        "tool_errors": sum(c.is_error for c in tool_log),
        "seconds": seconds,
        "transcript": str(path),
    }
    # E10 [DE]: the whole transcript, summary first, raw detail after. Reading transcripts is
    #   how you find out whether a failure is the agent's fault or the grader's.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                **summary,
                "input": task.input,
                "tool_log": [dataclasses.asdict(c) for c in tool_log],
                "cards_shown": person.cards,
                "environment": snapshot,
                "usage": usage,
                "messages": [m.model_dump(mode="json") for m in messages],
            },
            indent=2,
            default=str,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return summary


def _is_infra(error: BaseException) -> bool:
    return isinstance(error, (InfraError, openai.APIError))


def summarise(
    tasks: Sequence[Task], trials: list[dict[str, Any]], k: int
) -> dict[str, dict[str, Any]]:
    summary = {}
    for task in tasks:
        runs = [t for t in trials if t["task"] == task.id]
        ok = [t for t in runs if "error" not in t]
        n, c = len(ok), sum(t["passed"] for t in ok)
        mean = {
            key: sum(t[key] for t in ok) / n if n else 0.0
            for key in ("score", "cards", "tokens", "seconds", "model_calls", "tool_errors")
        }
        summary[task.id] = {
            "suite": task.suite,
            "n": n,
            "passed": c,
            "errors": len(runs) - n,
            "pass_at_k": pass_at_k(n, c, k) if n >= k else None,
            "pass_hat_k": pass_hat_k(n, c, k) if n >= k else None,
            **mean,
            "failed_graders": sorted({g for t in ok for g in t["failed"]}),
            "needs_review": sorted({g for t in ok for g in t["unknown"]}),
        }
    return summary


async def replay(
    suite: Suite, task: Task, tools: ToolFactory | None = None
) -> tuple[list[Grade], list[transcript.ToolCall]]:
    """E9: run a task's reference solution on the tools alone, then apply its code graders.

    The task's person answers any ask_user_question step, exactly as in a trial. Graders that
    read the reply are checked against the task's reference_reply, and skipped if it has none.
    Every grade should pass. If one doesn't, the task can't be passed as written or a grader is
    wrong, and you've found out for free, before any trial spends money.
    """
    # No model at all: the tools don't need one, which is what keeping them apart buys (A4).
    tools = tools or suite.tools
    if tools is None:
        raise ValueError("No tools to replay with: name them in the suite.")
    with tempfile.TemporaryDirectory() as tmp:
        env = suite.environment()
        env.setup(Path(tmp))
        hands = tools(env, ScriptedPerson(task.person))
        tool_calls = [
            await transcript.execute(hands, name, args, f"ref_{i}")
            for i, (name, args) in enumerate(task.reference_solution)
        ]
        trial = Trial(task, task.reference_reply or "", tool_calls, env, judge_client=None)
        checkable = [
            g
            for g in task.graders
            if not g.uses_model and (task.reference_reply is not None or not g.reads_reply)
        ]
        grades = [grade for spec in checkable for grade in await spec.grade(trial)]
        return grades, tool_calls
