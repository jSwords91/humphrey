"""The command line.

    python -m evals check TASKS               validate tasks, replay reference solutions (free)
    python -m evals run   TASKS [options]     run the evals against OpenAI (costs money)

Run it from the repository root, so the names in the tasks file can be imported.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import openai

from app.settings import active_provider_kind, get_settings
from providers import get_provider

from .runner import evaluate, replay
from .tasks import load_suite

# A3 [HD]: the judge should be at least as capable as the agent, and ideally a different model.
DEFAULT_JUDGE_MODEL = "gpt-6.1-sol"


async def check(args: argparse.Namespace) -> int:
    suite = load_suite(args.tasks)  # raises on any malformed task
    print(f"{len(suite.tasks)} tasks in {args.tasks} are valid. Replaying reference solutions:\n")
    failures = 0
    for task in suite.tasks:
        grades, calls = await replay(suite, task)
        bad = [g for g in grades if not g.passed]
        failures += bool(bad)
        unchecked = any(g.reads_reply for g in task.graders) and task.reference_reply is None
        print(
            f"  {'ok' if not bad else 'FAIL':6}{task.id}"
            + ("  (reply graders not checked: add a reference_reply)" if unchecked else "")
        )
        for g in bad:
            print(f"        x {g.name}: {g.detail}")
        for c in calls:
            if c.is_error:  # sometimes intended, often a typo
                print(f"        ! reference step {c.name} returned an error: {c.output[:120]}")
    return 1 if failures else 0


async def preflight(judge: openai.AsyncOpenAI) -> None:
    """Fail in a second, with a plain message, rather than after starting every trial."""
    if active_provider_kind(get_settings()) != "openai":
        raise SystemExit(
            "Evals run on OpenAI with an API key: set OPENAI_API_KEY (and not MOCK_MODE)."
        )
    try:
        await judge.models.list()  # free, and needs valid credentials
    except openai.AuthenticationError as e:
        raise SystemExit(f"Can't call the OpenAI API: {e}") from None


def contained(on_host: bool) -> None:
    """Refuse a live run outside the evals container unless asked for explicitly.

    The tasks ask Humphrey to delete and tidy files. Its file tools are confined to a trial's
    folder, but bash only starts there, so a confused command could reach your own files.
    """
    if on_host or os.environ.get("HUMPHREY_EVALS_IN_CONTAINER") == "1":
        return
    raise SystemExit(
        "Live evals run Humphrey's bash tool, which isn't confined to the trial's folder.\n"
        "Run them in the container:\n"
        "  docker compose run --rm evals run evals/ask_user/tasks.yaml\n"
        "or pass --on-host if you accept the risk."
    )


async def run(args: argparse.Namespace) -> int:
    contained(args.on_host)
    suite = load_suite(args.tasks)
    settings = get_settings()
    # K1: the experiments you'll actually run are a different model or effort on the same
    #   agent, so they're flags, not code changes.
    model = args.model or settings.openai_model
    effort = args.effort or settings.reasoning_effort
    if args.judge_model == model:
        # A3: ideally a different model, but it's still a separate call that sees only the
        #   transcript, so the same model is allowed. Said once, so it's a choice, not an accident.
        print(f"Note: the judge and the agent are both {model}. Ideally, pass --judge-model.")
    judge = openai.AsyncOpenAI()
    await preflight(judge)
    try:
        results = await evaluate(
            suite,
            provider=lambda: get_provider(model, reasoning_effort=effort),
            trials=args.trials,
            k=args.k,
            include=args.suite,
            task_ids=args.task,
            results_dir=args.results,
            concurrency=args.concurrency,
            judge_client=judge,
            judge_model=args.judge_model,
            label={"model": model, "effort": effort, "judge": args.judge_model},
        )
    except ValueError as e:  # an unknown --task, say
        raise SystemExit(str(e)) from None

    results.table()
    if args.compare:
        results.compare(args.compare)
    results.failures()
    out = results.save(args.results / "latest.json")
    print(f"\nWrote {out} and transcripts in {args.results / 'transcripts'}")

    if args.gate is None:
        return 0
    mean, passed = results.gate(args.gate)
    print(
        "Regression gate: FAIL, no regression tasks ran"
        if mean is None
        else f"Regression gate: mean pass^{results.k} {mean:.2f} (need {args.gate}): "
        + ("PASS" if passed else "FAIL")
    )
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m evals")
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser(
        "check", help="validate tasks and replay reference solutions, no API calls"
    )
    p.add_argument("tasks", type=Path)

    p = commands.add_parser("run", help="run the evals")
    p.add_argument("tasks", type=Path)
    p.add_argument("--suite", choices=["regression", "capability", "all"], default="all")
    p.add_argument("--task", action="append", default=[], help="run only this task id (repeatable)")
    p.add_argument("--trials", type=int, default=3, help="trials per task (n)")
    p.add_argument("--k", type=int, help="k for pass@k and pass^k (default: --trials)")
    p.add_argument("--model", help="the model under test (default: OPENAI_MODEL)")
    p.add_argument("--effort", help="its reasoning effort (default: REASONING_EFFORT)")
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="the judge's model")
    p.add_argument("--gate", type=float, help="exit 1 unless mean regression pass^k reaches this")
    p.add_argument("--compare", type=Path, help="an earlier results JSON to compare against")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument(
        "--on-host",
        action="store_true",
        help="run outside the evals container (bash can reach your files)",
    )

    args = parser.parse_args()
    return asyncio.run({"check": check, "run": run}[args.command](args))


if __name__ == "__main__":
    sys.exit(main())
