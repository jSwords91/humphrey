"""What an eval run prints. Built to be read: a table per task, then what went wrong in each
failed trial and which transcript to open first."""

from __future__ import annotations

from typing import Any


def fmt(x: float | None) -> str:
    return "     -" if x is None else f"{x:6.2f}"


def print_table(tasks: dict[str, dict[str, Any]], k: int) -> None:
    print(
        f"\n{'task':38} {'suite':10} {'pass':>5} pass@{k} pass^{k} {'score':>5} "
        f"{'cards':>5} {'tokens':>7} {'calls':>5} {'tool err':>8} {'secs':>5}"
    )
    for tid, r in tasks.items():
        notes = (
            [f"failed: {', '.join(r['failed_graders'])}"] * bool(r["failed_graders"])
            + [f"review: {', '.join(r['needs_review'])}"] * bool(r["needs_review"])
            + [f"infra errors: {r['errors']}"] * bool(r["errors"])
        )
        print(
            f"{tid:38} {r['suite']:10} {r['passed']:>2}/{r['n']:<2} {fmt(r['pass_at_k'])} "
            f"{fmt(r['pass_hat_k'])} {r['score']:5.2f} {r['cards']:5.1f} {r['tokens']:7.0f} "
            f"{r['model_calls']:5.1f} {r['tool_errors']:8.1f} {r['seconds']:5.1f}  "
            + "  ".join(notes)
        )
        if r["suite"] == "capability" and r["pass_hat_k"] == 1.0:
            # E6 [DE]: a capability task that passes every time has graduated. Move it to the
            #   regression suite and write a harder one, or the suite stops telling you anything.
            print(f"  ^ {tid} passed every trial: move it to regression")
        if r["n"] >= 3 and r["passed"] == 0:
            # [DE]: a task nothing ever passes is more often broken than hard. Before blaming the
            #   agent, check the task can be passed and everything graded is clear from the input.
            print(f"  ^ {tid} never passed: check the task and its graders before the agent")


def print_comparison(tasks: dict[str, dict[str, Any]], baseline: dict[str, dict[str, Any]]) -> None:
    """E10 [DE]: for a model or prompt change, what matters is which tasks flipped, and what
    happened to tokens and latency. That's the evidence for adopting a new model in days."""
    print("\nAgainst the baseline:")
    shared = [t for t in tasks if t in baseline]
    flipped = [t for t in shared if baseline[t]["pass_hat_k"] != tasks[t]["pass_hat_k"]]
    for t in flipped:
        print(f"  {t}: pass^k {fmt(baseline[t]['pass_hat_k'])} -> {fmt(tasks[t]['pass_hat_k'])}")
    if not flipped:
        print("  no task changed pass^k")
    for key in ("tokens", "seconds"):
        before = sum(baseline[t][key] for t in shared)
        after = sum(tasks[t][key] for t in shared)
        print(f"  total {key} per trial: {before:.3f} -> {after:.3f}")


def print_failures(trials: list[dict[str, Any]]) -> None:
    failed = [t for t in trials if "error" in t or not t["passed"]]
    if not failed:
        print("\nEvery trial passed.")
        return
    print(f"\nFailed trials ({len(failed)}):")
    for t in sorted(failed, key=lambda t: (t["task"], t["trial"])):
        print(f"\n  {t['task']} #{t['trial']}")
        if "error" in t:
            print(f"    infrastructure error: {t['error']}")
            continue
        for g in t["grades"]:
            if not g["passed"]:
                print(f"    x {g['name']}: {g['detail'] or 'failed'}")
        if t["stop_reason"] != "stop":
            print(f"    stopped early: {t['stop_reason']}")
        print(f"    reply: {' '.join(t['reply'].split())[:160]}")
        print(f"    transcript: {t['transcript']}")
