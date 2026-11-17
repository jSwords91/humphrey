"""The results of an eval run, and what you do with them: read, compare, gate, save.

results = await evaluate(suite, provider=...)
results.table()                   # one row per task
results.failures()                # what went wrong in each failed trial
results.transcript("two_sams_asks_which")
results.compare(baseline)         # which tasks flipped, and tokens and latency
results.save("results/latest.json")
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .metrics import regression_gate
from .report import print_comparison, print_failures, print_table


@dataclass
class Results:
    agent: dict[str, Any]  # the model and effort that were tested
    k: int
    tasks: dict[str, dict[str, Any]]  # per task: pass@k, pass^k, cards, tokens, latency...
    trials: list[dict[str, Any]]  # per trial: grades, reply, cards, tokens, transcript path

    @property
    def tokens(self) -> int:
        return sum(t.get("tokens", 0) for t in self.trials)

    def table(self) -> None:
        print(f"{self.agent}, k={self.k}, total tokens {self.tokens:,}")
        print_table(self.tasks, self.k)

    def failures(self) -> None:
        print_failures(self.trials)

    def compare(self, baseline: Results | str | Path) -> None:
        """E10: which tasks changed pass^k against an earlier run, and what happened to tokens."""
        if not isinstance(baseline, Results):
            baseline = Results.load(baseline)
        print_comparison(self.tasks, baseline.tasks)

    def transcript(self, task_id: str, trial: int = 0) -> dict[str, Any]:
        """The full record of one trial: read these to find out why a task failed."""
        summary = next(t for t in self.trials if t["task"] == task_id and t["trial"] == trial)
        if "error" in summary:
            raise ValueError(f"{task_id} #{trial} hit an infrastructure error: {summary['error']}")
        return json.loads(Path(summary["transcript"]).read_text(encoding="utf-8"))

    def gate(self, threshold: float) -> tuple[float | None, bool]:
        """Mean pass^k over the regression tasks, and whether it reaches `threshold` (E5)."""
        return regression_gate(
            [r["pass_hat_k"] for r in self.tasks.values() if r["suite"] == "regression"], threshold
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"agent": self.agent, "k": self.k, "tasks": self.tasks, "trials": self.trials},
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> Results:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            agent=data.get("agent", {}), k=data["k"], tasks=data["tasks"], trials=data["trials"]
        )
