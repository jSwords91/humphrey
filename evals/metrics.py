"""pass@k and pass^k, and the gate CI applies to them.

E5 [DE]: one run says little about a non-deterministic agent, so each task gets n trials.
  pass@k is the chance that at least one of k attempts succeeds: fine when a person picks the
  best of several. pass^k is the chance that all k succeed: at 75% per trial, pass^3 is about
  42%. An agent that acts for someone has to work every time, so CI gates on pass^k.
  Both are unbiased estimates from n >= k trials. With fewer usable trials than k there is no
  estimate, and the gate counts that as a fail.
"""

from __future__ import annotations

from collections.abc import Iterable
from math import comb


def pass_at_k(n: int, c: int, k: int) -> float:
    """Chance that at least one of k trials, drawn from n with c passes, passes."""
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """Chance that all k trials, drawn from n with c passes, pass."""
    return comb(c, k) / comb(n, k)


def regression_gate(scores: Iterable[float | None], threshold: float) -> tuple[float | None, bool]:
    """Mean pass^k over the regression tasks that ran, and whether it reaches `threshold`.

    A task without enough usable trials scores 0. No regression tasks at all is a failure, not a
    pass: a gate with nothing to check proves nothing, and a typo in --suite or --task would
    otherwise turn CI green.
    """
    values = [s or 0.0 for s in scores]
    if not values:
        return None, False
    mean = sum(values) / len(values)
    return mean, mean >= threshold
