"""Statistics for a paired A/B eval.

Every task is run twice (baseline vs compressed) on the SAME input, so the
comparison is *paired*. That's important: pairing removes task-to-task variance
and gives far tighter confidence intervals than comparing two independent groups.

We care about two questions:
  1. Savings: how much input/cost did compression remove?  (point estimate + CI)
  2. Quality: is compressed NON-INFERIOR to baseline within a margin delta?
     i.e. is the quality drop small enough to accept for the savings?

Non-inferiority, not superiority: we are not trying to prove compression makes
answers *better* -- only that it doesn't make them meaningfully *worse*. The test
is one-sided: the upper bound of the (baseline - compressed) quality-loss CI must
sit below the tolerance delta.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple


def _bootstrap_mean_ci(
    samples: Sequence[float], iters: int = 10000, alpha: float = 0.05, seed: int = 0
) -> Tuple[float, float, float]:
    """Return (mean, lo, hi) percentile bootstrap CI of the mean."""
    if not samples:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(samples)
    means = []
    for _ in range(iters):
        resample = [samples[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    lo = means[int((alpha / 2) * iters)]
    hi = means[int((1 - alpha / 2) * iters)]
    return sum(samples) / n, lo, hi


@dataclass
class SavingsReport:
    n: int
    mean_saved_pct: float
    ci_lo: float
    ci_hi: float
    mean_cost_baseline: float
    mean_cost_compressed: float
    mean_cost_reduction_pct: float


def savings_report(
    baseline_input: List[int],
    compressed_input: List[int],
    baseline_cost: List[float],
    compressed_cost: List[float],
) -> SavingsReport:
    pct = [
        100.0 * (b - c) / b if b else 0.0
        for b, c in zip(baseline_input, compressed_input)
    ]
    mean, lo, hi = _bootstrap_mean_ci(pct)
    mb = sum(baseline_cost) / len(baseline_cost) if baseline_cost else 0.0
    mc = sum(compressed_cost) / len(compressed_cost) if compressed_cost else 0.0
    cost_red = 100.0 * (mb - mc) / mb if mb else 0.0
    return SavingsReport(
        n=len(pct), mean_saved_pct=mean, ci_lo=lo, ci_hi=hi,
        mean_cost_baseline=mb, mean_cost_compressed=mc,
        mean_cost_reduction_pct=cost_red,
    )


@dataclass
class QualityReport:
    n: int
    mean_baseline: float
    mean_compressed: float
    mean_delta: float            # baseline - compressed (loss; >0 means worse)
    delta_ci_lo: float
    delta_ci_hi: float
    non_inferiority_margin: float
    non_inferior: bool           # is the WORST plausible loss within margin?
    wins: int
    ties: int
    losses: int


def quality_report(
    baseline_scores: List[float],
    compressed_scores: List[float],
    margin: float = 0.03,
    tie_eps: float = 1e-9,
) -> QualityReport:
    """Paired quality comparison.

    margin is expressed in the same units as the scores (e.g. 0.03 = a 3-point
    drop on a 0..1 quality scale is the most you'll tolerate). Non-inferiority
    holds when the upper CI bound of the per-task quality LOSS is <= margin.
    """
    losses_per_task = [b - c for b, c in zip(baseline_scores, compressed_scores)]
    mean_delta, lo, hi = _bootstrap_mean_ci(losses_per_task)
    wins = sum(1 for d in losses_per_task if d < -tie_eps)   # compressed better
    losses = sum(1 for d in losses_per_task if d > tie_eps)  # compressed worse
    ties = len(losses_per_task) - wins - losses
    mb = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.0
    mc = sum(compressed_scores) / len(compressed_scores) if compressed_scores else 0.0
    return QualityReport(
        n=len(losses_per_task),
        mean_baseline=mb, mean_compressed=mc,
        mean_delta=mean_delta, delta_ci_lo=lo, delta_ci_hi=hi,
        non_inferiority_margin=margin,
        non_inferior=(hi <= margin),
        wins=wins, ties=ties, losses=losses,
    )
