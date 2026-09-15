"""Recovery analysis: recommendation quality aligned to a known drift onset.

Each injected user has a known onset, so we can align every user's per-step quality to
``rel = position − onset`` and average across users into a *quality-vs-time-since-onset*
curve. Comparing this curve across α-policies is the core evidence for the project: a
drift-aware policy should dip less and recover faster after the shift.

Quality here is the **reciprocal rank** ``1/rank`` of the held-out item (i.e. MRR when
averaged), which — unlike a sparse hit@k — varies continuously and so has enough
resolution to show recovery dynamics on a subsample-scale recommender.

From a curve we read three quantities the protocol asks for: pre-drift quality, the
post-drift minimum (hence the degradation), and the recovery latency — the first
post-onset step at which quality returns to within a fraction of the pre-drift level.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from .streaming import UserRunResult

__all__ = [
    "RecoveryCurve",
    "RecoverySummary",
    "build_recovery_curve",
    "summarize_recovery",
]


@dataclass(frozen=True)
class RecoveryCurve:
    """Mean hit@k at each position relative to the onset."""

    rel_pos: NDArray[np.int64]
    hit_rate: NDArray[np.float64]
    count: NDArray[np.int64]


class RecoverySummary(BaseModel):
    """Quality degradation and recovery read off a recovery curve."""

    model_config = ConfigDict(extra="forbid")

    pre_quality: float
    min_quality: float
    degradation: float
    final_quality: float
    recovery_latency: int | None
    n_users: int


def build_recovery_curve(
    results: list[UserRunResult],
    onset_by_user: dict[int, int],
    k: int,  # noqa: ARG001 — kept for signature symmetry; MRR needs no cut-off
    lo: int = -10,
    hi: int = 25,
) -> RecoveryCurve:
    """Average reciprocal rank across users at each onset-relative position."""
    sums: dict[int, float] = defaultdict(float)
    counts: dict[int, int] = defaultdict(int)
    for result in results:
        onset = onset_by_user.get(result.user)
        if onset is None:
            continue
        quality = 1.0 / result.rank.astype(np.float64)
        rel = np.arange(result.rank.size) - onset
        keep = (rel >= lo) & (rel <= hi)
        for position, value in zip(rel[keep], quality[keep], strict=True):
            sums[int(position)] += float(value)
            counts[int(position)] += 1
    rel_pos = np.arange(lo, hi + 1, dtype=np.int64)
    hit_rate = np.array(
        [sums[p] / counts[p] if counts[p] else np.nan for p in rel_pos]
    )
    count = np.array([counts[p] for p in rel_pos], dtype=np.int64)
    return RecoveryCurve(rel_pos=rel_pos, hit_rate=hit_rate, count=count)


def summarize_recovery(
    curve: RecoveryCurve,
    n_users: int,
    pre_window: int = 8,
    recovery_frac: float = 0.9,
) -> RecoverySummary:
    """Reduce a recovery curve to pre/min/degradation/recovery-latency."""
    pre_mask = (curve.rel_pos >= -pre_window) & (curve.rel_pos < 0)
    post_mask = curve.rel_pos >= 0
    pre = (
        float(np.nanmean(curve.hit_rate[pre_mask])) if pre_mask.any() else float("nan")
    )
    post_rate = curve.hit_rate[post_mask]
    post_pos = curve.rel_pos[post_mask]
    min_q = float(np.nanmin(post_rate)) if post_rate.size else float("nan")
    threshold = recovery_frac * pre
    recovery = next(
        (int(p) for p, q in zip(post_pos, post_rate, strict=True)
         if not np.isnan(q) and q >= threshold),
        None,
    )
    final = float(np.nanmean(post_rate[-3:])) if post_rate.size >= 3 else min_q
    return RecoverySummary(
        pre_quality=pre,
        min_quality=min_q,
        degradation=pre - min_q,
        final_quality=final,
        recovery_latency=recovery,
        n_users=n_users,
    )
