"""Tests for onset-aligned recovery analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from drift_reco.evaluation.metrics import compute_ranking_metrics
from drift_reco.evaluation.recovery import build_recovery_curve, summarize_recovery
from drift_reco.evaluation.streaming import UserRunResult


def _result(user: int, ranks: np.ndarray) -> UserRunResult:
    n = ranks.size
    return UserRunResult(
        user=user,
        timestamp=pd.date_range("2018-01-01", periods=n, freq="D").to_numpy(),
        divergence=np.zeros(n),
        state=np.zeros(n, dtype=np.int64),
        alpha=np.full(n, 0.8),
        rank=ranks.astype(np.int64),
        target=np.ones(n, dtype=np.int64),
        is_seen=np.ones(n, dtype=bool),
        metrics=compute_ranking_metrics(ranks.astype(np.int64), np.ones(n, bool), 10),
    )


def test_recovery_curve_dips_after_onset() -> None:
    # good (rank 1) before onset=5, bad (rank 100) after -> quality drops
    ranks = np.array([1, 1, 1, 1, 1, 100, 100, 100, 100, 100])
    curve = build_recovery_curve([_result(1, ranks)], {1: 5}, k=10, lo=-5, hi=4)
    pre = curve.hit_rate[curve.rel_pos < 0]
    post = curve.hit_rate[curve.rel_pos >= 0]
    assert np.nanmean(pre) > np.nanmean(post)


def test_summary_reports_degradation_and_recovery() -> None:
    # good, then bad, then good again -> degradation then recovery
    ranks = np.array([1, 1, 1, 1, 1, 100, 100, 1, 1, 1])
    curve = build_recovery_curve([_result(1, ranks)], {1: 5}, k=10, lo=-5, hi=4)
    summary = summarize_recovery(curve, n_users=1, pre_window=5, recovery_frac=0.9)
    assert summary.degradation > 0
    assert summary.recovery_latency is not None
    assert summary.recovery_latency >= 2


def test_empty_users_produce_nan_curve() -> None:
    curve = build_recovery_curve([], {}, k=10, lo=-2, hi=2)
    assert np.isnan(curve.hit_rate).all()
