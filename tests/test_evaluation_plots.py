"""Smoke tests for the evaluation figures."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from drift_reco.evaluation.metrics import compute_ranking_metrics
from drift_reco.evaluation.plots import (
    detection_tradeoff,
    injection_overlay,
    recovery_curves,
)
from drift_reco.evaluation.recovery import build_recovery_curve
from drift_reco.evaluation.streaming import UserRunResult


def _result() -> UserRunResult:
    n = 20
    ranks = np.arange(1, n + 1, dtype=np.int64)
    return UserRunResult(
        user=1,
        timestamp=pd.date_range("2018-01-01", periods=n, freq="D").to_numpy(),
        divergence=np.linspace(0.2, 0.9, n),
        state=np.zeros(n, dtype=np.int64),
        alpha=np.full(n, 0.6),
        rank=ranks,
        target=np.ones(n, dtype=np.int64),
        is_seen=np.ones(n, dtype=bool),
        metrics=compute_ranking_metrics(ranks, np.ones(n, bool), 10),
    )


def test_recovery_curves_figure() -> None:
    curve = build_recovery_curve([_result()], {1: 8}, k=10)
    assert isinstance(recovery_curves({"proposed": curve}), Figure)


def test_detection_tradeoff_figure() -> None:
    points = [(2.0, 0.1, 0.01), (1.75, 0.3, 0.08), (1.5, 0.5, 0.2)]
    assert isinstance(detection_tradeoff(points), Figure)


def test_injection_overlay_figure() -> None:
    assert isinstance(injection_overlay(_result(), onset_index=8), Figure)
