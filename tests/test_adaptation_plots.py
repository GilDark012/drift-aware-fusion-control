"""Smoke tests for the adaptive-fusion figures."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from drift_reco.adaptation.plots import alpha_policy_comparison, alpha_trajectory
from drift_reco.evaluation.metrics import compute_ranking_metrics
from drift_reco.evaluation.streaming import UserRunResult


def _result(user: int, alpha_level: float) -> UserRunResult:
    n = 30
    dummy = np.ones(n, dtype=np.int64)
    return UserRunResult(
        user=user,
        timestamp=pd.date_range("2018-01-01", periods=n, freq="D").to_numpy(),
        divergence=np.linspace(0.4, 1.4, n),
        state=np.zeros(n, dtype=np.int64),
        alpha=np.full(n, alpha_level),
        rank=dummy,
        target=dummy,
        is_seen=np.zeros(n, dtype=bool),
        metrics=compute_ranking_metrics(dummy, np.zeros(n, dtype=bool), 10),
    )


def test_alpha_trajectory_returns_figure() -> None:
    assert isinstance(alpha_trajectory(_result(1, 0.5)), Figure)


def test_alpha_policy_comparison_returns_figure() -> None:
    results = {"gradual": _result(1, 0.6), "abrupt": _result(1, 0.4)}
    assert isinstance(alpha_policy_comparison(results), Figure)
