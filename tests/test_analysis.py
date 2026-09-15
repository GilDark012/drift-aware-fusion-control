"""Tests for detector aggregation and tuning helpers."""

from __future__ import annotations

import numpy as np

from drift_reco.detectors.analysis import (
    persistence_sweep,
    run_detector_over,
    summarize,
    volatility_strata,
)
from drift_reco.detectors.detector import DetectorConfig


def _cfg(**kw: object) -> DetectorConfig:
    base = {"strategy": "global", "enter": 1.0, "exit": 0.5, "persistence": 3, "warmup": 0}
    base.update(kw)
    return DetectorConfig(**base)  # type: ignore[arg-type]


def test_summarize_counts_confirmations() -> None:
    series = [np.array([0, 0, 0, 2, 2, 2, 0, 0], dtype=np.float64)]
    summary = summarize(run_detector_over(series, _cfg()))
    assert summary.n_users == 1
    assert summary.n_confirmed == 1
    assert summary.total_steps == 8
    assert summary.users_with_confirm_frac == 1.0


def test_persistence_sweep_reduces_confirmations() -> None:
    # a run of elevated steps long enough for low persistence, short for high
    series = [np.array([2, 2, 2, 2, 0, 0], dtype=np.float64)]
    sweep = persistence_sweep(series, _cfg(), [1, 3, 8])
    confirmed = [s.confirmed_per_1k for _, s in sweep]
    assert confirmed[0] >= confirmed[-1]
    assert confirmed[-1] == 0.0  # persistence 8 cannot be met in 4 elevated steps


def test_volatility_strata_orders_by_spread() -> None:
    rng = np.random.default_rng(0)
    series = [rng.normal(scale=scale, size=30) for scale in (0.1, 0.1, 1.0, 1.0, 5.0, 5.0)]
    strata = volatility_strata(series, 3)
    assert strata.shape == (6,)
    # the calmest series sits in a lower stratum than the noisiest
    assert strata[0] <= strata[-1]
