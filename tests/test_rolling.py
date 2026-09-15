"""Tests for causal rolling statistics."""

from __future__ import annotations

import numpy as np

from drift_reco.detectors.rolling import RollingConfig, RollingStatistics


def test_rolling_mean_is_causal() -> None:
    x = np.arange(10, dtype=np.float64)
    stats = RollingStatistics(RollingConfig(window=3, min_periods=1)).transform(x)
    # causal mean at index 4 uses x[2:5] = [2,3,4] -> 3.0, never future values.
    assert np.isclose(stats.mean[4], 3.0)
    assert np.isclose(stats.mean[0], 0.0)


def test_zscore_spikes_after_a_step_change() -> None:
    x = np.concatenate([np.zeros(15), np.full(15, 5.0)]).astype(np.float64)
    stats = RollingStatistics(RollingConfig(window=10, min_periods=3)).transform(x)
    # right after the jump the current value is far above its trailing baseline
    assert stats.zscore[15] > 2.5
    # deep inside the stable second regime the z-score has relaxed
    assert abs(stats.zscore[29]) < stats.zscore[15]


def test_output_length_matches_input() -> None:
    x = np.random.default_rng(0).normal(size=50)
    stats = RollingStatistics().transform(x)
    assert stats.mean.shape == x.shape
    assert stats.ewma.shape == x.shape
