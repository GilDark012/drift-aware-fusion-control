"""Tests for the threshold strategies."""

from __future__ import annotations

import numpy as np
import pytest

from drift_reco.detectors.rolling import RollingStatistics
from drift_reco.detectors.threshold import (
    THRESHOLD_STRATEGIES,
    get_threshold,
)


def test_global_score_is_raw_divergence() -> None:
    d = np.array([0.1, 0.9, 1.4], dtype=np.float64)
    rolling = RollingStatistics().transform(d)
    score = get_threshold("global").score(d, rolling)
    assert np.allclose(score, d)


def test_adaptive_score_is_the_zscore() -> None:
    d = np.concatenate([np.zeros(10), np.full(10, 3.0)]).astype(np.float64)
    rolling = RollingStatistics().transform(d)
    score = get_threshold("adaptive").score(d, rolling)
    assert np.allclose(score, rolling.zscore, equal_nan=True)


def test_registry_and_unknown() -> None:
    assert set(THRESHOLD_STRATEGIES) == {"global", "adaptive"}
    with pytest.raises(KeyError):
        get_threshold("nope")
