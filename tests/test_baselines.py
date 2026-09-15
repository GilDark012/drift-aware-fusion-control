"""Tests for the classical drift-detector baselines."""

from __future__ import annotations

import numpy as np

from drift_reco.detectors.baselines import (
    BASELINE_DETECTORS,
    Adwin,
    Cusum,
    CusumDetectorConfig,
    CusumDriftDetector,
    PageHinkley,
    build_baseline,
)
from drift_reco.detectors.detector import DriftState


def _step_series(n: int = 80, jump: float = 1.2) -> np.ndarray:
    """A flat low signal that steps up to a high level at the midpoint.

    ADWIN is inherently sluggish on short windows (it must accumulate post-change
    samples before its variance bound is exceeded), so the step is made clear enough
    that every baseline detects it; latency differences are studied separately.
    """
    rng = np.random.default_rng(0)
    low = 0.1 + 0.005 * rng.standard_normal(n // 2)
    high = 0.1 + jump + 0.005 * rng.standard_normal(n - n // 2)
    return np.concatenate([low, high]).astype(np.float64)


def _flat_series(n: int = 60) -> np.ndarray:
    """A stationary low signal with only small noise."""
    rng = np.random.default_rng(1)
    return (0.1 + 0.01 * rng.standard_normal(n)).astype(np.float64)


def _first_alarm(states: np.ndarray) -> int | None:
    """Index of the first CONFIRMED step, or None."""
    hits = np.where(states == int(DriftState.CONFIRMED))[0]
    return int(hits[0]) if hits.size else None


def test_all_detectors_fire_after_a_step() -> None:
    """Every baseline detects a clear step change, at or after the onset."""
    series = _step_series()
    onset = series.size // 2
    for detector in (PageHinkley(), Cusum(), Adwin()):
        states = detector.run(series)
        first = _first_alarm(states)
        assert first is not None, f"{detector.name} missed the step"
        assert first >= onset - 2, f"{detector.name} fired implausibly early ({first})"


def test_detectors_stay_quiet_on_stationary_signal() -> None:
    """No baseline raises many alarms on a stationary low signal."""
    series = _flat_series()
    for detector in (PageHinkley(), Cusum(), Adwin()):
        states = detector.run(series)
        n_alarms = int((states == int(DriftState.CONFIRMED)).sum())
        assert n_alarms <= 1, f"{detector.name} false-alarmed {n_alarms}x when flat"


def test_higher_threshold_is_less_sensitive() -> None:
    """Raising the Page-Hinkley/CUSUM threshold cannot increase alarms."""
    series = _step_series()
    for name in ("page-hinkley", "cusum"):
        low = build_baseline(name, threshold=0.2).run(series)
        high = build_baseline(name, threshold=2.0).run(series)
        n_low = int((low == int(DriftState.CONFIRMED)).sum())
        n_high = int((high == int(DriftState.CONFIRMED)).sum())
        assert n_high <= n_low


def test_registry_covers_all_baselines() -> None:
    """The registry exposes exactly the three implemented detectors."""
    assert set(BASELINE_DETECTORS) == {"page-hinkley", "cusum", "adwin"}


def test_empty_series_returns_empty_states() -> None:
    """An empty divergence series yields an empty states array, no error."""
    for detector in (PageHinkley(), Cusum(), Adwin()):
        states = detector.run(np.zeros(0, dtype=np.float64))
        assert states.shape == (0,)


def test_cusum_detector_holds_confirmed_after_a_step() -> None:
    """The CUSUM-as-detector enters and *holds* CONFIRMED across a sustained shift."""
    series = np.concatenate([np.full(20, 0.1), np.full(20, 0.9)]).astype(np.float64)
    states = CusumDriftDetector(CusumDetectorConfig(enter=0.3)).run(series).states
    confirmed = states == int(DriftState.CONFIRMED)
    assert not confirmed[:20].any(), "confirmed before the shift"
    assert confirmed[-5:].all(), "did not hold CONFIRMED through the sustained shift"


def test_cusum_detector_stays_stable_when_flat() -> None:
    """A stationary signal keeps the CUSUM detector in STABLE throughout."""
    rng = np.random.default_rng(3)
    series = (0.1 + 0.01 * rng.standard_normal(60)).astype(np.float64)
    states = CusumDriftDetector(CusumDetectorConfig(enter=0.5)).run(series).states
    assert (states == int(DriftState.STABLE)).all()
