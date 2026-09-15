"""Tests for the persistence-aware drift detector state machine."""

from __future__ import annotations

import numpy as np

from drift_reco.detectors.detector import (
    DetectorConfig,
    DriftState,
    PersistenceDriftDetector,
)


def _global_cfg(**kw: object) -> DetectorConfig:
    base = {
        "strategy": "global",
        "enter": 1.0,
        "exit": 0.5,
        "persistence": 3,
        "gap_tolerance": 1,
        "recovery_persistence": 2,
        "warmup": 0,
    }
    base.update(kw)
    return DetectorConfig(**base)  # type: ignore[arg-type]


def test_sustained_rise_confirms_with_correct_onset_and_detect() -> None:
    series = np.array([0, 0, 0, 2, 2, 2, 0, 0], dtype=np.float64)
    result = PersistenceDriftDetector(_global_cfg()).run(series)
    assert result.n_confirmed == 1
    drift = result.confirmed[0]
    assert drift.onset_step == 3
    assert drift.detect_step == 5  # third elevated step
    assert result.states[5] == int(DriftState.CONFIRMED)


def test_short_blip_is_temporary_not_confirmed() -> None:
    series = np.array([0, 0, 2, 0, 0, 0], dtype=np.float64)
    result = PersistenceDriftDetector(_global_cfg()).run(series)
    assert result.n_confirmed == 0
    assert result.n_temporary >= 1


def test_gap_tolerance_allows_one_dip() -> None:
    # elevated, dip once (within tolerance), elevated, elevated -> confirms
    series = np.array([2, 0, 2, 2, 2], dtype=np.float64)
    result = PersistenceDriftDetector(_global_cfg(persistence=3)).run(series)
    assert result.n_confirmed == 1


def test_recovery_returns_to_stable() -> None:
    series = np.array([2, 2, 2, 0, 0, 0], dtype=np.float64)
    result = PersistenceDriftDetector(_global_cfg()).run(series)
    assert result.states[-1] == int(DriftState.STABLE)


def test_warmup_suppresses_early_detection() -> None:
    series = np.full(10, 2.0, dtype=np.float64)
    result = PersistenceDriftDetector(_global_cfg(warmup=5, persistence=3)).run(series)
    # cannot confirm before warmup ends at index 5
    assert result.first_detection() is not None
    assert result.first_detection() >= 5


def test_adaptive_confirms_after_a_step_change() -> None:
    series = np.concatenate([np.zeros(20), np.full(20, 5.0)]).astype(np.float64)
    cfg = DetectorConfig(strategy="adaptive", enter=2.0, exit=1.0, persistence=3, warmup=5)
    result = PersistenceDriftDetector(cfg).run(series)
    assert result.n_confirmed >= 1
    assert result.first_detection() >= 20
