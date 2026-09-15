"""Smoke tests for divergence diagnostic figures."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from drift_reco.detectors.analysis import (
    persistence_sweep,
)
from drift_reco.detectors.detector import DetectorConfig, PersistenceDriftDetector
from drift_reco.detectors.plots import (
    DivergenceFigures,
    persistence_sweep_figure,
    strategy_stratum_figure,
)
from drift_reco.detectors.trajectory import UserDivergence


def _user_div(n: int = 30) -> UserDivergence:
    rng = np.random.default_rng(0)
    return UserDivergence(
        user=1,
        step=np.arange(n, dtype=np.int64),
        divergence=rng.random(n).astype(np.float64),
        timestamp=pd.date_range("2016-01-01", periods=n, freq="D").to_numpy(),
        category=np.array(["Books"] * (n // 2) + ["Electronics"] * (n - n // 2)),
        target=rng.integers(2, 12, size=n).astype(np.int64),
        is_seen=np.ones(n, dtype=bool),
    )


def _population() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 200
    return pd.DataFrame(
        {
            "divergence": rng.random(n),
            "category_changed": rng.random(n) > 0.5,
        }
    )


def test_trajectory_figure() -> None:
    assert isinstance(DivergenceFigures().trajectory(_user_div()), Figure)


def test_population_effect_figure() -> None:
    assert isinstance(
        DivergenceFigures().population_category_effect(_population()), Figure
    )


def test_distribution_figure() -> None:
    assert isinstance(DivergenceFigures().divergence_distribution(_population()), Figure)


def test_detection_overlay_figure() -> None:
    user = _user_div(40)
    cfg = DetectorConfig(strategy="global", enter=0.6, exit=0.3, persistence=2, warmup=0)
    result = PersistenceDriftDetector(cfg).run(user.divergence.astype(float))
    assert isinstance(DivergenceFigures().detection_overlay(user, result), Figure)


def test_persistence_sweep_figure() -> None:
    series = [_user_div(40).divergence.astype(float)]
    cfg = DetectorConfig(strategy="global", enter=0.6, exit=0.3, persistence=3, warmup=0)
    sweep = persistence_sweep(series, cfg, [1, 3, 5])
    assert isinstance(persistence_sweep_figure(sweep), Figure)


def test_strategy_stratum_figure() -> None:
    table = pd.DataFrame(
        {
            "strategy": ["adaptive", "adaptive", "global", "global"],
            "stratum": [0, 1, 0, 1],
            "confirmed_per_1k": [0.1, 0.2, 2.0, 3.0],
        }
    )
    assert isinstance(strategy_stratum_figure(table), Figure)
