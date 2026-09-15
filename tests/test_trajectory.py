"""Tests for per-user divergence trajectories."""

from __future__ import annotations

import numpy as np
import pandas as pd

from drift_reco.detectors.signal import get_signal
from drift_reco.detectors.trajectory import DivergenceTracker, user_streams
from drift_reco.models.config import ModelConfig
from drift_reco.models.fusion import LongShortFusion


def _stream(user: int, n: int) -> pd.DataFrame:
    rng = np.random.default_rng(user)
    return pd.DataFrame(
        {
            "s_user": user,
            "s_item": rng.integers(2, 12, size=n),
            "category": ["Books"] * (n // 2) + ["Electronics"] * (n - n // 2),
            "timestamp": pd.date_range("2016-01-01", periods=n, freq="D"),
        }
    )


def _tracker() -> DivergenceTracker:
    cfg = ModelConfig(embedding_dim=8, short_window=3, max_history=4, min_context=2)
    model = LongShortFusion(12, cfg)
    return DivergenceTracker(model, cfg, get_signal("short_term_drift"), "cpu")


def test_trajectory_starts_after_the_short_window() -> None:
    traj = _tracker().user_trajectory(_stream(1, 12))
    # first scored position must have a non-empty long history (> short_window)
    assert traj.step.min() >= 4
    assert len(traj) == traj.divergence.size


def test_category_change_mask_marks_the_switch() -> None:
    traj = _tracker().user_trajectory(_stream(2, 12))
    assert traj.category_changed().sum() >= 1
    assert traj.category_changed()[0] == False  # noqa: E712


def test_user_streams_groups_by_user() -> None:
    frames = [_stream(1, 6), _stream(2, 6)]
    streams = user_streams(frames)
    assert set(streams) == {1, 2}
    assert streams[1]["timestamp"].is_monotonic_increasing


def test_trajectory_frame_has_expected_columns() -> None:
    frame = _tracker().user_trajectory(_stream(3, 12)).to_frame()
    assert {"user", "step", "divergence", "category_changed"} <= set(frame.columns)
