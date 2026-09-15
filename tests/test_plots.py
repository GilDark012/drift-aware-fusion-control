"""Smoke tests for the diagnostic figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib
from matplotlib.figure import Figure

matplotlib.use("Agg")

from drift_reco.data.plots import DataValidationFigures  # noqa: E402
from drift_reco.data.schema import Split, SplitPaths, load_split  # noqa: E402


def _frames(root: Path) -> dict[str, object]:
    paths = SplitPaths(root / "processed")
    return {s.value: load_split(s, paths) for s in Split}


def test_user_activity_hist_returns_a_figure(synthetic_dataset: Path) -> None:
    figures = DataValidationFigures(_frames(synthetic_dataset))
    assert isinstance(figures.user_activity_hist(), Figure)


def test_category_share_over_time_returns_a_figure(synthetic_dataset: Path) -> None:
    figures = DataValidationFigures(_frames(synthetic_dataset))
    assert isinstance(figures.category_share_over_time(), Figure)


def test_rating_distribution_returns_a_figure(synthetic_dataset: Path) -> None:
    figures = DataValidationFigures(_frames(synthetic_dataset))
    assert isinstance(figures.rating_distribution(), Figure)
