"""Tests for the interaction schema helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from drift_reco.data.schema import Split, SplitPaths, assert_schema, load_split


def _valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u0"],
            "item_id": ["i0"],
            "category": ["Electronics"],
            "preference_score": [5.0],
            "timestamp": pd.to_datetime(["2016-01-01"]),
            "user_idx": [0],
            "item_idx": [0],
        }
    )


def test_assert_schema_accepts_a_valid_frame() -> None:
    assert_schema(_valid_frame())


def test_assert_schema_rejects_a_missing_column() -> None:
    frame = _valid_frame().drop(columns=["item_idx"])
    with pytest.raises(ValueError, match="missing required columns"):
        assert_schema(frame)


def test_assert_schema_rejects_a_bad_dtype() -> None:
    frame = _valid_frame()
    frame["timestamp"] = ["not-a-date"]
    with pytest.raises(ValueError, match="timestamp"):
        assert_schema(frame)


def test_split_paths_resolve_expected_filenames() -> None:
    paths = SplitPaths("data/processed")
    assert paths.path(Split.TRAIN) == Path("data/processed/train.parquet")


def test_load_split_sorts_and_validates(synthetic_dataset: Path) -> None:
    paths = SplitPaths(synthetic_dataset / "processed")
    frame = load_split(Split.TEST, paths)
    assert list(frame.columns)
    assert frame["timestamp"].is_monotonic_increasing
