"""Shared fixtures: a tiny synthetic processed+drift dataset on disk.

The fixture mirrors the real layout (``processed/{train,val,test}.parquet`` and
``drift/{test_control,test_drifted}.parquet`` + ground truth) at miniature scale,
with a deliberately broken drifted frame so the validator's failure paths are
exercised without touching the multi-gigabyte real data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


def _frame(rows: list[tuple[str, str, str, float, str, int, int]]) -> pd.DataFrame:
    """Build an interaction frame from positional tuples."""
    df = pd.DataFrame(
        rows,
        columns=[
            "user_id",
            "item_id",
            "category",
            "preference_score",
            "timestamp",
            "user_idx",
            "item_idx",
        ],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Path:
    """Write a miniature dataset and return its root directory."""
    root = tmp_path
    processed = root / "processed"
    drift = root / "drift"
    processed.mkdir()
    drift.mkdir()

    cat = "Electronics"
    train = _frame(
        [
            ("u0", "i0", cat, 5.0, "2014-01-01", 0, 0),
            ("u0", "i1", cat, 4.0, "2014-02-01", 0, 1),
            ("u1", "i0", cat, 3.0, "2014-03-01", 1, 0),
            ("u1", "i2", "Books", 5.0, "2014-04-01", 1, 2),
            ("u2", "i3", "Books", 4.0, "2014-05-01", 2, 3),
            ("u3", "i1", cat, 2.0, "2014-06-01", 3, 1),
        ]
    )
    val = _frame(
        [
            ("u0", "i2", "Books", 4.0, "2015-01-01", 0, 2),
            ("u2", "i0", cat, 5.0, "2015-02-01", 2, 0),
        ]
    )
    # test: u0,u1,u2 seen in train; u4 is cold-start (train-unseen). i9 is a new item.
    test = _frame(
        [
            ("u0", "i3", "Books", 5.0, "2016-01-01", 0, 3),
            ("u1", "i1", cat, 4.0, "2016-02-01", 1, 1),
            ("u2", "i9", "Books", 3.0, "2016-03-01", 2, 9),
            ("u4", "i0", cat, 5.0, "2016-04-01", 4, 0),
        ]
    )
    train.to_parquet(processed / "train.parquet")
    val.to_parquet(processed / "val.parquet")
    test.to_parquet(processed / "test.parquet")

    # Control == test; drifted swaps item_id but NOT item_idx -> index-inconsistent.
    test.to_parquet(drift / "test_control.parquet")
    drifted = test.copy()
    drifted.loc[2, "item_id"] = "iZZZ"
    drifted.loc[2, "category"] = "Clothing_Shoes_and_Jewelry"
    drifted.to_parquet(drift / "test_drifted.parquet")

    onset_ts = test.iloc[2]["timestamp"]
    (drift / "drift_ground_truth.json").write_text(
        json.dumps(
            {"D1": {"type": "sudden", "timestamp": str(onset_ts), "idx": 2}}
        )
    )
    return root
