"""Schema and loading helpers for the processed Amazon interaction splits.

The processed parquet files share one fixed schema. Rather than validate four
million rows through Pydantic (which would be needlessly slow), this module
validates the *frame-level* contract — column presence and dtype family — and
provides a single typed entry point for loading a split, so every downstream
component reads the data the same way.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Final

import pandas as pd

from ..config.constants import (
    CATEGORY_COLUMN,
    INTERACTION_COLUMNS,
    ITEM_COLUMN,
    ITEM_IDX_COLUMN,
    RATING_COLUMN,
    TIMESTAMP_COLUMN,
    USER_COLUMN,
    USER_IDX_COLUMN,
)

__all__ = ["Split", "SplitPaths", "assert_schema", "load_split", "load_all_splits"]


class Split(StrEnum):
    """The three temporal splits, ordered oldest to newest."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class SplitPaths:
    """Resolve split names to parquet paths under a processed-data directory."""

    def __init__(self, processed_dir: str | Path = "data/processed") -> None:
        """Store the directory holding ``train/val/test.parquet``."""
        self.processed_dir = Path(processed_dir)

    def path(self, split: Split) -> Path:
        """Return the parquet path for one split."""
        return self.processed_dir / f"{split.value}.parquet"


# Dtype family expected for each column, checked loosely so that a int32 vs
# int64 or a tz-aware vs naive timestamp does not trip a spurious failure.
_EXPECTED_KIND: Final[dict[str, tuple[str, ...]]] = {
    USER_COLUMN: ("object", "string"),
    ITEM_COLUMN: ("object", "string"),
    CATEGORY_COLUMN: ("object", "string"),
    RATING_COLUMN: ("floating", "integer"),
    TIMESTAMP_COLUMN: ("datetime",),
    USER_IDX_COLUMN: ("integer",),
    ITEM_IDX_COLUMN: ("integer",),
}


def assert_schema(frame: pd.DataFrame) -> None:
    """Validate that a frame matches the interaction contract.

    Args:
        frame: A loaded interaction split.

    Raises:
        ValueError: If a required column is missing or has an unexpected dtype.
    """
    missing = [c for c in INTERACTION_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    for column, kinds in _EXPECTED_KIND.items():
        inferred = pd.api.types.infer_dtype(frame[column], skipna=True)
        dtype = str(frame[column].dtype)
        ok = any(k in inferred or k in dtype for k in kinds)
        if not ok:
            raise ValueError(
                f"column {column!r} has dtype {dtype!r}/{inferred!r}, "
                f"expected one of {kinds}"
            )


def load_split(
    split: Split,
    paths: SplitPaths | None = None,
    *,
    validate: bool = True,
) -> pd.DataFrame:
    """Load one temporal split as a chronologically ordered frame.

    Args:
        split: Which split to load.
        paths: Location resolver; defaults to ``data/processed``.
        validate: Whether to assert the schema after loading.

    Returns:
        The split, sorted by timestamp with a fresh RangeIndex.
    """
    resolver = paths or SplitPaths()
    frame = pd.read_parquet(resolver.path(split))
    if validate:
        assert_schema(frame)
    return frame.sort_values(TIMESTAMP_COLUMN, kind="stable").reset_index(drop=True)


def load_all_splits(
    paths: SplitPaths | None = None, *, validate: bool = True
) -> dict[Split, pd.DataFrame]:
    """Load train/val/test as a single ``Split -> frame`` mapping."""
    resolver = paths or SplitPaths()
    return {split: load_split(split, resolver, validate=validate) for split in Split}
