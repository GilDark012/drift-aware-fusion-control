"""Data-quality validation for the processed Amazon interaction splits.

The validator operates strictly on the *already generated* parquet files. It does
not re-ingest or re-split anything; its job is to certify that the frozen data is
consistent, leakage-free and correctly annotated with drift ground truth before
any model is built on top of it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..config.constants import (
    CATEGORY_COLUMN,
    ITEM_COLUMN,
    ITEM_IDX_COLUMN,
    RATING_COLUMN,
    TIMESTAMP_COLUMN,
    USER_COLUMN,
    USER_IDX_COLUMN,
)
from .schema import Split, SplitPaths, load_all_splits
from .validation_models import (
    ActivityProfile,
    CrossSplitStats,
    DataValidationReport,
    DriftGroundTruthReport,
    DriftOnsetCheck,
    LeakageReport,
    SplitStats,
)

__all__ = ["DataValidator"]

_KEY_COLUMNS = [USER_COLUMN, ITEM_COLUMN, TIMESTAMP_COLUMN]


def _activity_profile(counts: NDArray[np.int64], entity: str) -> ActivityProfile:
    """Summarise a vector of per-entity interaction counts."""
    q = np.percentile(counts, [5, 25, 50, 75, 95, 99])
    return ActivityProfile(
        entity=entity,
        count=int(counts.size),
        min=int(counts.min()),
        p05=float(q[0]),
        p25=float(q[1]),
        median=float(q[2]),
        mean=float(counts.mean()),
        p75=float(q[3]),
        p95=float(q[4]),
        p99=float(q[5]),
        max=int(counts.max()),
    )


class DataValidator:
    """Compute a full data-quality report over the temporal splits.

    Args:
        processed_dir: Directory holding ``train/val/test.parquet``.
        drift_dir: Directory holding the controlled-drift artifacts.
    """

    def __init__(
        self,
        processed_dir: str | Path = "data/processed",
        drift_dir: str | Path = "data/drift",
    ) -> None:
        """Store data locations and prepare the split resolver."""
        self.paths = SplitPaths(processed_dir)
        self.drift_dir = Path(drift_dir)
        self._frames: dict[Split, pd.DataFrame] = {}

    def load(self) -> None:
        """Load the three splits into memory once, reused by every check."""
        self._frames = load_all_splits(self.paths)

    # ---------------------------------------------------------- per split

    def _split_stats(self, split: Split) -> SplitStats:
        """Descriptive statistics for one split."""
        frame = self._frames[split]
        user_counts = frame[USER_IDX_COLUMN].value_counts().to_numpy()
        item_counts = frame[ITEM_IDX_COLUMN].value_counts().to_numpy()
        cats = frame[CATEGORY_COLUMN].value_counts().to_dict()
        return SplitStats(
            split=split.value,
            n_rows=int(len(frame)),
            n_users=int(frame[USER_IDX_COLUMN].nunique()),
            n_items=int(frame[ITEM_IDX_COLUMN].nunique()),
            timestamp_min=frame[TIMESTAMP_COLUMN].min().to_pydatetime(),
            timestamp_max=frame[TIMESTAMP_COLUMN].max().to_pydatetime(),
            rating_mean=float(frame[RATING_COLUMN].mean()),
            rating_std=float(frame[RATING_COLUMN].std()),
            category_counts={str(k): int(v) for k, v in cats.items()},
            user_activity=_activity_profile(user_counts, "user"),
            item_activity=_activity_profile(item_counts, "item"),
        )

    # ------------------------------------------------------- cross split

    def _cross_split(self) -> CrossSplitStats:
        """Cold-start and unseen-item rates between train and test."""
        train, test = self._frames[Split.TRAIN], self._frames[Split.TEST]
        train_users = set(train[USER_IDX_COLUMN].unique())
        train_items = set(train[ITEM_IDX_COLUMN].unique())
        test_users = set(test[USER_IDX_COLUMN].unique())
        test_items = set(test[ITEM_IDX_COLUMN].unique())

        seen_users = len(test_users & train_users)
        cold_users = len(test_users - train_users)
        unseen_items = len(test_items - train_items)
        return CrossSplitStats(
            test_users=len(test_users),
            test_users_seen_in_train=seen_users,
            test_cold_start_users=cold_users,
            test_cold_start_user_rate=cold_users / max(len(test_users), 1),
            test_items=len(test_items),
            test_items_unseen_in_train=unseen_items,
            test_new_item_rate=unseen_items / max(len(test_items), 1),
        )

    # ----------------------------------------------------------- leakage

    def _leakage(self) -> LeakageReport:
        """Temporal-boundary ordering and cross-split row duplication."""
        train, val, test = (
            self._frames[Split.TRAIN],
            self._frames[Split.VAL],
            self._frames[Split.TEST],
        )
        train_max = train[TIMESTAMP_COLUMN].max()
        val_min = val[TIMESTAMP_COLUMN].min()
        val_max = val[TIMESTAMP_COLUMN].max()
        test_min = test[TIMESTAMP_COLUMN].min()
        monotonic = bool(train_max <= val_min and val_max <= test_min)

        keys = pd.concat(
            [train[_KEY_COLUMNS], val[_KEY_COLUMNS], test[_KEY_COLUMNS]],
            ignore_index=True,
        )
        duplicate_rows = int(keys.duplicated().sum())
        return LeakageReport(
            train_max_ts=train_max.to_pydatetime(),
            val_min_ts=val_min.to_pydatetime(),
            val_max_ts=val_max.to_pydatetime(),
            test_min_ts=test_min.to_pydatetime(),
            boundaries_monotonic=monotonic,
            duplicate_rows_across_splits=duplicate_rows,
            is_clean=monotonic and duplicate_rows == 0,
        )

    # --------------------------------------------------- drift artifacts

    def _drift_ground_truth(self) -> DriftGroundTruthReport:
        """Validate the pre-generated control/drifted test streams."""
        test = self._frames[Split.TEST]
        gt = json.loads((self.drift_dir / "drift_ground_truth.json").read_text())
        control = pd.read_parquet(self.drift_dir / "test_control.parquet")
        drifted = pd.read_parquet(self.drift_dir / "test_drifted.parquet")

        onsets = self._check_onsets(gt, test)
        control_equals_test = test.reset_index(drop=True).equals(
            control.reset_index(drop=True)
        )
        c, d = control.reset_index(drop=True), drifted.reset_index(drop=True)
        row_diff = c.ne(d).any(axis=1)
        columns_changed = {
            col: int((c[col].to_numpy() != d[col].to_numpy()).sum())
            for col in c.columns
        }
        # The item *string* changed but its integer index did not → the drifted
        # frame no longer maps item_id to item_idx consistently.
        index_broken = (
            columns_changed.get(ITEM_COLUMN, 0) > 0
            and columns_changed.get(ITEM_IDX_COLUMN, 0) == 0
        )
        positions = np.where(row_diff.to_numpy())[0]
        span_start = int(positions.min()) if positions.size else -1
        span_end = int(positions.max()) if positions.size else -1
        return DriftGroundTruthReport(
            onsets=onsets,
            control_equals_test=control_equals_test,
            drifted_rows_changed=int(row_diff.sum()),
            columns_changed=columns_changed,
            index_consistency_broken=index_broken,
            changed_span_start=span_start,
            changed_span_end=span_end,
            usable_for_training=not index_broken,
        )

    @staticmethod
    def _check_onsets(
        gt: dict[str, Any], test: pd.DataFrame
    ) -> list[DriftOnsetCheck]:
        """Verify each ground-truth onset index against the test stream."""
        checks: list[DriftOnsetCheck] = []
        for name, spec in gt.items():
            idx = int(spec["idx"])
            in_range = 0 <= idx < len(test)
            gt_ts = pd.Timestamp(spec["timestamp"])
            test_ts = (
                test.iloc[idx][TIMESTAMP_COLUMN] if in_range else pd.Timestamp(0)
            )
            checks.append(
                DriftOnsetCheck(
                    name=name,
                    drift_type=str(spec["type"]),
                    idx=idx,
                    idx_in_range=in_range,
                    gt_timestamp=gt_ts.to_pydatetime(),
                    test_timestamp_at_idx=pd.Timestamp(test_ts).to_pydatetime(),
                    timestamp_matches=bool(in_range and gt_ts == pd.Timestamp(test_ts)),
                )
            )
        return checks

    # ------------------------------------------------------------- build

    def validate(self) -> DataValidationReport:
        """Run every check and assemble the report."""
        if not self._frames:
            self.load()
        splits = [self._split_stats(s) for s in Split]
        total = sum(s.n_rows for s in splits)
        fractions = {s.split: s.n_rows / total for s in splits}
        cross = self._cross_split()
        leakage = self._leakage()
        drift = self._drift_ground_truth()
        return DataValidationReport(
            total_rows=total,
            split_fractions=fractions,
            splits=splits,
            cross_split=cross,
            leakage=leakage,
            drift_ground_truth=drift,
            warnings=self._warnings(cross, leakage, drift),
        )

    @staticmethod
    def _warnings(
        cross: CrossSplitStats,
        leakage: LeakageReport,
        drift: DriftGroundTruthReport,
    ) -> list[str]:
        """Collect human-facing flags from the computed checks."""
        flags: list[str] = []
        if not leakage.is_clean:
            flags.append("temporal boundaries not clean (overlap or duplicate rows)")
        if cross.test_cold_start_user_rate > 0.5:
            flags.append(
                f"high cold-start rate in test: "
                f"{cross.test_cold_start_user_rate:.1%} of test users unseen in train"
            )
        if drift.index_consistency_broken:
            flags.append(
                "pre-generated test_drifted.parquet is index-inconsistent "
                "(item_id changed but item_idx unchanged) — not usable for training; "
                "controlled drift must be re-injected at the preference-sequence level"
            )
        if not all(o.timestamp_matches for o in drift.onsets):
            flags.append("some drift onset timestamps do not match the test stream")
        return flags
