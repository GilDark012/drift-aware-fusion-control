"""Typed results of the data-validation pass.

These models are the schema of the data-quality report. They are serialised to
``config.json``/``results`` artifacts and re-read by tests and notebooks, so any
change to what we measure is a change to one of these classes.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "SplitStats",
    "ActivityProfile",
    "CrossSplitStats",
    "LeakageReport",
    "DriftOnsetCheck",
    "DriftGroundTruthReport",
    "DataValidationReport",
]


class ActivityProfile(BaseModel):
    """Distribution summary of interactions-per-entity (user or item)."""

    model_config = ConfigDict(extra="forbid")

    entity: str
    count: int
    min: int
    p05: float
    p25: float
    median: float
    mean: float
    p75: float
    p95: float
    p99: float
    max: int


class SplitStats(BaseModel):
    """Per-split descriptive statistics."""

    model_config = ConfigDict(extra="forbid")

    split: str
    n_rows: int
    n_users: int
    n_items: int
    timestamp_min: datetime
    timestamp_max: datetime
    rating_mean: float
    rating_std: float
    category_counts: dict[str, int]
    user_activity: ActivityProfile
    item_activity: ActivityProfile


class CrossSplitStats(BaseModel):
    """How users and items carry across the temporal boundaries."""

    model_config = ConfigDict(extra="forbid")

    test_users: int
    test_users_seen_in_train: int
    test_cold_start_users: int
    test_cold_start_user_rate: float
    test_items: int
    test_items_unseen_in_train: int
    test_new_item_rate: float


class LeakageReport(BaseModel):
    """Checks that no information crosses a temporal boundary illegitimately."""

    model_config = ConfigDict(extra="forbid")

    train_max_ts: datetime
    val_min_ts: datetime
    val_max_ts: datetime
    test_min_ts: datetime
    boundaries_monotonic: bool
    duplicate_rows_across_splits: int
    is_clean: bool


class DriftOnsetCheck(BaseModel):
    """Sanity check of a single ground-truth drift onset."""

    model_config = ConfigDict(extra="forbid")

    name: str
    drift_type: str
    idx: int
    idx_in_range: bool
    gt_timestamp: datetime
    test_timestamp_at_idx: datetime
    timestamp_matches: bool


class DriftGroundTruthReport(BaseModel):
    """Validation of the pre-generated controlled-drift artifacts."""

    model_config = ConfigDict(extra="forbid")

    onsets: list[DriftOnsetCheck]
    control_equals_test: bool
    drifted_rows_changed: int
    columns_changed: dict[str, int]
    index_consistency_broken: bool
    changed_span_start: int
    changed_span_end: int
    usable_for_training: bool


class DataValidationReport(BaseModel):
    """Complete data-quality report for the processed Amazon splits."""

    model_config = ConfigDict(extra="forbid")

    total_rows: int
    split_fractions: dict[str, float]
    splits: list[SplitStats]
    cross_split: CrossSplitStats
    leakage: LeakageReport
    drift_ground_truth: DriftGroundTruthReport
    warnings: list[str] = Field(default_factory=list)
