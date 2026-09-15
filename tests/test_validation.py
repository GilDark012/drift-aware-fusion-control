"""Tests for the data-quality validator on synthetic splits."""

from __future__ import annotations

from pathlib import Path

from drift_reco.data.validation import DataValidator


def test_validator_reports_clean_boundaries(synthetic_dataset: Path) -> None:
    report = DataValidator(
        synthetic_dataset / "processed", synthetic_dataset / "drift"
    ).validate()
    assert report.leakage.is_clean
    assert report.leakage.duplicate_rows_across_splits == 0
    assert report.total_rows == 12


def test_validator_detects_cold_start_user(synthetic_dataset: Path) -> None:
    report = DataValidator(
        synthetic_dataset / "processed", synthetic_dataset / "drift"
    ).validate()
    # u4 appears only in test.
    assert report.cross_split.test_cold_start_users == 1
    assert report.cross_split.test_items_unseen_in_train >= 1


def test_validator_flags_broken_drift_artifacts(synthetic_dataset: Path) -> None:
    report = DataValidator(
        synthetic_dataset / "processed", synthetic_dataset / "drift"
    ).validate()
    gt = report.drift_ground_truth
    assert gt.control_equals_test
    assert gt.index_consistency_broken
    assert gt.usable_for_training is False
    assert any("index-inconsistent" in w for w in report.warnings)


def test_validator_matches_onset_timestamp(synthetic_dataset: Path) -> None:
    report = DataValidator(
        synthetic_dataset / "processed", synthetic_dataset / "drift"
    ).validate()
    assert all(o.timestamp_matches for o in report.drift_ground_truth.onsets)
