#!/usr/bin/env python3
"""Phase 2 — data-quality validation of the processed Amazon splits.

Thin entry point: parse arguments, run :class:`DataValidator` under a
:class:`LabJournal`, log every headline number as a metric, save the diagnostic
figures, and write the full typed report to the run directory and to ``docs/``.

Usage:
    python scripts/validate_data.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make ``src`` importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from drift_reco.data.plots import DataValidationFigures  # noqa: E402
from drift_reco.data.validation import DataValidator  # noqa: E402
from drift_reco.data.validation_models import DataValidationReport  # noqa: E402
from drift_reco.observability.lab_journal import LabJournal, RunConfig  # noqa: E402


def _log_report(journal: LabJournal, report: DataValidationReport) -> None:
    """Emit headline numbers from the report as journal metrics and events."""
    for split in report.splits:
        journal.metric("n_rows", split.n_rows, split=split.split, unit="rows")
        journal.metric("n_users", split.n_users, split=split.split, unit="users")
        journal.metric("n_items", split.n_items, split=split.split, unit="items")
        journal.metric(
            "median_interactions_per_user",
            split.user_activity.median,
            split=split.split,
        )
    cs = report.cross_split
    journal.metric(
        "test_cold_start_user_rate", cs.test_cold_start_user_rate, split="test"
    )
    journal.metric("test_new_item_rate", cs.test_new_item_rate, split="test")
    journal.metric(
        "duplicate_rows_across_splits",
        report.leakage.duplicate_rows_across_splits,
        unit="rows",
    )
    journal.event(
        "data",
        "leakage_check",
        level="INFO" if report.leakage.is_clean else "WARNING",
        detail=f"clean={report.leakage.is_clean}",
    )
    broken = report.drift_ground_truth.index_consistency_broken
    journal.event(
        "data",
        "drift_artifacts_check",
        level="WARNING" if broken else "INFO",
        detail=f"usable_for_training={report.drift_ground_truth.usable_for_training}",
    )
    for warning in report.warnings:
        journal.note(f"WARNING: {warning}")


def _save_figures(journal: LabJournal, validator: DataValidator) -> None:
    """Render and persist the diagnostic figures."""
    frames = {split.value: validator._frames[split] for split in validator._frames}
    figures = DataValidationFigures(frames)
    journal.figure(figures.user_activity_hist(), "user_activity_distribution")
    journal.figure(figures.category_share_over_time(), "category_share_over_time")
    journal.figure(figures.rating_distribution(), "rating_distribution")


def main() -> int:
    """Run the validation pass and return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--drift-dir", default="data/drift")
    parser.add_argument("--docs-out", default="docs/data_validation_report.json")
    args = parser.parse_args()

    config = RunConfig(
        task="phase2-data-validation",
        description="Certify processed Amazon splits: counts, leakage, drift GT",
        params={"processed_dir": args.processed_dir, "drift_dir": args.drift_dir},
        dataset="amazon-processed",
        tags=["phase2", "data-quality"],
    )

    with LabJournal(config) as journal:
        journal.event("data", "load_start", detail="train/val/test parquet")
        validator = DataValidator(args.processed_dir, args.drift_dir)
        validator.load()
        report = validator.validate()
        _log_report(journal, report)
        _save_figures(journal, validator)
        cold_rate = report.cross_split.test_cold_start_user_rate
        usable = report.drift_ground_truth.usable_for_training

        report_json = report.model_dump(mode="json")
        (journal.run_dir / "data" / "validation_report.json").write_text(
            json.dumps(report_json, indent=2, default=str), encoding="utf-8"
        )
        Path(args.docs_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.docs_out).write_text(
            json.dumps(report_json, indent=2, default=str), encoding="utf-8"
        )

        journal.summary(
            objective="Certify the frozen processed splits before any modelling.",
            method="Load train/val/test; compute per-split, cross-split, leakage and "
            "drift-ground-truth checks; render diagnostic figures.",
            findings=[
                f"total interactions: {report.total_rows:,}",
                f"split fractions: {report.split_fractions}",
                f"temporal boundaries clean: {report.leakage.is_clean}",
                f"test cold-start user rate: {cold_rate:.1%}",
                f"drift artifacts usable for training: {usable}",
            ]
            + [f"WARNING: {w}" for w in report.warnings],
            limitations=[
                "Validation operates on frozen parquet; ingestion code was not re-run.",
            ],
            next_steps=[
                "Re-implement controlled drift at preference-sequence level (Phase 7).",
                "Build the long/short-term recommender baseline (Phase 3).",
            ],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
