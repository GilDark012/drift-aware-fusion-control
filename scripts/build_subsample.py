#!/usr/bin/env python3
"""Build the reproducible development subsample (Phase 3 support).

Usage:
    python scripts/build_subsample.py --target-users 3000 --min-test 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from drift_reco.data.subsample import (  # noqa: E402
    SubsampleBuilder,
    SubsampleConfig,
)
from drift_reco.observability.lab_journal import LabJournal, RunConfig  # noqa: E402


def main() -> int:
    """Build and journal the development subsample."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-users", type=int, default=3000)
    parser.add_argument("--min-test", type=int, default=20)
    parser.add_argument("--extra-train-users", type=int, default=20000)
    parser.add_argument("--k-core", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sub_cfg = SubsampleConfig(
        seed=args.seed,
        target_users=args.target_users,
        min_test_interactions=args.min_test,
        extra_train_users=args.extra_train_users,
        k_core=args.k_core,
    )
    config = RunConfig(
        task="phase3-build-subsample",
        description="Stratified evaluable-cohort subsample for fast iteration",
        params=sub_cfg.model_dump(mode="json"),
        dataset="amazon-processed",
        tags=["phase3", "subsample"],
        seed=args.seed,
    )

    with LabJournal(config) as journal:
        journal.event("data", "subsample_start", detail=str(sub_cfg.out_dir))
        manifest = SubsampleBuilder(sub_cfg).build()
        for split, rows in manifest.rows_per_split.items():
            journal.metric("subsample_rows", rows, split=split, unit="rows")
            journal.metric(
                "unseen_item_rate", manifest.unseen_item_rate[split], split=split
            )
        journal.metric("selected_users", manifest.selected_users, unit="users")
        journal.metric("train_users_total", manifest.train_users_total, unit="users")
        journal.metric("item_vocab", manifest.n_item_vocab, unit="items")
        journal.note(
            f"cohort pool (>= {args.min_test} test interactions): "
            f"{manifest.cohort_pool_size} users; cohort {manifest.selected_users}; "
            f"train users {manifest.train_users_total}"
        )
        journal.summary(
            objective="Provide a small, faithful subsample for pipeline iteration.",
            method="Select test-active cohort users (stratified by activity), keep "
            "full cross-split history, re-index densely with train-only item vocab.",
            findings=[
                f"eval cohort: {manifest.selected_users} users "
                f"(pool {manifest.cohort_pool_size})",
                f"train users incl. extra: {manifest.train_users_total}",
                f"item vocab: {manifest.n_item_vocab}",
                f"rows: {manifest.rows_per_split}",
                f"unseen-item rate: {manifest.unseen_item_rate}",
            ],
            next_steps=["Train the fixed-α baseline on this subsample."],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
