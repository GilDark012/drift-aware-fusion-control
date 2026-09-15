"""Tests for the run journaling library."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from drift_reco.observability.lab_journal import LabJournal, RunConfig


def _config() -> RunConfig:
    return RunConfig(task="unit-test", description="Journaling smoke test")


def test_journal_writes_every_expected_artifact(tmp_path: Path) -> None:
    with LabJournal(_config(), root=tmp_path / "artifacts", console=False) as journal:
        journal.metric("ndcg@10", 0.0412, split="test")
        journal.event("data", "loaded", detail="42 rows")
        journal.note("Sparse data, low absolute values are expected.")
        journal.summary(objective="o", method="m", findings=["f"])
        run_dir = journal.run_dir

    for name in (
        "config.json",
        "metrics.csv",
        "events.csv",
        "run.log",
        "results.md",
        "manifest.json",
        "notes.txt",
    ):
        assert (run_dir / name).is_file(), name

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["metric_count"] == 1


def test_metrics_csv_has_one_row_per_measurement(tmp_path: Path) -> None:
    with LabJournal(_config(), root=tmp_path / "artifacts", console=False) as journal:
        for step in range(3):
            journal.metric("recall@10", 0.1 * step, step=step)
        run_dir = journal.run_dir

    with (run_dir / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert rows[2]["step"] == "2"


def test_failed_run_is_documented_and_reraises(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    with (
        pytest.raises(RuntimeError),
        LabJournal(_config(), root=root, console=False) as journal,
    ):
        run_dir = journal.run_dir
        raise RuntimeError("boom")

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert "boom" in manifest["error"]


def test_run_is_appended_to_the_index(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    with LabJournal(_config(), root=root, console=False):
        pass
    with (root / "index.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["task"] == "unit-test"
