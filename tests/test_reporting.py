"""Tests for the reporting layer."""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path

from drift_reco.observability.models import METRIC_COLUMNS, RunManifest
from drift_reco.observability.reporting import ResultsRenderer, RunIndex


def _manifest() -> RunManifest:
    started = datetime(2026, 1, 1, tzinfo=UTC)
    return RunManifest(
        run_id="20260101-000000__demo",
        task="demo",
        description="Demo run",
        status="completed",
        started_at=started,
        finished_at=started,
        duration_seconds=1.5,
    )


def test_renderer_writes_a_readable_report(tmp_path: Path) -> None:
    with (tmp_path / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(METRIC_COLUMNS))
        writer.writeheader()
        writer.writerow(
            {
                "timestamp": "t",
                "step": "0",
                "split": "test",
                "metric": "ndcg@10",
                "value": "0.04",
                "unit": "",
                "context": "",
            }
        )

    path = ResultsRenderer(tmp_path).render(
        _manifest(),
        {"alpha": 0.3},
        {"objective": "o", "method": "m", "findings": ["f"]},
        ["note"],
        [],
    )
    content = path.read_text(encoding="utf-8")
    assert "ndcg@10" in content
    assert "## Findings" in content


def test_index_creates_the_csv_and_the_journal(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    RunIndex(artifacts, tmp_path).append(_manifest())

    with (artifacts / "index.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["run_id"] == "20260101-000000__demo"
    assert "demo" in (tmp_path / "JOURNAL.md").read_text(encoding="utf-8")
