"""Tests for the run record models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from drift_reco.observability.models import MetricRecord, RunConfig, slugify


def test_slugify_produces_a_path_safe_value() -> None:
    assert slugify("Ablation C — seuil KL") == "ablation-c-seuil-kl"
    assert slugify("!!!") == "run"


def test_run_config_rejects_a_task_with_uppercase_or_spaces() -> None:
    with pytest.raises(ValidationError):
        RunConfig(task="Ablation C", description="invalid task name")


def test_run_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        RunConfig(task="ok", description="d", unexpected=1)


def test_metric_record_row_matches_the_csv_schema() -> None:
    row = MetricRecord(metric="ndcg@10", value=0.0412, step=2).as_row()
    assert set(row) == {
        "timestamp",
        "step",
        "split",
        "metric",
        "value",
        "unit",
        "context",
    }
    assert row["step"] == "2"
