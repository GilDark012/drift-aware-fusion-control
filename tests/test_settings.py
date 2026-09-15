"""Tests for the configuration models."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from drift_reco.config.settings import DetectionSettings, ExperimentSettings


def test_experiment_settings_accepts_a_minimal_configuration() -> None:
    settings = ExperimentSettings(
        name="ablation-c",
        data={"parquet_path": Path("data/processed/amazon.parquet")},
    )
    assert settings.detection.votes_required == 2
    assert settings.top_k == 10


def test_detection_settings_rejects_an_unreachable_quorum() -> None:
    with pytest.raises(ValidationError):
        DetectionSettings(detectors=("kl",), votes_required=2)


def test_settings_are_frozen() -> None:
    settings = DetectionSettings()
    with pytest.raises(ValidationError):
        settings.votes_required = 3
