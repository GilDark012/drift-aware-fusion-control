"""Tests for the validation report models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from drift_reco.data.validation_models import ActivityProfile, DriftOnsetCheck


def test_activity_profile_roundtrips() -> None:
    profile = ActivityProfile(
        entity="user",
        count=3,
        min=1,
        p05=1.0,
        p25=1.0,
        median=2.0,
        mean=2.0,
        p75=3.0,
        p95=3.0,
        p99=3.0,
        max=3,
    )
    assert profile.model_dump()["entity"] == "user"


def test_models_forbid_extra_fields() -> None:
    with pytest.raises(ValidationError):
        DriftOnsetCheck(
            name="D1",
            drift_type="sudden",
            idx=0,
            idx_in_range=True,
            gt_timestamp="2016-01-01T00:00:00",
            test_timestamp_at_idx="2016-01-01T00:00:00",
            timestamp_matches=True,
            unexpected="x",
        )
