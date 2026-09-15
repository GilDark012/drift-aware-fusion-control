"""Validated configuration models.

Every parameter that reaches the pipeline passes through one of these models, so a
typo in a configuration file fails immediately and with a readable message rather
than silently changing an experiment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .constants import DEFAULT_SEED, DEFAULT_TOP_K

DetectorName = Literal["pudd", "kl", "performance"]
StrategyName = Literal["alpha_shift", "background_retrain", "incremental_update"]


class DataSettings(BaseModel):
    """Where the data lives and how it is windowed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    parquet_path: Path
    window_size: int = Field(default=5_000, gt=0)
    min_interactions_per_user: int = Field(default=5, ge=1)
    test_fraction: float = Field(default=0.2, gt=0, lt=1)


class DetectionSettings(BaseModel):
    """Which detectors vote and how their thresholds are obtained."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    detectors: tuple[DetectorName, ...] = ("pudd", "kl", "performance")
    votes_required: int = Field(default=2, ge=1)
    dynamic_thresholds: bool = True
    warmup_windows: int = Field(default=3, ge=0)

    @model_validator(mode="after")
    def _validate_quorum(self) -> DetectionSettings:
        """Reject a quorum that no combination of detectors could ever reach."""
        if self.votes_required > len(self.detectors):
            raise ValueError(
                "votes_required cannot exceed the number of active detectors"
            )
        return self


class AdaptationSettings(BaseModel):
    """How the recommender reacts once drift is confirmed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: StrategyName = "alpha_shift"
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    alpha_step: float = Field(default=0.1, gt=0.0, le=1.0)
    cooldown_windows: int = Field(default=2, ge=0)


class ExperimentSettings(BaseModel):
    """Full description of one experimental condition."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, max_length=64)
    data: DataSettings
    detection: DetectionSettings = DetectionSettings()
    adaptation: AdaptationSettings = AdaptationSettings()
    top_k: int = Field(default=DEFAULT_TOP_K, gt=0)
    seed: int = DEFAULT_SEED
    repetitions: int = Field(default=5, ge=1)
