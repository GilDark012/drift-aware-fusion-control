"""Causal rolling statistics over a divergence series.

These summaries turn a raw ``D_u(t)`` trajectory into the quantities a detector
reasons about: a trailing mean/std baseline, a smoothed level, and a z-score of the
current value against its recent past. Every statistic is **causal** — computed from
values at or before ``t`` only — so nothing here can leak the future into a decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

__all__ = ["RollingConfig", "RollingSeries", "RollingStatistics"]

_EPS: Final[float] = 1e-8


class RollingConfig(BaseModel):
    """Window sizes for the rolling summaries."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    window: int = Field(default=20, ge=2)
    ewma_halflife: float = Field(default=10.0, gt=0.0)
    min_periods: int = Field(default=5, ge=1)


@dataclass(frozen=True)
class RollingSeries:
    """The causal statistics aligned to the input series (NaN during warm-up)."""

    mean: NDArray[np.float64]
    std: NDArray[np.float64]
    ewma: NDArray[np.float64]
    zscore: NDArray[np.float64]


class RollingStatistics:
    """Compute trailing mean/std, an EWMA level and a trailing z-score."""

    def __init__(self, config: RollingConfig | None = None) -> None:
        """Store the window configuration."""
        self.config = config or RollingConfig()

    def transform(self, values: NDArray[np.float64]) -> RollingSeries:
        """Return causal rolling statistics for a 1-D divergence series."""
        series = pd.Series(np.asarray(values, dtype=np.float64))
        window, min_periods = self.config.window, self.config.min_periods
        mean = series.rolling(window, min_periods=min_periods).mean()
        std = series.rolling(window, min_periods=min_periods).std(ddof=0)
        ewma = series.ewm(halflife=self.config.ewma_halflife, adjust=False).mean()
        zscore = (series - mean) / (std + _EPS)
        return RollingSeries(
            mean=mean.to_numpy(),
            std=std.to_numpy(),
            ewma=ewma.to_numpy(),
            zscore=zscore.to_numpy(),
        )
