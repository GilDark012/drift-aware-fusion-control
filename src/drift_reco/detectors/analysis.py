"""Aggregate detector behaviour across many users, for validation-set tuning.

Without injected ground truth (that arrives in Phase 7), the validation quantity we
control is the **alarm rate**: how often the detector confirms a shift and how often
it flags a temporary deviation, per 1000 scored steps. Persistence should drive the
confirmed rate down (H5); an adaptive threshold should keep the rate more uniform
across users of differing volatility than a global one (H6).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from .detector import DetectionResult, DetectorConfig, PersistenceDriftDetector

__all__ = [
    "DetectionSummary",
    "run_detector_over",
    "summarize",
    "persistence_sweep",
    "volatility_strata",
]


class DetectionSummary(BaseModel):
    """Alarm-rate summary over a set of per-user detection results."""

    model_config = ConfigDict(extra="forbid")

    n_users: int
    total_steps: int
    n_confirmed: int
    n_temporary: int
    confirmed_per_1k: float
    temporary_per_1k: float
    users_with_confirm_frac: float


def run_detector_over(
    series: list[NDArray[np.float64]], config: DetectorConfig
) -> list[DetectionResult]:
    """Run one detector configuration over many per-user divergence series."""
    detector = PersistenceDriftDetector(config)
    return [detector.run(s) for s in series]


def summarize(results: list[DetectionResult]) -> DetectionSummary:
    """Reduce per-user results to an alarm-rate summary."""
    total_steps = sum(int(r.states.size) for r in results)
    n_confirmed = sum(r.n_confirmed for r in results)
    n_temporary = sum(r.n_temporary for r in results)
    with_confirm = sum(1 for r in results if r.n_confirmed > 0)
    scale = 1000.0 / max(total_steps, 1)
    return DetectionSummary(
        n_users=len(results),
        total_steps=total_steps,
        n_confirmed=n_confirmed,
        n_temporary=n_temporary,
        confirmed_per_1k=n_confirmed * scale,
        temporary_per_1k=n_temporary * scale,
        users_with_confirm_frac=with_confirm / max(len(results), 1),
    )


def persistence_sweep(
    series: list[NDArray[np.float64]],
    config: DetectorConfig,
    persistences: list[int],
) -> list[tuple[int, DetectionSummary]]:
    """Summarise the alarm rate at each persistence value (H5)."""
    out: list[tuple[int, DetectionSummary]] = []
    for persistence in persistences:
        cfg = config.model_copy(update={"persistence": persistence})
        out.append((persistence, summarize(run_detector_over(series, cfg))))
    return out


def volatility_strata(
    series: list[NDArray[np.float64]], n_strata: int = 3
) -> NDArray[np.int64]:
    """Assign each user's series to a volatility (divergence-std) stratum."""
    stds = np.array([float(np.nanstd(s)) if s.size else 0.0 for s in series])
    ranks = pd.qcut(stds, n_strata, labels=False, duplicates="drop")
    return np.asarray(ranks, dtype=np.int64)
