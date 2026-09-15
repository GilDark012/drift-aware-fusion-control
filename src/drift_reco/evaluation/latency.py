"""Latency measurements against a known injected onset.

Detection latency is the number of interactions between the true onset and the first
confirmation at or after it. A confirmation that fires *before* the onset is not a
detection of this shift but a premature/false alarm, and is counted separately so the
two never get conflated.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from ..detectors.detector import DriftState

__all__ = [
    "LatencyRecord",
    "DetectionLatencySummary",
    "detection_latency",
    "summarize_detection",
]


class LatencyRecord(BaseModel):
    """Per-user detection outcome relative to the known onset."""

    model_config = ConfigDict(extra="forbid")

    user: int
    onset_index: int
    detect_index: int | None
    detected: bool
    detection_latency: int | None
    premature: bool


class DetectionLatencySummary(BaseModel):
    """Aggregate detection behaviour over injected users."""

    model_config = ConfigDict(extra="forbid")

    n_users: int
    detection_rate: float
    premature_rate: float
    median_latency: float | None
    mean_latency: float | None


def detection_latency(
    user: int, states: NDArray[np.int64], onset_index: int
) -> LatencyRecord:
    """Detection latency of the first confirmation at or after the onset."""
    confirmed = np.where(states == int(DriftState.CONFIRMED))[0]
    premature = bool(confirmed.size and confirmed[0] < onset_index)
    after = confirmed[confirmed >= onset_index]
    if after.size:
        detect = int(after[0])
        return LatencyRecord(
            user=user,
            onset_index=onset_index,
            detect_index=detect,
            detected=True,
            detection_latency=detect - onset_index,
            premature=premature,
        )
    return LatencyRecord(
        user=user,
        onset_index=onset_index,
        detect_index=None,
        detected=False,
        detection_latency=None,
        premature=premature,
    )


def summarize_detection(records: list[LatencyRecord]) -> DetectionLatencySummary:
    """Reduce per-user records to detection-rate and latency statistics."""
    latencies = [
        r.detection_latency for r in records if r.detection_latency is not None
    ]
    detected = sum(1 for r in records if r.detected)
    premature = sum(1 for r in records if r.premature)
    n = max(len(records), 1)
    return DetectionLatencySummary(
        n_users=len(records),
        detection_rate=detected / n,
        premature_rate=premature / n,
        median_latency=float(np.median(latencies)) if latencies else None,
        mean_latency=float(np.mean(latencies)) if latencies else None,
    )
