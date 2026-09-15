"""Figures for the controlled-drift experiments.

An injection overlay shows a single user's divergence with the **known** onset and
the detector's confirmation, so detection latency is visible as the gap between the
green (truth) and red (detection) lines.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ..detectors.detector import DriftState
from ..detectors.rolling import RollingStatistics
from .recovery import RecoveryCurve
from .streaming import UserRunResult

__all__ = [
    "injection_overlay",
    "recovery_curves",
    "detection_tradeoff",
    "detector_comparison",
]

_PALETTE = ("#4C72B0", "#C44E52", "#55A868", "#8172B3", "#CCB974")


def detector_comparison(curves: dict[str, list[tuple[float, float]]]) -> Figure:
    """Overlay several detectors' detection-vs-false-alarm operating curves.

    Each entry maps a detector name to a list of ``(false_alarm, detection)`` points
    (its threshold sweep). The chance diagonal is drawn for reference.
    """
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for (name, points), colour in zip(curves.items(), _PALETTE, strict=False):
        ordered = sorted(points)
        fa = [p[0] for p in ordered]
        det = [p[1] for p in ordered]
        ax.plot(fa, det, "o-", ms=4, lw=1.5, label=name, color=colour)
    ax.plot([0, 1], [0, 1], ls=":", lw=1.0, color="#999999", label="chance")
    ax.set_xlabel("false-alarm rate (control streams)")
    ax.set_ylabel("detection rate (injected streams)")
    ax.set_title("Drift detectors compared (same divergence signal)")
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def detection_tradeoff(points: list[tuple[float, float, float]]) -> Figure:
    """Detection rate vs false-alarm rate as the detector threshold varies."""
    enters = [p[0] for p in points]
    detection = [p[1] for p in points]
    false_alarm = [p[2] for p in points]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(false_alarm, detection, "o-", color="#4C72B0")
    for enter, det, fa in zip(enters, detection, false_alarm, strict=True):
        ax.annotate(f"z={enter:g}", (fa, det), fontsize=8,
                    textcoords="offset points", xytext=(5, -2))
    ax.set_xlabel("false-alarm rate (control streams)")
    ax.set_ylabel("detection rate (injected streams)")
    ax.set_title("Detector operating characteristic (threshold sweep)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def recovery_curves(curves: dict[str, RecoveryCurve]) -> Figure:
    """Overlay quality-vs-time-since-onset curves for several α-policies."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, curve in curves.items():
        ax.plot(curve.rel_pos, curve.hit_rate, marker="o", ms=3, lw=1.4, label=name)
    ax.axvline(0, color="#55A868", ls="--", lw=1.2, label="onset")
    ax.set_xlabel("interactions since onset")
    ax.set_ylabel("reciprocal rank (MRR)")
    ax.set_title("Recommendation quality around the drift onset, by policy")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def injection_overlay(result: UserRunResult, onset_index: int) -> Figure:
    """Plot divergence with the known onset (green) and the detection (red)."""
    rolling = RollingStatistics().transform(result.divergence)
    t = result.timestamp
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, result.divergence, color="#B0B0B0", lw=0.8, label="D(t)")
    ax.plot(t, rolling.mean, color="#4C72B0", lw=1.6, label="rolling mean")
    confirmed = result.state == int(DriftState.CONFIRMED)
    ax.fill_between(
        t, 0, 1, where=confirmed, transform=ax.get_xaxis_transform(),
        color="#C44E52", alpha=0.12,
    )
    if 0 <= onset_index < t.size:
        ax.axvline(t[onset_index], color="#55A868", ls="--", lw=1.4, label="true onset")
    detections = [i for i, c in enumerate(confirmed) if c]
    if detections:
        ax.axvline(t[detections[0]], color="#C44E52", ls="-", lw=1.4, label="detected")
    ax.set_xlabel("time")
    ax.set_ylabel("divergence D(L,S)")
    ax.set_title(f"User {result.user}: injected shift — onset vs detection")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig
