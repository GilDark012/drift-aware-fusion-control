"""Figures for adaptive fusion: how α(t) tracks the drift signal.

One view shows a single policy — divergence with the confirmed region shaded and
α(t) on a second axis, so the reader sees α fall as a shift is confirmed and recover
afterwards. Another overlays several policies' α(t) on the same user to contrast
gradual, abrupt and detector-free adaptation (H4).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ..detectors.detector import DriftState
from ..detectors.rolling import RollingStatistics
from ..evaluation.streaming import UserRunResult

__all__ = ["alpha_trajectory", "alpha_policy_comparison"]


def alpha_trajectory(result: UserRunResult) -> Figure:
    """Plot one user's divergence and α(t) with the confirmed region shaded."""
    rolling = RollingStatistics().transform(result.divergence)
    t = result.timestamp
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, result.divergence, color="#B0B0B0", lw=0.8, label="D(t)")
    ax.plot(t, rolling.mean, color="#4C72B0", lw=1.5, label="rolling mean")
    confirmed = result.state == int(DriftState.CONFIRMED)
    ax.fill_between(
        t, 0, 1, where=confirmed, transform=ax.get_xaxis_transform(),
        color="#C44E52", alpha=0.15, label="confirmed",
    )
    ax.set_xlabel("time")
    ax.set_ylabel("divergence D(L,S)")
    ax2 = ax.twinx()
    ax2.plot(t, result.alpha, color="#DD8452", lw=1.8, label="α(t)")
    ax2.set_ylabel("fusion weight α")
    ax2.set_ylim(0, 1)
    ax.set_title(f"User {result.user}: α(t) falls as drift is confirmed")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def alpha_policy_comparison(results: dict[str, UserRunResult]) -> Figure:
    """Overlay α(t) of several policies on one user, with divergence beneath."""
    any_result = next(iter(results.values()))
    t = any_result.timestamp
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(9, 5), sharex=True, height_ratios=[2, 1]
    )
    for name, result in results.items():
        top.plot(t, result.alpha, lw=1.6, label=name)
    top.set_ylabel("fusion weight α")
    top.set_ylim(0, 1)
    top.set_title(f"User {any_result.user}: α(t) by policy (gradual vs abrupt vs D)")
    top.legend(fontsize=8, loc="lower left")
    bottom.plot(t, any_result.divergence, color="#B0B0B0", lw=0.8)
    bottom.set_ylabel("D(t)")
    bottom.set_xlabel("time")
    fig.tight_layout()
    return fig
