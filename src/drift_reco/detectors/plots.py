"""Diagnostic figures for the divergence signal.

Two views: a single user's ``D_u(t)`` trajectory with its rolling summaries and
category-change markers, and a population view showing that divergence responds to
category changes (the interpretable indicator) without being defined by them.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from .analysis import DetectionSummary
from .detector import DetectionResult, DriftState
from .rolling import RollingStatistics
from .trajectory import UserDivergence

__all__ = [
    "DivergenceFigures",
    "persistence_sweep_figure",
    "strategy_stratum_figure",
]


def persistence_sweep_figure(
    sweep: list[tuple[int, DetectionSummary]],
) -> Figure:
    """Plot confirmed/temporary alarm rates against the persistence requirement (H5)."""
    persistences = [p for p, _ in sweep]
    confirmed = [s.confirmed_per_1k for _, s in sweep]
    temporary = [s.temporary_per_1k for _, s in sweep]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(persistences, confirmed, "o-", color="#C44E52", label="confirmed / 1k")
    ax.plot(persistences, temporary, "s--", color="#8172B3", label="temporary / 1k")
    ax.set_xlabel("persistence requirement (elevated steps)")
    ax.set_ylabel("alarms per 1000 steps")
    ax.set_title("Persistence suppresses false alarms (H5)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def strategy_stratum_figure(table: pd.DataFrame) -> Figure:
    """Grouped bars of confirmed rate by volatility stratum and strategy (H6)."""
    strata = sorted(table["stratum"].unique())
    strategies = list(table["strategy"].unique())
    width = 0.8 / max(len(strategies), 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, strat in enumerate(strategies):
        sub = table[table["strategy"] == strat].set_index("stratum")
        rates = [float(sub.loc[s, "confirmed_per_1k"]) for s in strata]
        positions = [s + i * width for s in range(len(strata))]
        ax.bar(positions, rates, width=width, label=strat)
    ax.set_xticks([s + width * (len(strategies) - 1) / 2 for s in range(len(strata))])
    ax.set_xticklabels([f"stratum {s}" for s in strata])
    ax.set_ylabel("confirmed / 1000 steps")
    ax.set_title("Adaptive threshold is more uniform across volatility (H6)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


class DivergenceFigures:
    """Render divergence diagnostics from trajectories and population tables."""

    def __init__(self, rolling: RollingStatistics | None = None) -> None:
        """Store the rolling-statistics helper used to overlay summaries."""
        self.rolling = rolling or RollingStatistics()

    def trajectory(self, user_div: UserDivergence) -> Figure:
        """Plot one user's divergence with rolling mean/EWMA and change markers."""
        stats = self.rolling.transform(user_div.divergence)
        t = user_div.timestamp
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(t, user_div.divergence, color="#B0B0B0", lw=0.8, label="D(t) raw")
        ax.plot(t, stats.mean, color="#4C72B0", lw=1.8, label="rolling mean")
        ax.plot(t, stats.ewma, color="#DD8452", lw=1.3, label="EWMA")
        for x in t[user_div.category_changed()]:
            ax.axvline(x, color="#C44E52", alpha=0.25, lw=0.8)
        ax.set_xlabel("time")
        ax.set_ylabel("divergence D(L,S)")
        ax.set_title(f"User {user_div.user}: long/short divergence over time")
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        return fig

    def population_category_effect(self, population: pd.DataFrame) -> Figure:
        """Compare divergence at category-change vs stable positions."""
        changed = population.loc[population["category_changed"], "divergence"]
        stable = population.loc[~population["category_changed"], "divergence"]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(
            [stable.to_numpy(), changed.to_numpy()],
            tick_labels=["same category", "category changed"],
            showfliers=False,
        )
        ax.set_ylabel("divergence D(L,S)")
        ax.set_title("Divergence responds to category change (indicator)")
        means = [float(stable.mean()), float(changed.mean())]
        ax.plot([1, 2], means, "o-", color="#DD8452", label="mean")
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    def detection_overlay(
        self, user_div: UserDivergence, result: DetectionResult
    ) -> Figure:
        """Plot divergence with detector states and onset/detect markers."""
        stats = self.rolling.transform(user_div.divergence)
        t = user_div.timestamp
        states = result.states
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(t, user_div.divergence, color="#B0B0B0", lw=0.8, label="D(t)")
        ax.plot(t, stats.mean, color="#4C72B0", lw=1.6, label="rolling mean")
        confirmed = states == int(DriftState.CONFIRMED)
        ax.fill_between(
            t, 0, 1, where=confirmed, transform=ax.get_xaxis_transform(),
            color="#C44E52", alpha=0.15, label="confirmed",
        )
        for drift in result.confirmed:
            ax.axvline(t[drift.onset_step], color="#55A868", ls="--", lw=1.0)
            ax.axvline(t[drift.detect_step], color="#C44E52", ls="-", lw=1.2)
        temporary = states == int(DriftState.TEMPORARY)
        ax.scatter(
            t[temporary], user_div.divergence[temporary],
            color="#8172B3", s=14, zorder=5, label="temporary",
        )
        ax.set_xlabel("time")
        ax.set_ylabel("divergence D(L,S)")
        ax.set_title(f"User {user_div.user}: detector states (onset --, detect |)")
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        return fig

    def divergence_distribution(self, population: pd.DataFrame) -> Figure:
        """Histogram of the raw divergence values across the population."""
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(population["divergence"].to_numpy(), bins=50, color="#4C72B0")
        ax.set_xlabel("divergence D(L,S)")
        ax.set_ylabel("count (positions)")
        ax.set_title("Population divergence distribution")
        fig.tight_layout()
        return fig
