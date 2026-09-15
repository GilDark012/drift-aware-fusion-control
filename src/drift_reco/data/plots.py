"""Diagnostic figures for the data-validation pass.

Kept separate from the validator so that report *numbers* and report *pictures*
can evolve independently. Every method returns a Matplotlib figure; persisting it
is the caller's responsibility (the LabJournal does this).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ..config.constants import (
    CATEGORY_COLUMN,
    TIMESTAMP_COLUMN,
    USER_IDX_COLUMN,
)

__all__ = ["DataValidationFigures"]


class DataValidationFigures:
    """Produce the standard diagnostic plots from loaded splits."""

    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        """Store split-name -> frame mapping (keys: train/val/test)."""
        self._frames = frames

    def user_activity_hist(self) -> Figure:
        """Log-scaled histogram of interactions per user, per split."""
        fig, ax = plt.subplots(figsize=(7, 4))
        for name, frame in self._frames.items():
            counts = frame[USER_IDX_COLUMN].value_counts().to_numpy()
            ax.hist(
                counts,
                bins=list(np.logspace(0, np.log10(counts.max() + 1), 40)),
                histtype="step",
                label=f"{name} (n={counts.size:,})",
                linewidth=1.5,
            )
        ax.set_xscale("log")
        ax.set_xlabel("interactions per user")
        ax.set_ylabel("number of users")
        ax.set_title("Per-user activity distribution")
        ax.legend()
        fig.tight_layout()
        return fig

    def category_share_over_time(self, freq: str = "M") -> Figure:
        """Monthly category share across all splits — the natural-drift context."""
        parts = []
        for name, frame in self._frames.items():
            sub = frame[[TIMESTAMP_COLUMN, CATEGORY_COLUMN]].copy()
            sub["split"] = name
            parts.append(sub)
        full = pd.concat(parts, ignore_index=True)
        full["period"] = full[TIMESTAMP_COLUMN].dt.to_period(freq).dt.to_timestamp()
        share = (
            full.groupby(["period", CATEGORY_COLUMN]).size().unstack(fill_value=0)
        )
        share = share.div(share.sum(axis=1), axis=0)

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.stackplot(
            share.index, share.to_numpy().T, labels=list(share.columns), alpha=0.85
        )
        ax.set_xlabel("month")
        ax.set_ylabel("category share of interactions")
        ax.set_title("Category mix over time (aggregate natural drift)")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left", fontsize=8, ncol=3)
        fig.tight_layout()
        return fig

    def rating_distribution(self) -> Figure:
        """Bar chart of the preference-score distribution on train."""
        from ..config.constants import RATING_COLUMN

        train = self._frames["train"]
        counts = train[RATING_COLUMN].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(counts.index.astype(str), counts.to_numpy(), color="#4C72B0")
        ax.set_xlabel("preference_score (rating)")
        ax.set_ylabel("count (train)")
        ax.set_title("Rating distribution")
        fig.tight_layout()
        return fig
