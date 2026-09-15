"""Threshold strategies that turn a divergence series into an elevation score.

The detector reasons about a single *score* per step and compares it to enter/exit
levels. How that score is derived is the threshold strategy — the axis of the §14
ablation and hypothesis H6 (does a per-user adaptive threshold beat a global one?):

* **global** — score is the raw divergence ``D``; a single tuned level applies to
  everyone, so naturally volatile users trip more often;
* **adaptive** — score is the causal z-score of ``D`` against the user's own trailing
  baseline, so the level means the same thing for calm and volatile users alike.

Both are causal: the score at ``t`` depends only on values at or before ``t``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Final

import numpy as np
from numpy.typing import NDArray

from .rolling import RollingSeries

__all__ = [
    "ThresholdStrategy",
    "GlobalThreshold",
    "AdaptiveThreshold",
    "THRESHOLD_STRATEGIES",
    "get_threshold",
]


class ThresholdStrategy(ABC):
    """Map a divergence series (and its rolling stats) to an elevation score."""

    name: str

    @abstractmethod
    def score(
        self, divergence: NDArray[np.float64], rolling: RollingSeries
    ) -> NDArray[np.float64]:
        """Return the per-step score the detector thresholds."""


class GlobalThreshold(ThresholdStrategy):
    """Score is the raw divergence — one global level for every user."""

    name = "global"

    def score(
        self, divergence: NDArray[np.float64], rolling: RollingSeries  # noqa: ARG002
    ) -> NDArray[np.float64]:
        """Return the raw divergence as the score (``rolling`` unused here)."""
        return np.asarray(divergence, dtype=np.float64)


class AdaptiveThreshold(ThresholdStrategy):
    """Score is the causal z-score against the user's own trailing baseline."""

    name = "adaptive"

    def score(
        self, divergence: NDArray[np.float64], rolling: RollingSeries  # noqa: ARG002
    ) -> NDArray[np.float64]:
        """Return the rolling z-score as the score (``divergence`` unused here)."""
        return rolling.zscore


THRESHOLD_STRATEGIES: Final[dict[str, type[ThresholdStrategy]]] = {
    GlobalThreshold.name: GlobalThreshold,
    AdaptiveThreshold.name: AdaptiveThreshold,
}


def get_threshold(name: str) -> ThresholdStrategy:
    """Instantiate a threshold strategy by name.

    Raises:
        KeyError: If ``name`` is not a registered strategy.
    """
    if name not in THRESHOLD_STRATEGIES:
        known = list(THRESHOLD_STRATEGIES)
        raise KeyError(f"unknown threshold {name!r}; choose from {known}")
    return THRESHOLD_STRATEGIES[name]()
