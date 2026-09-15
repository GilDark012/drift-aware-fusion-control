"""Drift signals ``D_u(t)`` derived from the long/short representation trajectory.

Two families, both producing a per-step ``D_u(t)`` from a user's stream of
long/short representations:

* **short-term drift** (primary) — ``1 − cos(S_t, EWMA_slow(S))``: how far the current
  short-term state is from its own slowly-updated average. Both terms live in the GRU
  output space, so a *stable* user has ``D≈0`` and a genuine short-term shift raises
  ``D`` and keeps it raised until the slow average catches up. This is the long/short
  divergence the project studies, expressed where the two are actually comparable.
* **long-short divergence** (ablation) — ``dist(L_t, S_t)`` for a pointwise
  :class:`DivergenceMetric` (cosine/euclidean/sym-KL), the original §4 formulation.

Both are causal: the slow average excludes the current step.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from torch import Tensor

from .divergence import DivergenceMetric, get_divergence

__all__ = [
    "DriftSignal",
    "ShortTermDrift",
    "LongShortDivergence",
    "get_signal",
]

_EPS: Final[float] = 1e-8


class DriftSignal(ABC):
    """Map a trajectory of long/short representations to a per-step drift signal."""

    name: str

    @abstractmethod
    def compute(self, long: Tensor, short: Tensor) -> NDArray[np.float64]:
        """Return ``D_u(t)`` for stacked ``(N, d)`` long/short representations."""


class ShortTermDrift(DriftSignal):
    """``1 − cos(S_t, slow-EWMA(S))`` — short-term vs its own slow average."""

    name = "short_term_drift"

    def __init__(self, halflife: float = 30.0) -> None:
        """Store the half-life of the slow short-term average."""
        self.halflife = halflife

    def compute(self, long: Tensor, short: Tensor) -> NDArray[np.float64]:  # noqa: ARG002
        """Cosine distance of each short-term state from the slow average."""
        s = short.detach().cpu().numpy().astype(np.float64)
        slow = (
            pd.DataFrame(s)
            .ewm(halflife=self.halflife, adjust=False)
            .mean()
            .shift(1)
            .to_numpy()
        )
        num = (s * slow).sum(axis=1)
        den = np.linalg.norm(s, axis=1) * np.linalg.norm(np.nan_to_num(slow), axis=1)
        drift = 1.0 - num / (den + _EPS)
        drift[0] = 0.0
        return np.nan_to_num(drift)


class LongShortDivergence(DriftSignal):
    """Pointwise ``dist(L_t, S_t)`` for a divergence metric (the §4 ablation)."""

    def __init__(self, metric: DivergenceMetric) -> None:
        """Store the pointwise divergence metric and adopt its name."""
        self.metric = metric
        self.name = f"ls_{metric.name}"

    def compute(self, long: Tensor, short: Tensor) -> NDArray[np.float64]:
        """Pointwise divergence between the long- and short-term representations."""
        return self.metric(long, short).detach().cpu().numpy().astype(np.float64)


def get_signal(name: str, halflife: float = 30.0) -> DriftSignal:
    """Build a drift signal by name.

    ``short_term_drift`` (primary) or ``ls_<metric>`` for a long-short divergence
    ablation (e.g. ``ls_cosine``).

    Raises:
        KeyError: If the name is neither the short-term signal nor ``ls_<metric>``.
    """
    if name == ShortTermDrift.name:
        return ShortTermDrift(halflife)
    if name.startswith("ls_"):
        return LongShortDivergence(get_divergence(name[3:]))
    raise KeyError(f"unknown signal {name!r}; use 'short_term_drift' or 'ls_<metric>'")
