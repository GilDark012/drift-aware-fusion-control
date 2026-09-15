"""Classical concept-drift detectors as external baselines.

The review asks us to compare our persistence-aware state machine against the standard
concept-drift literature rather than only against static-α ablations. These are
faithful, dependency-free implementations of three canonical detectors, run on the same
per-user divergence series ``D_u(t)`` our detector consumes, so the comparison is
apples-to-apples (same signal, same evaluation, only the decision rule differs):

* **Page-Hinkley** (Page, 1954) — cumulative deviation of the signal from its running
  mean; alarms on a sustained *increase*;
* **CUSUM** (Page, 1954) — one-sided cumulative sum against a self-estimated reference
  with slack ``k`` and threshold ``h``;
* **ADWIN0** (Bifet & Gavaldà, 2007) — adaptive windowing: keep a window, and when two
  sub-windows' means differ by more than a Hoeffding bound, drop the old part and flag a
  change. This is the exact O(n²) variant from the paper (per-user streams are short).

Each detector returns a ``states`` array using the same :class:`DriftState` encoding as
the main detector (``CONFIRMED`` at every alarm step), so the existing latency /
false-alarm evaluation applies unchanged. DDM/EDDM (Gama et al., 2004) are deliberately
omitted: they monitor a *classifier error* stream, not a continuous divergence, and do
not apply to this signal without an arbitrary binarisation — we cite them but do not
force the fit.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Final

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from .detector import ConfirmedDrift, DetectionResult, DriftState
from .rolling import RollingConfig

__all__ = [
    "BaselineDetector",
    "PageHinkleyConfig",
    "PageHinkley",
    "CusumConfig",
    "Cusum",
    "AdwinConfig",
    "Adwin",
    "CusumDetectorConfig",
    "CusumDriftDetector",
    "BASELINE_DETECTORS",
    "build_baseline",
]

_EPS: Final[float] = 1e-8


def _confirmed_states(size: int, alarms: list[int]) -> NDArray[np.int64]:
    """Encode alarm step indices as a CONFIRMED-marked states array."""
    states = np.zeros(size, dtype=np.int64)
    for idx in alarms:
        states[idx] = int(DriftState.CONFIRMED)
    return states


class BaselineDetector(ABC):
    """A classical drift detector over a 1-D real-valued signal."""

    name: str

    @abstractmethod
    def run(self, divergence: NDArray[np.float64]) -> NDArray[np.int64]:
        """Return a per-step states array (CONFIRMED at alarm steps)."""


class PageHinkleyConfig(BaseModel):
    """Page-Hinkley parameters (increase-detecting form)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    delta: float = Field(default=0.005, ge=0.0)
    threshold: float = Field(default=0.5, gt=0.0)
    warmup: int = Field(default=3, ge=0)


class PageHinkley(BaselineDetector):
    """Page-Hinkley test for a sustained increase in the signal mean."""

    name = "page-hinkley"

    def __init__(self, config: PageHinkleyConfig | None = None) -> None:
        """Store the configuration."""
        self.config = config or PageHinkleyConfig()

    def run(self, divergence: NDArray[np.float64]) -> NDArray[np.int64]:
        """Alarm when the cumulative upward deviation exceeds ``threshold``."""
        cfg = self.config
        values = np.asarray(divergence, dtype=np.float64)
        mean = 0.0
        cumulative = 0.0
        min_cumulative = 0.0
        alarms: list[int] = []
        for t, x in enumerate(values):
            if not math.isfinite(x):
                continue
            mean += (x - mean) / (t + 1)
            cumulative += x - mean - cfg.delta
            min_cumulative = min(min_cumulative, cumulative)
            if t >= cfg.warmup and cumulative - min_cumulative > cfg.threshold:
                alarms.append(t)
        return _confirmed_states(values.size, alarms)


class CusumConfig(BaseModel):
    """CUSUM parameters (one-sided, upward)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    slack: float = Field(default=0.05, ge=0.0)
    threshold: float = Field(default=0.5, gt=0.0)
    warmup: int = Field(default=3, ge=0)


class Cusum(BaselineDetector):
    """One-sided CUSUM against a running-mean reference."""

    name = "cusum"

    def __init__(self, config: CusumConfig | None = None) -> None:
        """Store the configuration."""
        self.config = config or CusumConfig()

    def run(self, divergence: NDArray[np.float64]) -> NDArray[np.int64]:
        """Alarm when the upward cumulative sum exceeds ``threshold``."""
        cfg = self.config
        values = np.asarray(divergence, dtype=np.float64)
        mean = 0.0
        s_hi = 0.0
        alarms: list[int] = []
        for t, x in enumerate(values):
            if not math.isfinite(x):
                continue
            mean += (x - mean) / (t + 1)
            s_hi = max(0.0, s_hi + x - mean - cfg.slack)
            if t >= cfg.warmup and s_hi > cfg.threshold:
                alarms.append(t)
                s_hi = 0.0
        return _confirmed_states(values.size, alarms)


class AdwinConfig(BaseModel):
    """ADWIN0 parameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    delta: float = Field(default=0.05, gt=0.0, lt=1.0)
    warmup: int = Field(default=3, ge=0)


class Adwin(BaselineDetector):
    """ADWIN0 — exact adaptive windowing (Bifet & Gavaldà, 2007)."""

    name = "adwin"

    def __init__(self, config: AdwinConfig | None = None) -> None:
        """Store the configuration."""
        self.config = config or AdwinConfig()

    def _cut_expands(self, window: list[float], n_total: int) -> bool:
        """True if some split shows a mean gap beyond the variance-based bound.

        Uses the practical ADWIN bound (Bifet & Gavaldà, 2007, §3.2) which corrects
        the raw Hoeffding bound with the window variance — far tighter, and the form
        used in real implementations:
        ``ε = sqrt(2/m · σ²_W · ln(2/δ')) + 2/(3m) · ln(2/δ')``.
        """
        cfg = self.config
        arr = np.asarray(window, dtype=np.float64)
        variance = float(arr.var())
        delta_prime = cfg.delta / max(n_total, 1)
        log_term = math.log(2.0 / delta_prime)
        for split in range(1, len(window)):
            n0, n1 = split, len(window) - split
            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            epsilon = math.sqrt(2.0 / m * variance * log_term) + (
                2.0 / (3.0 * m) * log_term
            )
            if abs(float(arr[:split].mean()) - float(arr[split:].mean())) > epsilon:
                return True
        return False

    def run(self, divergence: NDArray[np.float64]) -> NDArray[np.int64]:
        """Alarm at each step where the adaptive window drops its old sub-window."""
        cfg = self.config
        values = np.asarray(divergence, dtype=np.float64)
        window: list[float] = []
        alarms: list[int] = []
        for t, x in enumerate(values):
            if not math.isfinite(x):
                continue
            window.append(float(x))
            while len(window) >= 2 and self._cut_expands(window, len(window)):
                window.pop(0)
                if t >= cfg.warmup:
                    alarms.append(t)
        return _confirmed_states(values.size, alarms)


class CusumDetectorConfig(BaseModel):
    """CUSUM-as-detector parameters (drives the α-policy on the dense cohort)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    slack: float = Field(default=0.05, ge=0.0)
    enter: float = Field(default=0.5, gt=0.0)
    exit: float = Field(default=0.05, ge=0.0)
    warmup: int = Field(default=3, ge=0)
    recovery_persistence: int = Field(default=3, ge=1)
    rolling: RollingConfig = RollingConfig()


class CusumDriftDetector:
    """Off-the-shelf CUSUM exposed as a drop-in detector for the streaming loop.

    A standard one-sided CUSUM decides *when* preference drifted (the literature's
    detector); on an alarm the detector enters a **held** ``CONFIRMED`` state and stays
    there until the CUSUM statistic relaxes below ``exit`` for ``recovery_persistence``
    steps. Holding the state (rather than firing single steps) is what lets the gradual
    α-glide move toward short-term and stay there while the shift persists, then glide
    back on recovery — the adaptation logic is unchanged; only the detector swapped.

    Exposes the same ``.config`` / ``.run(divergence) -> DetectionResult`` surface as
    :class:`~drift_reco.detectors.detector.PersistenceDriftDetector`, so
    :class:`~drift_reco.evaluation.streaming.StreamingEvaluator` accepts it unchanged.
    """

    def __init__(self, config: CusumDetectorConfig | None = None) -> None:
        """Store the configuration."""
        self.config = config or CusumDetectorConfig()

    def run(self, divergence: NDArray[np.float64]) -> DetectionResult:
        """Run CUSUM and emit a held-CONFIRMED states array with its statistic."""
        cfg = self.config
        values = np.asarray(divergence, dtype=np.float64)
        states = np.zeros(values.size, dtype=np.int64)
        score = np.zeros(values.size, dtype=np.float64)
        confirmed: list[ConfirmedDrift] = []
        mean = 0.0
        s_hi = 0.0
        state = DriftState.STABLE
        recover = 0
        for t, x in enumerate(values):
            if not math.isfinite(x):
                states[t] = int(state)
                continue
            mean += (x - mean) / (t + 1)
            s_hi = max(0.0, s_hi + x - mean - cfg.slack)
            score[t] = s_hi
            if t < cfg.warmup:
                states[t] = int(DriftState.STABLE)
                continue
            if state == DriftState.STABLE:
                if s_hi > cfg.enter:
                    state, recover = DriftState.CONFIRMED, 0
                    confirmed.append(ConfirmedDrift(onset_step=t, detect_step=t))
            elif s_hi <= cfg.exit:
                recover += 1
                if recover >= cfg.recovery_persistence:
                    state = DriftState.STABLE
            else:
                recover = 0
            states[t] = int(state)
        return DetectionResult(states=states, score=score, confirmed=confirmed)


BASELINE_DETECTORS: Final[dict[str, type[BaselineDetector]]] = {
    PageHinkley.name: PageHinkley,
    Cusum.name: Cusum,
    Adwin.name: Adwin,
}


def build_baseline(name: str, threshold: float) -> BaselineDetector:
    """Build a baseline detector with its primary sensitivity knob set.

    ``threshold`` maps to each detector's main knob: Page-Hinkley/CUSUM ``threshold``
    (higher = less sensitive) and ADWIN ``delta`` (lower = less sensitive).

    Raises:
        KeyError: If ``name`` is not a registered baseline detector.
    """
    if name == PageHinkley.name:
        return PageHinkley(PageHinkleyConfig(threshold=threshold))
    if name == Cusum.name:
        return Cusum(CusumConfig(threshold=threshold))
    if name == Adwin.name:
        return Adwin(AdwinConfig(delta=threshold))
    raise KeyError(f"unknown baseline {name!r}; choose from {list(BASELINE_DETECTORS)}")
