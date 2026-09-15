"""Persistence-aware drift detector: a state machine over the divergence signal.

A single high divergence reading must not trigger adaptation — only a *persistent*
rise should. The detector walks a user's causal score series through the states

    STABLE → EMERGING → CONFIRMED → (recover) → STABLE

and treats a rise that fades before it persists as a TEMPORARY deviation, which is
counted but never confirmed. Persistence (how many elevated steps confirm a shift)
and the threshold strategy are the knobs; both are tuned on validation only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from .rolling import RollingConfig, RollingStatistics
from .threshold import get_threshold

__all__ = [
    "DriftState",
    "DetectorConfig",
    "ConfirmedDrift",
    "DetectionResult",
    "PersistenceDriftDetector",
]


class DriftState(IntEnum):
    """The detector's state at a given step."""

    STABLE = 0
    EMERGING = 1
    CONFIRMED = 2
    TEMPORARY = 3


class DetectorConfig(BaseModel):
    """Detector hyperparameters (tuned on validation).

    ``enter``/``exit`` are on the scale of the chosen strategy's score: z-score units
    for ``adaptive`` (e.g. 2.5 / 1.0), raw-divergence units for ``global``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: str = "adaptive"
    enter: float = 2.5
    exit: float = 1.0
    persistence: int = Field(default=5, ge=1)
    gap_tolerance: int = Field(default=1, ge=0)
    recovery_persistence: int = Field(default=5, ge=1)
    warmup: int = Field(default=8, ge=0)
    cooldown: int = Field(default=0, ge=0)
    rolling: RollingConfig = RollingConfig()


@dataclass(frozen=True)
class ConfirmedDrift:
    """A confirmed shift: where it was first suspected and where it was confirmed."""

    onset_step: int
    detect_step: int

    @property
    def confirmation_lag(self) -> int:
        """Steps between first suspicion and confirmation."""
        return self.detect_step - self.onset_step


@dataclass
class DetectionResult:
    """Per-step states plus the confirmed shifts and temporary-deviation count."""

    states: NDArray[np.int64]
    score: NDArray[np.float64]
    confirmed: list[ConfirmedDrift] = field(default_factory=list)
    n_temporary: int = 0

    @property
    def n_confirmed(self) -> int:
        """Number of confirmed shifts."""
        return len(self.confirmed)

    def first_detection(self) -> int | None:
        """Step index of the first confirmation, or None if never confirmed."""
        return self.confirmed[0].detect_step if self.confirmed else None


class _Walk:
    """Mutable cursor state for one pass of the state machine."""

    def __init__(self) -> None:
        """Initialise counters at STABLE."""
        self.state = DriftState.STABLE
        self.streak = 0
        self.gap = 0
        self.onset = -1
        self.recover = 0
        self.cooldown = 0


class PersistenceDriftDetector:
    """Detect persistent divergence rises via a small state machine."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        """Store config and build the rolling helper and threshold strategy."""
        self.config = config or DetectorConfig()
        self.rolling = RollingStatistics(self.config.rolling)
        self.strategy = get_threshold(self.config.strategy)

    def run(self, divergence: NDArray[np.float64]) -> DetectionResult:
        """Run the detector over a full divergence series (causal)."""
        values = np.asarray(divergence, dtype=np.float64)
        score = self.strategy.score(values, self.rolling.transform(values))
        return self._walk(score)

    def _walk(self, score: NDArray[np.float64]) -> DetectionResult:
        """Sweep the score left-to-right through the state machine."""
        cfg = self.config
        states = np.zeros(score.size, dtype=np.int64)
        result = DetectionResult(states=states, score=score)
        w = _Walk()
        for t in range(score.size):
            if w.cooldown > 0:
                w.cooldown -= 1
            if t < cfg.warmup or not np.isfinite(score[t]):
                states[t] = DriftState.STABLE
                continue
            self._transition(w, score[t], t, result)
            states[t] = w.state
        return result

    def _transition(
        self, w: _Walk, value: float, t: int, result: DetectionResult
    ) -> None:
        """Apply one step of the state machine, mutating ``w`` and ``result``."""
        cfg = self.config
        high, low = value >= cfg.enter, value <= cfg.exit
        if w.state in (DriftState.STABLE, DriftState.TEMPORARY):
            w.state = DriftState.STABLE
            if high and w.cooldown == 0:
                w.state, w.streak, w.gap, w.onset = DriftState.EMERGING, 1, 0, t
        elif w.state == DriftState.EMERGING:
            self._emerging(w, high, t, result)
        elif w.state == DriftState.CONFIRMED:
            self._confirmed(w, low)

    def _emerging(self, w: _Walk, high: bool, t: int, result: DetectionResult) -> None:
        """Advance an EMERGING deviation toward CONFIRMED or TEMPORARY."""
        cfg = self.config
        if high:
            w.streak += 1
            w.gap = 0
            if w.streak >= cfg.persistence:
                w.state = DriftState.CONFIRMED
                w.recover = 0
                result.confirmed.append(
                    ConfirmedDrift(onset_step=w.onset, detect_step=t)
                )
            return
        w.gap += 1
        if w.gap > cfg.gap_tolerance:
            w.state = DriftState.TEMPORARY
            result.n_temporary += 1

    def _confirmed(self, w: _Walk, low: bool) -> None:
        """Hold CONFIRMED until the score relaxes for enough consecutive steps."""
        cfg = self.config
        if low:
            w.recover += 1
            if w.recover >= cfg.recovery_persistence:
                w.state = DriftState.STABLE
                w.cooldown = cfg.cooldown
        else:
            w.recover = 0
