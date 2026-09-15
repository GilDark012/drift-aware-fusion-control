"""Fusion-weight policies: how α(t) is chosen from the drift signal.

This is the project's contribution — controlling the long/short **fusion weight**
online, over the *same* frozen backbone (ADR-0002), rather than retraining. Each
baseline and the proposed system is one policy here:

* **static**     — constant α (Baseline A: no drift awareness);
* **continuous** — α reacts to the raw divergence z-score every step, with no state
  machine or persistence (Baseline D: adaptive α *without* drift detection);
* **state**      — α is a function of the detector state STABLE/EMERGING/CONFIRMED
  (the proposed drift-aware fusion), moved **abruptly** or **gradually** (H4).

A policy maps the per-step detector states and z-scores to an α series; it never
sees the future beyond what those causal signals already encode.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from ..detectors.detector import DriftState

__all__ = [
    "AlphaPolicy",
    "StaticAlphaPolicy",
    "ContinuousAlphaPolicy",
    "StateAlphaPolicy",
    "AdaptationConfig",
    "build_policy",
]


class AlphaPolicy(ABC):
    """Map causal drift signals to a per-step fusion weight α∈[0,1]."""

    name: str

    @abstractmethod
    def alpha_series(
        self,
        states: NDArray[np.int64],
        zscore: NDArray[np.float64],
        **reps: object,
    ) -> NDArray[np.float64]:
        """Return the α value for every step.

        ``reps`` may carry the per-step long/short representation tensors (keys
        ``long`` and ``short``) for policies that need them, such as a learned gate;
        detector-driven policies ignore them.
        """


class StaticAlphaPolicy(AlphaPolicy):
    """Constant α — the no-drift-awareness baseline (A)."""

    name = "static"

    def __init__(self, alpha: float) -> None:
        """Store the fixed α."""
        self.alpha = alpha

    def alpha_series(
        self,
        states: NDArray[np.int64],
        zscore: NDArray[np.float64],  # noqa: ARG002
        **reps: object,  # noqa: ARG002
    ) -> NDArray[np.float64]:
        """Return a constant α for every step."""
        return np.full(states.shape[0], self.alpha, dtype=np.float64)


class ContinuousAlphaPolicy(AlphaPolicy):
    """α decreases smoothly with the divergence z-score — Baseline D (no detector)."""

    name = "continuous"

    def __init__(self, high: float, sensitivity: float, minimum: float) -> None:
        """Store the ceiling, per-z sensitivity and floor."""
        self.high = high
        self.sensitivity = sensitivity
        self.minimum = minimum

    def alpha_series(
        self,
        states: NDArray[np.int64],  # noqa: ARG002
        zscore: NDArray[np.float64],
        **reps: object,  # noqa: ARG002
    ) -> NDArray[np.float64]:
        """Lower α proportionally to positive z, clipped to [minimum, high]."""
        z = np.nan_to_num(zscore, nan=0.0)
        raw = self.high - self.sensitivity * np.maximum(z, 0.0)
        return np.clip(raw, self.minimum, self.high)


class StateAlphaPolicy(AlphaPolicy):
    """α is a function of detector state; moved abruptly or gradually (H4)."""

    name = "state"

    def __init__(self, levels: dict[int, float], step: float | None) -> None:
        """Store the per-state α targets and the max per-step move (None=abrupt)."""
        self.levels = levels
        self.step = step

    def alpha_series(
        self,
        states: NDArray[np.int64],
        zscore: NDArray[np.float64],  # noqa: ARG002
        **reps: object,  # noqa: ARG002
    ) -> NDArray[np.float64]:
        """Track the state's target α, abruptly or rate-limited by ``step``."""
        targets = np.array([self.levels[int(s)] for s in states], dtype=np.float64)
        if self.step is None:
            return targets
        alpha = np.empty_like(targets)
        current = self.levels[int(DriftState.STABLE)]
        for i, target in enumerate(targets):
            delta = float(np.clip(target - current, -self.step, self.step))
            current += delta
            alpha[i] = current
        return alpha


_TransitionMode = Literal["abrupt", "gradual"]
_DEFAULT_STEP: Final[float] = 0.05


class AdaptationConfig(BaseModel):
    """Configuration of the α policy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    policy: Literal["static", "continuous", "state"] = "state"
    alpha_stable: float = Field(default=0.8, ge=0.0, le=1.0)
    alpha_emerging: float = Field(default=0.5, ge=0.0, le=1.0)
    alpha_confirmed: float = Field(default=0.2, ge=0.0, le=1.0)
    mode: _TransitionMode = "gradual"
    alpha_step: float = Field(default=_DEFAULT_STEP, gt=0.0, le=1.0)
    alpha_static: float = Field(default=0.5, ge=0.0, le=1.0)
    continuous_high: float = Field(default=0.8, ge=0.0, le=1.0)
    continuous_sensitivity: float = Field(default=0.1, ge=0.0)
    continuous_min: float = Field(default=0.2, ge=0.0, le=1.0)


def _state_levels(config: AdaptationConfig) -> dict[int, float]:
    """Map each drift state to its target α (TEMPORARY behaves like STABLE)."""
    return {
        int(DriftState.STABLE): config.alpha_stable,
        int(DriftState.EMERGING): config.alpha_emerging,
        int(DriftState.CONFIRMED): config.alpha_confirmed,
        int(DriftState.TEMPORARY): config.alpha_stable,
    }


def build_policy(config: AdaptationConfig) -> AlphaPolicy:
    """Construct the α policy described by ``config``."""
    if config.policy == "static":
        return StaticAlphaPolicy(config.alpha_static)
    if config.policy == "continuous":
        return ContinuousAlphaPolicy(
            config.continuous_high, config.continuous_sensitivity, config.continuous_min
        )
    step = None if config.mode == "abrupt" else config.alpha_step
    return StateAlphaPolicy(_state_levels(config), step)
