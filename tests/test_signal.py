"""Tests for the drift-signal family."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from drift_reco.detectors.signal import (
    LongShortDivergence,
    ShortTermDrift,
    get_signal,
)


def test_short_term_drift_is_low_for_a_stable_series() -> None:
    short = torch.ones(12, 4)
    drift = ShortTermDrift(halflife=5.0).compute(short, short)
    assert drift.shape == (12,)
    assert np.all(drift < 1e-3)


def test_short_term_drift_rises_after_a_jump() -> None:
    stable = torch.tensor([[1.0, 0.0, 0.0]]).repeat(8, 1)
    shifted = torch.tensor([[0.0, 1.0, 0.0]]).repeat(8, 1)
    short = torch.cat([stable, shifted])
    drift = ShortTermDrift(halflife=5.0).compute(short, short)
    assert drift[:8].max() < 0.2
    assert drift[8:].max() > 0.5


def test_long_short_divergence_wraps_a_metric() -> None:
    signal = get_signal("ls_cosine")
    assert isinstance(signal, LongShortDivergence)
    v = torch.tensor([[1.0, 0.0]])
    assert np.isclose(signal.compute(v, v)[0], 0.0, atol=1e-6)


def test_get_signal_unknown_raises() -> None:
    with pytest.raises(KeyError):
        get_signal("nope")
