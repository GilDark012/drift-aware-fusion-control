"""Tests for the learned fusion gate and its policy wrapper."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from drift_reco.adaptation.gate import GatePolicy, LearnedGate


def test_learned_gate_outputs_alpha_in_unit_interval() -> None:
    """The gate returns one alpha per row, each strictly within (0, 1)."""
    gate = LearnedGate(dim=8)
    long, short = torch.randn(5, 8), torch.randn(5, 8)
    alpha = gate(long, short)
    assert alpha.shape == (5,)
    assert torch.all(alpha > 0) and torch.all(alpha < 1)


def test_learned_gate_hidden_layer_variant_runs() -> None:
    """The one-hidden-layer variant produces the same output shape."""
    gate = LearnedGate(dim=8, hidden=16)
    alpha = gate(torch.randn(3, 8), torch.randn(3, 8))
    assert alpha.shape == (3,)


def test_gate_policy_maps_reps_to_alpha_series() -> None:
    """GatePolicy returns a numpy alpha array aligned to the representation rows."""
    policy = GatePolicy(LearnedGate(dim=4))
    long, short = torch.randn(6, 4), torch.randn(6, 4)
    states = np.zeros(6, dtype=np.int64)
    zscore = np.zeros(6, dtype=np.float64)
    alpha = policy.alpha_series(states, zscore, long=long, short=short)
    assert alpha.shape == (6,)
    assert np.all((alpha > 0) & (alpha < 1))


def test_gate_policy_requires_representations() -> None:
    """Without the long/short tensors the gate policy raises rather than guessing."""
    policy = GatePolicy(LearnedGate(dim=4))
    with pytest.raises(ValueError, match="requires"):
        policy.alpha_series(np.zeros(2, dtype=np.int64), np.zeros(2, dtype=np.float64))
