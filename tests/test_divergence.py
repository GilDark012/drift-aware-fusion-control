"""Tests for the divergence metrics."""

from __future__ import annotations

import pytest
import torch

from drift_reco.detectors.divergence import (
    DIVERGENCE_METRICS,
    get_divergence,
)


def test_cosine_is_zero_for_identical_and_two_for_opposite() -> None:
    metric = get_divergence("cosine")
    a = torch.tensor([[1.0, 0.0, 0.0]])
    assert torch.allclose(metric(a, a), torch.zeros(1), atol=1e-6)
    assert torch.allclose(metric(a, -a), torch.full((1,), 2.0), atol=1e-6)


def test_euclidean_matches_norm() -> None:
    metric = get_divergence("euclidean")
    a = torch.tensor([[0.0, 0.0]])
    b = torch.tensor([[3.0, 4.0]])
    assert torch.allclose(metric(a, b), torch.tensor([5.0]))


def test_symmetric_kl_is_nonnegative_and_symmetric() -> None:
    metric = get_divergence("sym_kl")
    a = torch.tensor([[1.0, 2.0, 0.5]])
    b = torch.tensor([[0.2, 1.0, 3.0]])
    dab = metric(a, b)
    dba = metric(b, a)
    assert torch.all(dab >= 0)
    assert torch.allclose(dab, dba, atol=1e-6)
    assert torch.allclose(metric(a, a), torch.zeros(1), atol=1e-6)


def test_registry_and_unknown_metric() -> None:
    assert set(DIVERGENCE_METRICS) == {"cosine", "euclidean", "sym_kl"}
    with pytest.raises(KeyError):
        get_divergence("nope")
