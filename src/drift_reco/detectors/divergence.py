"""Divergence metrics between the long-term and short-term representations.

The drift signal is ``D_u(t) = dist(L_u(t), S_u(t))`` — how far the user's recent
behaviour has moved from their persistent profile, measured in the shared embedding
space. Cosine distance is the primary metric (scale-free, bounded); Euclidean and a
symmetric-KL variant are provided for the ablation of §14.

Each metric is a class in one family (an abstract base plus concretes), selected by
name through :func:`get_divergence` — never by scattered ``if name == ...`` tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Final

import torch
from torch import Tensor

__all__ = [
    "DivergenceMetric",
    "CosineDivergence",
    "EuclideanDivergence",
    "SymmetricKLDivergence",
    "DIVERGENCE_METRICS",
    "get_divergence",
]

_EPS: Final[float] = 1e-8


class DivergenceMetric(ABC):
    """A distance between paired long-term and short-term representation batches."""

    name: str

    @abstractmethod
    def distance(self, long: Tensor, short: Tensor) -> Tensor:
        """Return the ``(B,)`` per-row divergence between ``long`` and ``short``."""

    def __call__(self, long: Tensor, short: Tensor) -> Tensor:
        """Alias for :meth:`distance`."""
        return self.distance(long, short)


class CosineDivergence(DivergenceMetric):
    """``1 - cosine_similarity``; 0 when the two point the same way, 2 when opposed."""

    name = "cosine"

    def distance(self, long: Tensor, short: Tensor) -> Tensor:
        """Cosine distance, numerically guarded against zero-norm vectors."""
        sim = torch.nn.functional.cosine_similarity(long, short, dim=-1, eps=_EPS)
        return 1.0 - sim


class EuclideanDivergence(DivergenceMetric):
    """L2 distance ``||L - S||`` in the embedding space."""

    name = "euclidean"

    def distance(self, long: Tensor, short: Tensor) -> Tensor:
        """Euclidean distance between the representations."""
        return torch.linalg.norm(long - short, dim=-1)


class SymmetricKLDivergence(DivergenceMetric):
    """Symmetric KL between softmax distributions over representation activations.

    The representations are turned into distributions with a softmax over their
    dimensions; the symmetric KL ``½(KL(P‖Q)+KL(Q‖P))`` is a distributional
    divergence, offered as an alternative to the geometric metrics.
    """

    name = "sym_kl"

    def distance(self, long: Tensor, short: Tensor) -> Tensor:
        """Jeffreys (symmetric KL) divergence of the softmaxed representations."""
        p = torch.softmax(long, dim=-1).clamp_min(_EPS)
        q = torch.softmax(short, dim=-1).clamp_min(_EPS)
        kl_pq = (p * (p / q).log()).sum(dim=-1)
        kl_qp = (q * (q / p).log()).sum(dim=-1)
        return 0.5 * (kl_pq + kl_qp)


DIVERGENCE_METRICS: Final[dict[str, type[DivergenceMetric]]] = {
    CosineDivergence.name: CosineDivergence,
    EuclideanDivergence.name: EuclideanDivergence,
    SymmetricKLDivergence.name: SymmetricKLDivergence,
}


def get_divergence(name: str) -> DivergenceMetric:
    """Instantiate a divergence metric by name.

    Raises:
        KeyError: If ``name`` is not a registered metric.
    """
    if name not in DIVERGENCE_METRICS:
        known = list(DIVERGENCE_METRICS)
        raise KeyError(f"unknown divergence {name!r}; choose from {known}")
    return DIVERGENCE_METRICS[name]()
