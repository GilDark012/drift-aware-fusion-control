"""Ranking metrics from held-out ranks — the one place the maths lives.

Both the static next-item evaluator (Phase 3) and the streaming α-policy evaluator
(Phase 6+) reduce to "given the 1-based rank of each held-out positive, compute
NDCG@k / HR@k / MRR". Keeping that reduction here means the two evaluators cannot
disagree on how a metric is defined. With a single positive per query, Recall@k and
HitRate@k coincide.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["RankingMetrics", "compute_ranking_metrics"]


@dataclass(frozen=True)
class RankingMetrics:
    """Aggregate ranking quality over a set of scored positions."""

    n_scored: int
    n_seen: int
    ndcg_at_k: float
    recall_at_k: float
    hit_rate_at_k: float
    mrr: float
    ndcg_at_k_seen: float
    hit_rate_at_k_seen: float


def compute_ranking_metrics(
    ranks: NDArray[np.int64], seen: NDArray[np.bool_], k: int
) -> RankingMetrics:
    """Reduce per-position ranks to aggregate metrics (overall and seen-only).

    Args:
        ranks: 1-based rank of each held-out target against the full vocabulary.
        seen: Whether each target is a rankable (non-OOV) item.
        k: Cut-off for the @k metrics.

    Returns:
        The aggregated :class:`RankingMetrics`.
    """
    if ranks.size == 0:
        return RankingMetrics(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ranks_f = ranks.astype(np.float64)
    hit = (ranks <= k).astype(np.float64)
    ndcg = np.where(ranks <= k, 1.0 / np.log2(ranks_f + 1.0), 0.0)
    mrr = 1.0 / ranks_f
    seen_f = seen.astype(np.float64)
    n_seen = float(seen_f.sum()) or 1.0
    return RankingMetrics(
        n_scored=int(ranks.size),
        n_seen=int(seen.sum()),
        ndcg_at_k=float(ndcg.mean()),
        recall_at_k=float(hit.mean()),
        hit_rate_at_k=float(hit.mean()),
        mrr=float(mrr.mean()),
        ndcg_at_k_seen=float((ndcg * seen_f).sum() / n_seen),
        hit_rate_at_k_seen=float((hit * seen_f).sum() / n_seen),
    )
