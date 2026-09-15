"""Tests for the shared ranking metrics."""

from __future__ import annotations

import numpy as np

from drift_reco.evaluation.metrics import compute_ranking_metrics


def test_metrics_on_known_ranks() -> None:
    ranks = np.array([1, 2, 11], dtype=np.int64)
    seen = np.array([True, True, False])
    m = compute_ranking_metrics(ranks, seen, k=10)
    assert m.n_scored == 3
    assert m.n_seen == 2
    assert np.isclose(m.hit_rate_at_k, 2 / 3)
    assert np.isclose(m.mrr, (1 + 0.5 + 1 / 11) / 3)
    # ndcg: 1/log2(2)=1, 1/log2(3)=0.63093, rank 11 outside top-10 -> 0
    assert np.isclose(m.ndcg_at_k, (1 + 0.63092975) / 3)
    # seen-only conditions on the two rankable targets
    assert np.isclose(m.hit_rate_at_k_seen, 1.0)
    assert np.isclose(m.ndcg_at_k_seen, (1 + 0.63092975) / 2)


def test_empty_input_is_zero() -> None:
    m = compute_ranking_metrics(
        np.zeros(0, np.int64), np.zeros(0, np.bool_), k=10
    )
    assert m.n_scored == 0
    assert m.ndcg_at_k == 0.0


def test_recall_equals_hit_rate_for_single_positive() -> None:
    ranks = np.array([3, 20], dtype=np.int64)
    seen = np.array([True, True])
    m = compute_ranking_metrics(ranks, seen, k=10)
    assert m.recall_at_k == m.hit_rate_at_k
