"""Tests for the streaming α-policy evaluator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from drift_reco.adaptation.policy import AdaptationConfig, build_policy
from drift_reco.detectors.detector import DetectorConfig, PersistenceDriftDetector
from drift_reco.detectors.signal import get_signal
from drift_reco.evaluation.streaming import StreamingEvaluator, pool_metrics
from drift_reco.models.config import ModelConfig
from drift_reco.models.fusion import LongShortFusion


def _frame(user: int, items: np.ndarray, start: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "s_user": user,
            "s_item": items,
            "timestamp": pd.date_range(start, periods=len(items), freq="D"),
        }
    )


def _evaluator() -> StreamingEvaluator:
    cfg = ModelConfig(embedding_dim=8, short_window=3, max_history=5, min_context=2)
    model = LongShortFusion(15, cfg)
    return StreamingEvaluator(
        model, cfg, get_signal("short_term_drift"), "cpu", top_k=5, score_chunk=4
    )


def _detector() -> PersistenceDriftDetector:
    return PersistenceDriftDetector(
        DetectorConfig(strategy="adaptive", enter=2.0, persistence=3, warmup=2)
    )


def test_run_user_aligns_all_series_to_eval_length() -> None:
    ev = _evaluator()
    ctx = _frame(7, np.random.default_rng(0).integers(2, 15, 12), "2015-01-01")
    evl = _frame(7, np.random.default_rng(1).integers(2, 15, 10), "2016-01-01")
    result = ev.run_user(ctx, evl, _detector(), build_policy(AdaptationConfig(policy="state")))
    assert result is not None
    assert result.user == 7
    n = result.metrics.n_scored
    assert n == 10
    assert result.alpha.shape[0] == n
    assert result.divergence.shape[0] == n
    assert result.timestamp.shape[0] == n


def test_metrics_are_within_unit_range_and_poolable() -> None:
    ev = _evaluator()
    detector, policy = _detector(), build_policy(AdaptationConfig(policy="static"))
    results = []
    for u in range(3):
        ctx = _frame(u, np.random.default_rng(u).integers(2, 15, 12), "2015-01-01")
        evl = _frame(u, np.random.default_rng(u + 10).integers(2, 15, 8), "2016-01-01")
        results.append(ev.run_user(ctx, evl, detector, policy))
    pooled = pool_metrics(results, 5)
    assert 0.0 <= pooled.ndcg_at_k <= 1.0
    assert pooled.n_scored == 24


def test_static_policy_yields_constant_alpha() -> None:
    ev = _evaluator()
    ctx = _frame(1, np.random.default_rng(2).integers(2, 15, 12), "2015-01-01")
    evl = _frame(1, np.random.default_rng(3).integers(2, 15, 8), "2016-01-01")
    result = ev.run_user(
        ctx, evl, _detector(), build_policy(AdaptationConfig(policy="static", alpha_static=0.5))
    )
    assert np.allclose(result.alpha, 0.5)
