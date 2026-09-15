"""Tests for the temporal next-item evaluator."""

from __future__ import annotations

import numpy as np

from drift_reco.models.config import EvalConfig, ModelConfig
from drift_reco.models.evaluate import NextItemEvaluator, build_eval_examples
from drift_reco.models.fusion import LongShortFusion


def _context_and_eval() -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    rng = np.random.default_rng(1)
    context = {u: rng.integers(2, 12, size=6, dtype=np.int64) for u in range(4)}
    eval_seqs = {u: rng.integers(2, 12, size=4, dtype=np.int64) for u in range(4)}
    return context, eval_seqs


def test_metrics_are_within_unit_range() -> None:
    model_cfg = ModelConfig(embedding_dim=8, short_window=3, max_history=4, min_context=2)
    model = LongShortFusion(12, model_cfg)
    context, eval_seqs = _context_and_eval()
    examples = build_eval_examples(context, eval_seqs, model_cfg)
    result = NextItemEvaluator(model, EvalConfig(top_k=5, score_chunk=4)).evaluate(
        examples, "test"
    )
    assert result.n_scored == len(examples)
    for value in (result.ndcg_at_k, result.hit_rate_at_k, result.mrr):
        assert 0.0 <= value <= 1.0
    # single held-out positive → recall equals hit rate
    assert result.recall_at_k == result.hit_rate_at_k


def test_context_grows_history_across_the_boundary() -> None:
    model_cfg = ModelConfig(short_window=2, max_history=4, min_context=1)
    # With context supplied, the first eval position already has enough history.
    context = {0: np.array([2, 3, 4], dtype=np.int64)}
    eval_seqs = {0: np.array([5, 6], dtype=np.int64)}
    examples = build_eval_examples(context, eval_seqs, model_cfg)
    assert len(examples) == 2
    assert list(examples.target) == [5, 6]
