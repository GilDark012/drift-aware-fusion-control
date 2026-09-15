"""Temporal next-item evaluation for the fusion recommender.

Evaluation is chronological: each target position is scored using only the history
that precedes it (earlier splits as context plus the already-seen prefix of the
eval split), so no future information leaks. Ranking is full — the target competes
against the entire item vocabulary — which is more honest than sampled metrics.

Because each query has exactly one held-out positive, ``Recall@k`` and
``HitRate@k`` coincide; both names are reported for continuity with the protocol.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor

from ..config.constants import OOV_IDX
from ..evaluation.metrics import compute_ranking_metrics
from .config import EvalConfig, ModelConfig
from .fusion import LongShortFusion
from .sequences import ExampleArrays, build_examples
from .train import concat_examples

__all__ = ["EvalResult", "NextItemEvaluator", "build_eval_examples"]

LOGGER = logging.getLogger(__name__)


class EvalResult(BaseModel):
    """Aggregate next-item metrics over an evaluated split."""

    model_config = ConfigDict(extra="forbid")

    split: str
    alpha: float
    top_k: int
    n_scored: int
    n_seen_target: int
    ndcg_at_k: float
    recall_at_k: float
    hit_rate_at_k: float
    mrr: float
    ndcg_at_k_seen: float
    hit_rate_at_k_seen: float


def build_eval_examples(
    context: dict[int, np.ndarray],
    eval_seqs: dict[int, np.ndarray],
    cfg: ModelConfig,
) -> ExampleArrays:
    """Build eval examples: predict each eval item from all preceding history."""
    empty = np.zeros(0, dtype=np.int64)
    parts = []
    for user, tail in eval_seqs.items():
        head = context.get(user, empty)
        full = np.concatenate([head, tail])
        positions = np.arange(head.size, full.size)
        parts.append(build_examples(full, positions, cfg))
    return concat_examples(parts)


class NextItemEvaluator:
    """Score eval examples in chunks and aggregate ranking metrics."""

    def __init__(self, model: LongShortFusion, cfg: EvalConfig) -> None:
        """Store the model and evaluation configuration."""
        self.model = model
        self.cfg = cfg
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def _ranks(self, examples: ExampleArrays) -> tuple[Tensor, Tensor]:
        """Return the 1-based rank of every target and whether it is a seen item."""
        self.model.eval()
        targets = torch.from_numpy(examples.target)
        ranks = torch.empty(len(examples), dtype=torch.long)
        for start in range(0, len(examples), self.cfg.score_chunk):
            stop = start + self.cfg.score_chunk
            query = self._query(examples, start, stop)
            scores = self.model.score_all(query)
            tgt = targets[start:stop].to(self.device)
            tgt_scores = scores.gather(1, tgt.unsqueeze(1))
            ranks[start:stop] = (scores > tgt_scores).sum(dim=1).cpu() + 1
        seen = targets != OOV_IDX
        return ranks, seen

    def _slice(self, arr: np.ndarray, start: int, stop: int) -> Tensor:
        """Slice one example array and move it to the model device."""
        return torch.from_numpy(arr[start:stop]).to(self.device)

    def _query(self, examples: ExampleArrays, start: int, stop: int) -> Tensor:
        """Build fused queries for a slice of examples."""
        return self.model.query(
            self._slice(examples.short_items, start, stop),
            self._slice(examples.short_len, start, stop),
            self._slice(examples.long_items, start, stop),
            self._slice(examples.long_weights, start, stop),
            self.cfg.alpha,
        )

    def evaluate(self, examples: ExampleArrays, split: str) -> EvalResult:
        """Compute aggregate metrics over the given eval examples."""
        ranks, seen = self._ranks(examples)
        metrics = compute_ranking_metrics(
            ranks.numpy(), seen.numpy(), self.cfg.top_k
        )
        return EvalResult(
            split=split,
            alpha=self.cfg.alpha,
            top_k=self.cfg.top_k,
            n_scored=metrics.n_scored,
            n_seen_target=metrics.n_seen,
            ndcg_at_k=metrics.ndcg_at_k,
            recall_at_k=metrics.recall_at_k,
            hit_rate_at_k=metrics.hit_rate_at_k,
            mrr=metrics.mrr,
            ndcg_at_k_seen=metrics.ndcg_at_k_seen,
            hit_rate_at_k_seen=metrics.hit_rate_at_k_seen,
        )
