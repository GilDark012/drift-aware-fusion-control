"""Streaming next-item evaluation with a per-step, policy-driven fusion weight.

This is the loop the whole project is about: replay a user's test interactions in
time; at each step form ``L`` and ``S`` from the preceding history, measure their
divergence, let the detector update its state, let the α-policy choose the fusion
weight, score the next item with *that* α, then move on. Because divergence, detector
and policy are all causal, computing them as batched passes over the user's stream is
identical to a true online loop — and far faster.

The same evaluator serves every baseline and the proposed system: only the injected
`AlphaPolicy` differs. `L`/`S` are computed once per user and re-fused under whatever
α the policy dictates, so policies are compared on an identical backbone and history.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch import Tensor

from ..adaptation.policy import AlphaPolicy
from ..config.constants import (
    OOV_IDX,
    S_ITEM_COLUMN,
    S_USER_COLUMN,
    TIMESTAMP_COLUMN,
)
from ..detectors.detector import PersistenceDriftDetector
from ..detectors.rolling import RollingStatistics
from ..detectors.signal import DriftSignal
from ..models.config import ModelConfig
from ..models.fusion import LongShortFusion
from ..models.sequences import ExampleArrays, build_examples
from ..models.tensors import to_device
from .metrics import RankingMetrics, compute_ranking_metrics

__all__ = ["UserRunResult", "StreamingEvaluator", "pool_metrics"]


@dataclass
class UserRunResult:
    """Per-step record of one user's streamed evaluation under a policy."""

    user: int
    timestamp: NDArray[np.datetime64]
    divergence: NDArray[np.float64]
    state: NDArray[np.int64]
    alpha: NDArray[np.float64]
    rank: NDArray[np.int64]
    target: NDArray[np.int64]
    is_seen: NDArray[np.bool_]
    metrics: RankingMetrics

    def to_frame(self) -> pd.DataFrame:
        """Return the per-step record as a tidy DataFrame."""
        return pd.DataFrame(
            {
                "user": self.user,
                "timestamp": self.timestamp,
                "divergence": self.divergence,
                "state": self.state,
                "alpha": self.alpha,
                "rank": self.rank,
                "target": self.target,
                "is_seen": self.is_seen,
            }
        )


class StreamingEvaluator:
    """Replay users' test streams under a given detector and α-policy.

    ``signal_context`` (class attribute, default 40) extra context positions are
    scored before the eval region so the drift signal's slow baseline is established
    from the user's history, not reset at the eval boundary; only the eval region is
    ranked and reported. Override it on the instance to change the warm-up.
    """

    signal_context: int = 40

    def __init__(
        self,
        model: LongShortFusion,
        model_cfg: ModelConfig,
        signal: DriftSignal,
        device: str | torch.device = "cpu",
        top_k: int = 10,
        score_chunk: int = 512,
    ) -> None:
        """Store the frozen backbone, drift signal and scoring settings."""
        self.model = model.to(device).eval()
        self.model_cfg = model_cfg
        self.signal = signal
        self.device = torch.device(device)
        self.top_k = top_k
        self.score_chunk = score_chunk

    @torch.no_grad()
    def _representations(self, examples: ExampleArrays) -> tuple[Tensor, Tensor]:
        """Compute long and short representations for every example."""
        longs, shorts = [], []
        for start in range(0, len(examples), self.score_chunk):
            stop = start + self.score_chunk
            longs.append(
                self.model.long_term(
                    to_device(examples.long_items[start:stop], self.device),
                    to_device(examples.long_weights[start:stop], self.device),
                )
            )
            shorts.append(
                self.model.short_term(
                    to_device(examples.short_items[start:stop], self.device),
                    to_device(examples.short_len[start:stop], self.device),
                )
            )
        return torch.cat(longs), torch.cat(shorts)

    @torch.no_grad()
    def _ranks(
        self,
        long: Tensor,
        short: Tensor,
        alpha: NDArray[np.float64],
        target: NDArray[np.int64],
    ) -> NDArray[np.int64]:
        """Rank each target against the full vocabulary under per-step α."""
        alpha_t = to_device(alpha.astype(np.float64), self.device).float()
        ranks = np.empty(target.size, dtype=np.int64)
        for start in range(0, target.size, self.score_chunk):
            stop = start + self.score_chunk
            query = self.model.fuse(
                long[start:stop], short[start:stop], alpha_t[start:stop]
            )
            scores = self.model.score_all(query)
            tgt = to_device(target[start:stop], self.device)
            tgt_scores = scores.gather(1, tgt.unsqueeze(1))
            ranks[start:stop] = (scores > tgt_scores).sum(dim=1).cpu().numpy() + 1
        return ranks

    def run_user(
        self,
        context: pd.DataFrame,
        evaluation: pd.DataFrame,
        detector: PersistenceDriftDetector,
        policy: AlphaPolicy,
    ) -> UserRunResult | None:
        """Stream one user's eval rows; signal warmed up from context."""
        ctx = context.sort_values(TIMESTAMP_COLUMN, kind="stable")
        ev = evaluation.sort_values(TIMESTAMP_COLUMN, kind="stable")
        n_context = len(ctx)
        ctx_items = ctx[S_ITEM_COLUMN].to_numpy(np.int64)
        seq = np.concatenate([ctx_items, ev[S_ITEM_COLUMN].to_numpy(np.int64)])
        first = max(self.model_cfg.min_context, self.model_cfg.short_window + 1)
        positions = np.arange(max(first, n_context - self.signal_context), seq.size)
        if positions.size == 0 or (positions >= n_context).sum() == 0:
            return None
        examples = build_examples(seq, positions, self.model_cfg)
        long, short = self._representations(examples)
        divergence = self.signal.compute(long, short)
        states = detector.run(divergence).states
        zscore = RollingStatistics(detector.config.rolling).transform(divergence).zscore
        alpha = policy.alpha_series(states, zscore, long=long, short=short)

        mask = positions >= n_context
        target = examples.target[mask]
        ranks = self._ranks(
            long[to_device(mask, self.device)],
            short[to_device(mask, self.device)],
            alpha[mask],
            target,
        )
        seen = target != OOV_IDX
        return UserRunResult(
            user=int(ev[S_USER_COLUMN].iloc[0]),
            timestamp=ev[TIMESTAMP_COLUMN].to_numpy()[positions[mask] - n_context],
            divergence=divergence[mask],
            state=states[mask],
            alpha=alpha[mask],
            rank=ranks,
            target=target,
            is_seen=seen,
            metrics=compute_ranking_metrics(ranks, seen, self.top_k),
        )


def pool_metrics(results: list[UserRunResult], k: int) -> RankingMetrics:
    """Pool per-user ranks into one cohort-level metric set."""
    ranks = np.concatenate([r.rank for r in results])
    seen = np.concatenate([r.is_seen for r in results])
    return compute_ranking_metrics(ranks, seen, k)
