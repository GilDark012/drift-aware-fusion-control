"""The long/short-term fusion recommender.

The model keeps a single item-embedding space in which *both* preference signals
live, so their fusion and their divergence are meaningful:

* ``S_u(t)`` — a GRU over the recent short window (recent/volatile behaviour);
* ``L_u(t)`` — a recency-weighted pool of prior history (persistent behaviour);
* ``h = α·L + (1-α)·S`` — the fused query, scored against the item embeddings.

Because L and S share the embedding space, the drift signal ``D_u(t)=dist(L,S)``
(Phase 4) is a distance between two points in one space, and α simply slides the
query along the segment between them.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..config.constants import OOV_IDX, PAD_IDX
from .config import ModelConfig

__all__ = ["LongShortFusion"]


class LongShortFusion(nn.Module):
    """Long/short-term fusion next-item recommender."""

    def __init__(self, n_items: int, cfg: ModelConfig) -> None:
        """Build the embedding table, GRU short encoder and item bias.

        Args:
            n_items: Size of the dense item vocabulary (including reserved ids).
            cfg: Architecture configuration.
        """
        super().__init__()
        self.cfg = cfg
        d = cfg.embedding_dim
        self.item_emb = nn.Embedding(n_items, d, padding_idx=PAD_IDX)
        self.gru = nn.GRU(
            d,
            d,
            num_layers=cfg.gru_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.gru_layers > 1 else 0.0,
        )
        self.item_bias = nn.Parameter(torch.zeros(n_items))
        self.dropout = nn.Dropout(cfg.dropout)
        if cfg.long_term_mode == "attention":
            self.long_attn = nn.Linear(d, d)
            self.long_query = nn.Parameter(torch.randn(d) * 0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        with torch.no_grad():
            self.item_emb.weight[PAD_IDX].zero_()

    def short_term(self, short_items: Tensor, short_len: Tensor) -> Tensor:
        """Encode the recent window into ``S_u(t)`` via the GRU last state."""
        emb = self.dropout(self.item_emb(short_items))
        lengths = short_len.clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)
        return h_n[-1]

    def long_term(self, long_items: Tensor, long_weights: Tensor) -> Tensor:
        """Pool prior history into ``L_u(t)``.

        ``ewma``/``mean`` use the precomputed recency weights directly. ``attention``
        pools with learned content attention (a learnable query over the item
        embeddings), using ``long_weights > 0`` only as the valid-position mask — a
        more expressive long-term encoder than fixed recency pooling.
        """
        emb = self.item_emb(long_items)
        if self.cfg.long_term_mode != "attention":
            return torch.einsum("bmd,bm->bd", emb, long_weights)
        mask = long_weights > 0
        scores = torch.tanh(self.long_attn(emb)) @ self.long_query
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.nan_to_num(torch.softmax(scores, dim=1))
        return torch.einsum("bmd,bm->bd", emb, attn)

    @staticmethod
    def fuse(long: Tensor, short: Tensor, alpha: Tensor | float) -> Tensor:
        """Fuse the two representations: ``h = α·L + (1-α)·S``."""
        if not isinstance(alpha, Tensor):
            alpha = short.new_full((short.shape[0], 1), float(alpha))
        elif alpha.dim() == 1:
            alpha = alpha.unsqueeze(1)
        return alpha * long + (1.0 - alpha) * short

    def query(
        self,
        short_items: Tensor,
        short_len: Tensor,
        long_items: Tensor,
        long_weights: Tensor,
        alpha: Tensor | float,
    ) -> Tensor:
        """Full forward to the fused query vector ``h``."""
        long = self.long_term(long_items, long_weights)
        short = self.short_term(short_items, short_len)
        return self.fuse(long, short, alpha)

    def score_items(self, query: Tensor, items: Tensor) -> Tensor:
        """Score a query against a specific set of candidate items.

        Args:
            query: ``(B, d)`` fused queries.
            items: ``(B, C)`` candidate item ids.

        Returns:
            ``(B, C)`` scores.
        """
        emb = self.item_emb(items)
        dots = torch.einsum("bd,bcd->bc", query, emb)
        return dots + self.item_bias[items]

    def score_all(self, query: Tensor) -> Tensor:
        """Score a query against the whole vocabulary, masking PAD and OOV."""
        scores = query @ self.item_emb.weight.t() + self.item_bias
        scores[:, PAD_IDX] = float("-inf")
        scores[:, OOV_IDX] = float("-inf")
        return scores
