"""Tests for the LongShortFusion model."""

from __future__ import annotations

import torch

from drift_reco.models.config import ModelConfig
from drift_reco.models.fusion import LongShortFusion


def _model(n_items: int = 20) -> LongShortFusion:
    return LongShortFusion(n_items, ModelConfig(embedding_dim=8, short_window=3, max_history=4))


def _batch() -> dict[str, torch.Tensor]:
    return {
        "short_items": torch.tensor([[2, 3, 4], [5, 6, 0]]),
        "short_len": torch.tensor([3, 2]),
        "long_items": torch.tensor([[2, 3, 0, 0], [7, 8, 9, 0]]),
        "long_weights": torch.tensor(
            [[0.5, 0.5, 0.0, 0.0], [0.2, 0.3, 0.5, 0.0]]
        ),
    }


def _attention_model(n_items: int = 20) -> LongShortFusion:
    return LongShortFusion(
        n_items,
        ModelConfig(
            embedding_dim=8, short_window=3, max_history=4, long_term_mode="attention"
        ),
    )


def test_attention_long_term_shape_and_empty_history() -> None:
    """Attention pooling returns (B, d) and yields a finite vector with no history."""
    model = _attention_model()
    b = _batch()
    long = model.long_term(b["long_items"], b["long_weights"])
    assert long.shape == (2, 8)
    assert torch.isfinite(long).all()
    # A row with an all-zero weight mask (no valid history) must not produce NaNs.
    empty_items = torch.zeros(1, 4, dtype=torch.long)
    empty_weights = torch.zeros(1, 4)
    empty_long = model.long_term(empty_items, empty_weights)
    assert torch.isfinite(empty_long).all()


def test_fuse_is_a_convex_combination() -> None:
    long = torch.ones(2, 4)
    short = torch.zeros(2, 4)
    fused = LongShortFusion.fuse(long, short, 0.25)
    assert torch.allclose(fused, torch.full((2, 4), 0.25))


def test_query_shape_matches_embedding_dim() -> None:
    model = _model()
    b = _batch()
    q = model.query(b["short_items"], b["short_len"], b["long_items"], b["long_weights"], 0.5)
    assert q.shape == (2, 8)


def test_score_all_masks_reserved_ids() -> None:
    model = _model()
    b = _batch()
    q = model.query(b["short_items"], b["short_len"], b["long_items"], b["long_weights"], 0.5)
    scores = model.score_all(q)
    assert torch.isneginf(scores[:, 0]).all()  # PAD
    assert torch.isneginf(scores[:, 1]).all()  # OOV


def test_per_example_alpha_vector_is_accepted() -> None:
    model = _model()
    b = _batch()
    alpha = torch.tensor([0.1, 0.9])
    q = model.query(b["short_items"], b["short_len"], b["long_items"], b["long_weights"], alpha)
    assert q.shape == (2, 8)
