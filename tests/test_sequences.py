"""Tests for long/short example construction."""

from __future__ import annotations

import numpy as np

from drift_reco.config.constants import PAD_IDX
from drift_reco.models.config import ModelConfig
from drift_reco.models.sequences import build_examples


def _cfg(**kw: object) -> ModelConfig:
    base = {"short_window": 3, "max_history": 4, "min_context": 2, "ewma_halflife": 10.0}
    base.update(kw)
    return ModelConfig(**base)  # type: ignore[arg-type]


def test_short_window_is_right_padded_with_correct_length() -> None:
    seq = np.array([10, 11, 12, 13, 14], dtype=np.int64)
    ex = build_examples(seq, np.array([4]), _cfg())
    # position 4: short window = last 3 items before idx 4 = [11,12,13]
    assert ex.short_len[0] == 3
    assert list(ex.short_items[0]) == [11, 12, 13]
    assert ex.target[0] == 14


def test_positions_below_min_context_are_skipped() -> None:
    seq = np.array([10, 11, 12], dtype=np.int64)
    ex = build_examples(seq, np.array([0, 1, 2]), _cfg())
    # only position 2 has >= min_context(2) prior items
    assert len(ex) == 1
    assert ex.target[0] == 12


def test_long_history_is_capped_and_weighted() -> None:
    seq = np.arange(20, 40, dtype=np.int64)  # 20 items
    ex = build_examples(seq, np.array([19]), _cfg(short_window=2, max_history=4))
    # long = items before the 2-item short window, capped to last 4
    valid = ex.long_items[0][ex.long_items[0] != PAD_IDX]
    assert valid.size == 4
    weights = ex.long_weights[0]
    assert np.isclose(weights.sum(), 1.0)
    # ewma: most recent long item weighted more than the oldest
    nonzero = weights[weights > 0]
    assert nonzero[-1] > nonzero[0]


def test_mean_mode_weights_are_uniform() -> None:
    seq = np.arange(0, 10, dtype=np.int64)
    ex = build_examples(seq, np.array([9]), _cfg(short_window=2, long_term_mode="mean"))
    weights = ex.long_weights[0]
    nonzero = weights[weights > 0]
    assert np.allclose(nonzero, nonzero[0])
