"""Turn ordered interaction sequences into long/short-term training examples.

For a target position ``p`` in a user's chronological item sequence, the example is:

* **short window** — the ``short_window`` items immediately before ``p`` (right-padded,
  with an explicit length so the GRU reads only real items);
* **long history** — every item before the short window, capped to ``max_history`` most
  recent, with recency weights (uniform for ``mean`` mode, exponential for ``ewma``);
* **target** — the item at ``p``.

The same primitive builds training examples (target positions inside the training
sequence) and evaluation examples (target positions in a later split, with the
earlier splits supplied as growing context) — so train and eval never diverge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config.constants import (
    PAD_IDX,
    S_ITEM_COLUMN,
    S_USER_COLUMN,
    TIMESTAMP_COLUMN,
)
from .config import ModelConfig

__all__ = ["ExampleArrays", "user_sequences", "build_examples"]


@dataclass(frozen=True)
class ExampleArrays:
    """A batch of examples as dense numpy arrays."""

    short_items: np.ndarray  # (N, short_window) int64
    short_len: np.ndarray  # (N,) int64
    long_items: np.ndarray  # (N, max_history) int64
    long_weights: np.ndarray  # (N, max_history) float32
    target: np.ndarray  # (N,) int64

    def __len__(self) -> int:
        """Number of examples in the batch."""
        return int(self.target.shape[0])


def user_sequences(frame: pd.DataFrame) -> dict[int, np.ndarray]:
    """Group a split into ``s_user -> ordered s_item array`` (time order)."""
    ordered = frame.sort_values([S_USER_COLUMN, TIMESTAMP_COLUMN], kind="stable")
    return {
        int(user): group.to_numpy(dtype=np.int64)
        for user, group in ordered.groupby(S_USER_COLUMN, sort=False)[S_ITEM_COLUMN]
    }


def _long_weights(valid_len: int, cfg: ModelConfig) -> np.ndarray:
    """Recency weights for a valid long-history slice (oldest→newest)."""
    if valid_len == 0:
        return np.zeros(0, dtype=np.float32)
    if cfg.long_term_mode == "mean":
        return np.full(valid_len, 1.0 / valid_len, dtype=np.float32)
    ages = np.arange(valid_len - 1, -1, -1, dtype=np.float32)  # newest age 0
    weights = np.power(0.5, ages / cfg.ewma_halflife)
    return (weights / weights.sum()).astype(np.float32)


def _one_example(
    seq: np.ndarray, pos: int, cfg: ModelConfig
) -> tuple[np.ndarray, int, np.ndarray, np.ndarray, int]:
    """Build the padded arrays for a single target position ``pos``."""
    k, m = cfg.short_window, cfg.max_history
    short_valid = seq[max(0, pos - k) : pos]
    short_items = np.full(k, PAD_IDX, dtype=np.int64)
    short_items[: short_valid.size] = short_valid

    long_valid = seq[: max(0, pos - k)][-m:]
    long_items = np.full(m, PAD_IDX, dtype=np.int64)
    long_weights = np.zeros(m, dtype=np.float32)
    long_items[: long_valid.size] = long_valid
    long_weights[: long_valid.size] = _long_weights(long_valid.size, cfg)
    return short_items, int(short_valid.size), long_items, long_weights, int(seq[pos])


def build_examples(
    seq: np.ndarray, target_positions: np.ndarray, cfg: ModelConfig
) -> ExampleArrays:
    """Assemble example arrays for the given target positions of one sequence.

    Positions with fewer than ``min_context`` prior items are skipped.
    """
    rows = [
        _one_example(seq, int(pos), cfg)
        for pos in target_positions
        if pos >= cfg.min_context
    ]
    if not rows:
        k, m = cfg.short_window, cfg.max_history
        return ExampleArrays(
            np.zeros((0, k), np.int64),
            np.zeros(0, np.int64),
            np.zeros((0, m), np.int64),
            np.zeros((0, m), np.float32),
            np.zeros(0, np.int64),
        )
    short_items, short_len, long_items, long_weights, target = zip(*rows, strict=True)
    return ExampleArrays(
        np.stack(short_items),
        np.asarray(short_len, dtype=np.int64),
        np.stack(long_items),
        np.stack(long_weights),
        np.asarray(target, dtype=np.int64),
    )
