"""Compute per-user ``D_u(t)`` trajectories over a trained backbone.

For a user's chronological stream, at every position with enough context we form the
long-term and short-term representations from the *preceding* history and measure
their divergence. The result is a time series per user, annotated with the item, its
category and whether the category just changed — the interpretable indicator used to
sanity-check the signal (category change is an *indicator*, never the definition).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

from ..config.constants import (
    CATEGORY_COLUMN,
    OOV_IDX,
    S_ITEM_COLUMN,
    S_USER_COLUMN,
    TIMESTAMP_COLUMN,
)
from ..models.config import ModelConfig
from ..models.fusion import LongShortFusion
from ..models.sequences import ExampleArrays, build_examples
from ..models.tensors import to_device
from .signal import DriftSignal

__all__ = ["UserDivergence", "DivergenceTracker", "user_streams"]


@dataclass(frozen=True)
class UserDivergence:
    """A user's divergence trajectory aligned to stream positions."""

    user: int
    step: NDArray[np.int64]
    divergence: NDArray[np.float64]
    timestamp: NDArray[np.datetime64]
    category: NDArray[np.object_]
    target: NDArray[np.int64]
    is_seen: NDArray[np.bool_]

    def __len__(self) -> int:
        """Number of scored positions."""
        return int(self.step.shape[0])

    def category_changed(self) -> NDArray[np.bool_]:
        """Boolean mask: the item category differs from the previous position."""
        if len(self) == 0:
            return np.zeros(0, dtype=bool)
        prev = np.roll(self.category, 1)
        changed = self.category != prev
        changed[0] = False
        return changed

    def to_frame(self) -> pd.DataFrame:
        """Return the trajectory as a tidy DataFrame."""
        return pd.DataFrame(
            {
                "user": self.user,
                "step": self.step,
                "divergence": self.divergence,
                "timestamp": self.timestamp,
                "category": self.category,
                "target": self.target,
                "is_seen": self.is_seen,
                "category_changed": self.category_changed(),
            }
        )


def user_streams(
    frames_by_split: list[pd.DataFrame],
) -> dict[int, pd.DataFrame]:
    """Concatenate per-user rows across splits into one time-ordered stream each."""
    full = pd.concat(frames_by_split, ignore_index=True)
    full = full.sort_values([S_USER_COLUMN, TIMESTAMP_COLUMN], kind="stable")
    return {int(user): group for user, group in full.groupby(S_USER_COLUMN, sort=False)}


class DivergenceTracker:
    """Produce divergence trajectories for individual users."""

    def __init__(
        self,
        model: LongShortFusion,
        model_cfg: ModelConfig,
        signal: DriftSignal,
        device: str | torch.device = "cpu",
        chunk: int = 2048,
    ) -> None:
        """Store the model, architecture, drift signal and scoring device."""
        self.model = model.to(device)
        self.model_cfg = model_cfg
        self.signal = signal
        self.device = torch.device(device)
        self.chunk = chunk

    @torch.no_grad()
    def _divergence(self, examples: ExampleArrays) -> NDArray[np.float64]:
        """Compute long/short reps for every example, then the drift signal once."""
        self.model.eval()
        if len(examples) == 0:
            return np.zeros(0, dtype=np.float64)
        longs, shorts = [], []
        for start in range(0, len(examples), self.chunk):
            stop = start + self.chunk
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
        return self.signal.compute(torch.cat(longs), torch.cat(shorts))

    def user_trajectory(self, stream: pd.DataFrame) -> UserDivergence:
        """Compute the divergence trajectory for one user's ordered stream.

        Divergence is only defined where *both* representations exist, so scoring
        starts once the long history is non-empty (position beyond the short window),
        not merely once there is minimal context.
        """
        ordered = stream.sort_values(TIMESTAMP_COLUMN, kind="stable")
        seq = ordered[S_ITEM_COLUMN].to_numpy(dtype=np.int64)
        first = max(self.model_cfg.min_context, self.model_cfg.short_window + 1)
        positions = np.arange(first, seq.size, dtype=np.int64)
        examples = build_examples(seq, positions, self.model_cfg)
        divergence = self._divergence(examples)
        return UserDivergence(
            user=int(ordered[S_USER_COLUMN].iloc[0]),
            step=positions,
            divergence=divergence,
            timestamp=ordered[TIMESTAMP_COLUMN].to_numpy()[positions],
            category=ordered[CATEGORY_COLUMN].to_numpy()[positions],
            target=seq[positions],
            is_seen=seq[positions] != OOV_IDX,
        )
