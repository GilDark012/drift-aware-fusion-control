"""A learned fusion gate, the SLSRec-style comparison for the drift-aware policy.

The project's contribution drives the fusion weight from an *explicit* online drift
detector. The natural point of comparison is an *implicit, learned* gate that maps the
long- and short-term representations directly to a weight, optimised end-to-end for
next-item accuracy on the training stream (the mechanism used by SLSRec and related
long/short models). This module trains such a gate on top of the *frozen* backbone, so
that the two approaches differ only in how α is chosen — the fair contrast — and exposes
it as an :class:`AlphaPolicy` for the streaming evaluator.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn

from ..config.constants import OOV_IDX
from ..models.config import ModelConfig
from ..models.fusion import LongShortFusion
from ..models.sequences import ExampleArrays, build_examples
from ..models.tensors import to_device
from ..models.train import concat_examples
from .policy import AlphaPolicy

__all__ = ["LearnedGate", "GatePolicy", "GateTrainConfig", "train_gate"]


class GateTrainConfig(BaseModel):
    """Optimisation hyperparameters for the learned fusion gate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    epochs: int = Field(default=10, ge=1)
    lr: float = Field(default=1e-3, gt=0.0)
    batch_size: int = Field(default=512, gt=0)
    hidden: int = Field(default=0, ge=0)
    seed: int = 42


class LearnedGate(nn.Module):
    """α = σ(w · [L ; S] + b): a learned gate over the two representations."""

    def __init__(self, dim: int, hidden: int = 0) -> None:
        """Build a linear (or one-hidden-layer) gate over the concatenated reps."""
        super().__init__()
        if hidden > 0:
            self.net: nn.Module = nn.Sequential(
                nn.Linear(2 * dim, hidden), nn.Tanh(), nn.Linear(hidden, 1)
            )
        else:
            self.net = nn.Linear(2 * dim, 1)

    def forward(self, long: Tensor, short: Tensor) -> Tensor:
        """Return α∈(0,1) for each row of the batched (long, short) reps."""
        logit = self.net(torch.cat([long, short], dim=-1)).squeeze(-1)
        return torch.sigmoid(logit)


class GatePolicy(AlphaPolicy):
    """Expose a trained :class:`LearnedGate` as a per-step α policy."""

    name = "learned-gate"

    def __init__(self, gate: LearnedGate) -> None:
        """Store the trained gate in eval mode."""
        self.gate = gate.eval()

    def alpha_series(
        self,
        states: NDArray[np.int64],  # noqa: ARG002
        zscore: NDArray[np.float64],  # noqa: ARG002
        **reps: object,
    ) -> NDArray[np.float64]:
        """Compute α from the long/short representation tensors from the evaluator."""
        long, short = reps.get("long"), reps.get("short")
        if not isinstance(long, Tensor) or not isinstance(short, Tensor):
            raise ValueError("GatePolicy requires 'long' and 'short' representations")
        with torch.no_grad():
            alpha = self.gate(long, short)
        return alpha.detach().cpu().numpy().astype(np.float64)


def _training_examples(
    sequences: dict[int, np.ndarray], cfg: ModelConfig
) -> ExampleArrays:
    """Build in-vocabulary next-item examples across all training users."""
    first = max(cfg.min_context, cfg.short_window + 1)
    parts = [
        build_examples(seq, np.arange(first, seq.size, dtype=np.int64), cfg)
        for seq in sequences.values()
        if seq.size > first
    ]
    ex = concat_examples(parts)
    keep = ex.target != OOV_IDX  # OOV targets are unrankable under full softmax
    return ExampleArrays(
        long_items=ex.long_items[keep],
        long_weights=ex.long_weights[keep],
        short_items=ex.short_items[keep],
        short_len=ex.short_len[keep],
        target=ex.target[keep],
    )


def train_gate(
    model: LongShortFusion,
    cfg: ModelConfig,
    sequences: dict[int, np.ndarray],
    device: str | torch.device = "cpu",
    train_cfg: GateTrainConfig | None = None,
) -> LearnedGate:
    """Train a learned fusion gate on the frozen backbone's representations."""
    tc = train_cfg or GateTrainConfig()
    torch.manual_seed(tc.seed)
    rng = np.random.default_rng(tc.seed)
    dev = torch.device(device)
    model.to(dev).eval()
    gate = LearnedGate(cfg.embedding_dim, hidden=tc.hidden).to(dev)
    optim = torch.optim.Adam(gate.parameters(), lr=tc.lr)
    examples = _training_examples(sequences, cfg)
    n = len(examples)
    for _ in range(tc.epochs):
        order = rng.permutation(n)
        for start in range(0, n, tc.batch_size):
            idx = order[start : start + tc.batch_size]
            with torch.no_grad():
                long = model.long_term(
                    to_device(examples.long_items[idx], dev),
                    to_device(examples.long_weights[idx], dev),
                )
                short = model.short_term(
                    to_device(examples.short_items[idx], dev),
                    to_device(examples.short_len[idx], dev),
                )
            alpha = gate(long, short).unsqueeze(1)
            query = alpha * long + (1.0 - alpha) * short
            logits = model.score_all(query)
            target = to_device(examples.target[idx], dev)
            loss = nn.functional.cross_entropy(logits, target)
            optim.zero_grad()
            loss.backward()
            optim.step()
    return gate
