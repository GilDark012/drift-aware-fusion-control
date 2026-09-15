"""Training loop for the long/short-term fusion backbone.

Optimisation is next-item prediction with sampled-softmax negatives. When
``alpha_train_mode='random'`` a fresh α~U(0,1) is drawn per example every step, so
the backbone never collapses onto one path and stays fair to any downstream α
policy (fixed baselines or adaptive control).
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from ..config.constants import N_RESERVED_ITEM_IDS
from .config import ModelConfig, TrainConfig
from .fusion import LongShortFusion
from .sequences import ExampleArrays, build_examples

__all__ = ["Trainer", "TrainOutcome", "resolve_device", "concat_examples"]

LOGGER = logging.getLogger(__name__)


class TrainOutcome(BaseModel):
    """Summary of a completed training run."""

    model_config = ConfigDict(extra="forbid")

    n_examples: int
    epochs: int
    final_loss: float
    device: str


def resolve_device(requested: str) -> torch.device:
    """Resolve ``'auto'`` to CUDA when available, else honour the request."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def concat_examples(parts: list[ExampleArrays]) -> ExampleArrays:
    """Concatenate per-user example arrays into one batch."""
    parts = [p for p in parts if len(p) > 0]
    if not parts:
        raise ValueError("no training examples were produced")
    return ExampleArrays(
        np.concatenate([p.short_items for p in parts]),
        np.concatenate([p.short_len for p in parts]),
        np.concatenate([p.long_items for p in parts]),
        np.concatenate([p.long_weights for p in parts]),
        np.concatenate([p.target for p in parts]),
    )


def build_training_examples(
    sequences: dict[int, np.ndarray], cfg: ModelConfig
) -> ExampleArrays:
    """Build examples for every valid position of every user sequence."""
    parts = [
        build_examples(seq, np.arange(seq.size), cfg) for seq in sequences.values()
    ]
    return concat_examples(parts)


class Trainer:
    """Optimise a :class:`LongShortFusion` model on next-item prediction."""

    def __init__(self, model: LongShortFusion, cfg: TrainConfig, n_items: int) -> None:
        """Store the model, config and derive the device and optimiser."""
        self.cfg = cfg
        self.n_items = n_items
        self.device = resolve_device(cfg.device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self._generator = torch.Generator(device="cpu").manual_seed(cfg.seed)
        self._neg_weights: Tensor | None = None

    def _build_negative_weights(self, examples: ExampleArrays) -> None:
        """Popularity^0.75 negative-sampling weights (reserved ids zeroed)."""
        if self.cfg.negatives != "popularity":
            return
        counts = np.bincount(examples.target, minlength=self.n_items).astype(np.float64)
        weights = np.power(counts, 0.75)
        weights[:N_RESERVED_ITEM_IDS] = 0.0
        self._neg_weights = torch.from_numpy(weights).to(self.device)

    def _loader(self, examples: ExampleArrays) -> DataLoader:
        """Wrap example arrays in a shuffling DataLoader."""
        tensors = TensorDataset(
            torch.from_numpy(examples.short_items),
            torch.from_numpy(examples.short_len),
            torch.from_numpy(examples.long_items),
            torch.from_numpy(examples.long_weights),
            torch.from_numpy(examples.target),
        )
        return DataLoader(
            tensors,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            generator=self._generator,
            drop_last=False,
        )

    def _alpha(self, batch_size: int) -> Tensor | float:
        """Draw the fusion weight for a step (per-example if random)."""
        if self.cfg.alpha_train_mode == "fixed":
            return self.cfg.alpha_fixed
        return torch.rand(batch_size, generator=self._generator).to(self.device)

    def _negatives(self, batch_size: int) -> Tensor:
        """Sample negative items, uniformly or from the popularity distribution."""
        shape = (batch_size, self.cfg.n_negatives)
        if self._neg_weights is None:
            return torch.randint(
                N_RESERVED_ITEM_IDS, self.n_items, shape, generator=self._generator
            ).to(self.device)
        flat = torch.multinomial(
            self._neg_weights, batch_size * self.cfg.n_negatives, replacement=True
        )
        return flat.view(shape)

    def _step(self, batch: tuple[Tensor, ...]) -> Tensor:
        """Compute the sampled-softmax loss (plus optional L-S alignment)."""
        short_items, short_len, long_items, long_weights, target = (
            t.to(self.device) for t in batch
        )
        long = self.model.long_term(long_items, long_weights)
        short = self.model.short_term(short_items, short_len)
        query = self.model.fuse(long, short, self._alpha(target.shape[0]))
        negatives = self._negatives(target.shape[0])
        candidates = torch.cat([target.unsqueeze(1), negatives], dim=1)
        logits = self.model.score_items(query, candidates)
        labels = torch.zeros(target.shape[0], dtype=torch.long, device=self.device)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        if self.cfg.align_weight > 0.0:
            align = (
                1.0 - torch.nn.functional.cosine_similarity(long, short, dim=-1)
            ).mean()
            loss = loss + self.cfg.align_weight * align
        return loss

    def fit(self, sequences: dict[int, np.ndarray]) -> TrainOutcome:
        """Train for the configured number of epochs over all user sequences."""
        examples = build_training_examples(sequences, self.model.cfg)
        self._build_negative_weights(examples)
        loader = self._loader(examples)
        last_loss = float("nan")
        self.model.train()
        for epoch in range(self.cfg.epochs):
            epoch_loss = self._run_epoch(loader)
            last_loss = epoch_loss
            LOGGER.info("epoch %d/%d loss=%.4f", epoch + 1, self.cfg.epochs, epoch_loss)
        return TrainOutcome(
            n_examples=len(examples),
            epochs=self.cfg.epochs,
            final_loss=last_loss,
            device=str(self.device),
        )

    def _run_epoch(self, loader: DataLoader) -> float:
        """Run one epoch and return its mean loss."""
        total, n_batches = 0.0, 0
        for batch in loader:
            self.optimizer.zero_grad()
            loss = self._step(batch)
            loss.backward()
            self.optimizer.step()
            total += float(loss.detach())
            n_batches += 1
        return total / max(n_batches, 1)
