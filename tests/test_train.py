"""End-to-end training test on a tiny in-memory dataset."""

from __future__ import annotations

import numpy as np

from drift_reco.models.config import ModelConfig, TrainConfig
from drift_reco.models.fusion import LongShortFusion
from drift_reco.models.sequences import build_examples
from drift_reco.models.train import Trainer, concat_examples


def _sequences() -> dict[int, np.ndarray]:
    rng = np.random.default_rng(0)
    return {u: rng.integers(2, 12, size=8, dtype=np.int64) for u in range(6)}


def test_trainer_reduces_a_finite_loss() -> None:
    model_cfg = ModelConfig(embedding_dim=8, short_window=3, max_history=4, min_context=2)
    train_cfg = TrainConfig(epochs=1, batch_size=8, n_negatives=4, device="cpu")
    model = LongShortFusion(12, model_cfg)
    outcome = Trainer(model, train_cfg, 12).fit(_sequences())
    assert outcome.n_examples > 0
    assert np.isfinite(outcome.final_loss)
    assert outcome.device == "cpu"


def test_concat_examples_rejects_empty_input() -> None:
    import pytest

    cfg = ModelConfig(min_context=100)
    empty = build_examples(np.arange(5, dtype=np.int64), np.arange(5), cfg)
    with pytest.raises(ValueError, match="no training examples"):
        concat_examples([empty])
