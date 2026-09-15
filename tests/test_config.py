"""Tests for the recommender configuration models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from drift_reco.models.config import EvalConfig, ModelConfig, TrainConfig


def test_defaults_are_sensible() -> None:
    assert ModelConfig().long_term_mode == "ewma"
    assert TrainConfig().alpha_train_mode == "random"
    assert EvalConfig().top_k == 10


def test_configs_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ModelConfig(unknown=1)  # type: ignore[call-arg]


def test_eval_alpha_is_bounded() -> None:
    with pytest.raises(ValidationError):
        EvalConfig(alpha=1.5)


def test_model_config_is_frozen() -> None:
    cfg = ModelConfig()
    with pytest.raises(ValidationError):
        cfg.embedding_dim = 128
