"""Validated configuration for the long/short-term recommender.

Separated from the detector/adaptation settings because the recommender backbone
is shared, unchanged, across every baseline and the drift-aware system — only the
α policy differs downstream. Keeping its configuration isolated makes that shared
contract explicit.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..config.constants import DEFAULT_SEED, DEFAULT_TOP_K

LongTermMode = Literal["mean", "ewma", "attention"]
AlphaTrainMode = Literal["fixed", "random"]
NegativeSampling = Literal["uniform", "popularity"]

__all__ = ["ModelConfig", "TrainConfig", "EvalConfig", "LongTermMode", "AlphaTrainMode"]


class ModelConfig(BaseModel):
    """Architecture of the long/short-term fusion recommender."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    embedding_dim: int = Field(default=64, gt=0)
    short_window: int = Field(default=10, ge=1)
    gru_layers: int = Field(default=1, ge=1)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    long_term_mode: LongTermMode = "ewma"
    ewma_halflife: float = Field(default=20.0, gt=0.0)
    max_history: int = Field(default=50, ge=1)
    min_context: int = Field(default=2, ge=1)


class TrainConfig(BaseModel):
    """How the backbone is optimised.

    ``alpha_train_mode='random'`` samples α~U(0,1) per step so that *both* the
    long-term and short-term paths stay independently predictive — a prerequisite
    for varying α at inference (fixed baselines or adaptive control) over one shared
    backbone. See ADR-0002.

    ``align_weight`` adds ``w·mean(1 - cos(L, S))`` to the loss so that, on the
    user's *actual* (stable) sequences, the long- and short-term representations
    agree in direction. This gives the drift signal ``D=1-cos(L,S)`` headroom: it is
    low while behaviour is stable and rises when the short term genuinely shifts. See
    ADR-0005.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    alpha_train_mode: AlphaTrainMode = "random"
    alpha_fixed: float = Field(default=0.5, ge=0.0, le=1.0)
    align_weight: float = Field(default=0.0, ge=0.0)
    n_negatives: int = Field(default=100, ge=1)
    negatives: NegativeSampling = "uniform"
    lr: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    batch_size: int = Field(default=512, gt=0)
    epochs: int = Field(default=3, ge=1)
    device: str = "auto"
    seed: int = DEFAULT_SEED


class EvalConfig(BaseModel):
    """Next-item evaluation parameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    top_k: int = Field(default=DEFAULT_TOP_K, gt=0)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    score_chunk: int = Field(default=1024, gt=0)
    seed: int = DEFAULT_SEED
