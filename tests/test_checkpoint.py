"""Tests for backbone checkpointing."""

from __future__ import annotations

from pathlib import Path

import torch

from drift_reco.models.checkpoint import load_backbone, save_backbone
from drift_reco.models.config import ModelConfig
from drift_reco.models.fusion import LongShortFusion


def test_save_load_roundtrip_preserves_weights(tmp_path: Path) -> None:
    cfg = ModelConfig(embedding_dim=8, short_window=3, max_history=4)
    model = LongShortFusion(20, cfg)
    save_backbone(model, cfg, 20, tmp_path / "ckpt")
    reloaded, meta = load_backbone(tmp_path / "ckpt")
    assert meta.n_items == 20
    assert meta.architecture == cfg
    for key, tensor in model.state_dict().items():
        assert torch.allclose(tensor, reloaded.state_dict()[key])
