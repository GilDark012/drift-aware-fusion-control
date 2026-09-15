"""Persist and reload a trained backbone with its architecture.

A checkpoint is a directory holding the weights (``model.pt``) and a ``meta.json``
that records the vocabulary size and the exact :class:`ModelConfig`. Reloading is
therefore self-describing: later phases (divergence, detection, adaptation) rebuild
the identical architecture without guessing hyperparameters.
"""

from __future__ import annotations

from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict

from .config import ModelConfig
from .fusion import LongShortFusion

__all__ = ["BackboneMeta", "save_backbone", "load_backbone"]

_WEIGHTS_FILE = "model.pt"
_META_FILE = "meta.json"


class BackboneMeta(BaseModel):
    """Everything needed to reconstruct a trained backbone besides its weights."""

    model_config = ConfigDict(extra="forbid")

    n_items: int
    architecture: ModelConfig


def save_backbone(
    model: LongShortFusion,
    architecture: ModelConfig,
    n_items: int,
    out_dir: str | Path,
) -> Path:
    """Save weights and metadata to ``out_dir``; return the directory."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), directory / _WEIGHTS_FILE)
    meta = BackboneMeta(n_items=n_items, architecture=architecture)
    (directory / _META_FILE).write_text(
        meta.model_dump_json(indent=2), encoding="utf-8"
    )
    return directory


def load_backbone(
    directory: str | Path, device: str | torch.device = "cpu"
) -> tuple[LongShortFusion, BackboneMeta]:
    """Rebuild the architecture and load its trained weights."""
    directory = Path(directory)
    meta = BackboneMeta.model_validate_json(
        (directory / _META_FILE).read_text(encoding="utf-8")
    )
    model = LongShortFusion(meta.n_items, meta.architecture)
    state = torch.load(directory / _WEIGHTS_FILE, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, meta
