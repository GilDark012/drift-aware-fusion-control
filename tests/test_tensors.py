"""Tests for the shared tensor helper."""

from __future__ import annotations

import numpy as np
import torch

from drift_reco.models.tensors import to_device


def test_to_device_moves_array_to_cpu() -> None:
    tensor = to_device(np.arange(3, dtype=np.int64), torch.device("cpu"))
    assert tensor.tolist() == [0, 1, 2]
    assert tensor.device.type == "cpu"
