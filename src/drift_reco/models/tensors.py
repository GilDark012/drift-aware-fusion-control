"""Small tensor helpers shared across packages."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

__all__ = ["to_device"]


def to_device(arr: NDArray[np.generic], device: torch.device) -> torch.Tensor:
    """Move a numpy array onto a torch device as a tensor."""
    return torch.from_numpy(arr).to(device)
