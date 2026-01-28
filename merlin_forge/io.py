from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from .config import ConfigError


def load_npy(path: str | Path, cfg: Dict[str, Any] | None = None) -> torch.Tensor:
    arr = np.load(path)
    if arr.ndim != 3:
        raise ConfigError(f"Expected .npy with 3 dims [H, W, 2], got shape {arr.shape}")
    if arr.shape[-1] != 2:
        raise ConfigError(f"Expected last dim=2 for real/imag, got shape {arr.shape}")
    tensor = torch.from_numpy(arr)
    if cfg is not None:
        expected_dtype = cfg.get("input", {}).get("dtype", "float32")
        if expected_dtype == "float32":
            tensor = tensor.to(torch.float32)
    return tensor
