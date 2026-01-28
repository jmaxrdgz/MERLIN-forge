from __future__ import annotations

from typing import Any, Dict, List

import torch

from .config import ConfigError


def aggregate_channels(channels: List[torch.Tensor], cfg: Dict[str, Any]) -> torch.Tensor:
    output_cfg = cfg["output"]
    layout = output_cfg.get("layout", "CHW")

    for idx, ch in enumerate(channels):
        if ch.ndim != 2:
            raise ConfigError(f"Channel {idx} must be 2D [H,W], got shape {tuple(ch.shape)}")

    if layout == "CHW":
        y = torch.stack(channels, dim=0)
    elif layout == "HWC":
        y = torch.stack(channels, dim=-1)
    else:
        raise ConfigError(f"Unsupported output.layout: {layout}")
    return y
