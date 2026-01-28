from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from torch import nn

from .aggregate import aggregate_channels
from .channels.adaptive_quant import AdaptiveQuantizationChannels
from .channels.merlin import MerlinDespeckleChannel
from .config import ConfigError


def _compute_amplitude(x: torch.Tensor) -> torch.Tensor:
    re = x[..., 0]
    im = x[..., 1]
    return torch.sqrt(re * re + im * im)


def _apply_dtype_policy(y: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
    output_cfg = cfg["output"]
    dtype = output_cfg.get("dtype", "float32")
    if dtype == "float32":
        return y.to(torch.float32)
    if dtype != "uint16":
        raise ConfigError(f"Unsupported output.dtype: {dtype}")

    uint16_cfg = output_cfg.get("uint16", {})
    range_mode = uint16_cfg.get("range_mode", "use_output_endpoints")
    clamp = bool(uint16_cfg.get("clamp", True))

    layout = output_cfg.get("layout", "CHW")
    if range_mode == "use_output_endpoints":
        endpoints = uint16_cfg.get("endpoints")
        if endpoints is None:
            raise ConfigError("output.uint16.endpoints must be set for range_mode='use_output_endpoints'")
        if len(endpoints) != 3:
            raise ConfigError("output.uint16.endpoints must have 3 entries for 3 channels")
        if layout == "CHW":
            mins = torch.tensor([e[0] for e in endpoints], device=y.device, dtype=y.dtype).view(-1, 1, 1)
            maxs = torch.tensor([e[1] for e in endpoints], device=y.device, dtype=y.dtype).view(-1, 1, 1)
        elif layout == "HWC":
            mins = torch.tensor([e[0] for e in endpoints], device=y.device, dtype=y.dtype).view(1, 1, -1)
            maxs = torch.tensor([e[1] for e in endpoints], device=y.device, dtype=y.dtype).view(1, 1, -1)
        else:
            raise ConfigError(f"Unsupported output.layout: {layout}")
        scale = 65535.0 / torch.clamp(maxs - mins, min=1e-12)
        y_scaled = (y - mins) * scale
    elif range_mode == "minmax_per_channel":
        # compute per-channel min/max
        if y.ndim == 3 and layout == "CHW":
            mins = y.view(y.shape[0], -1).min(dim=1).values.view(-1, 1, 1)
            maxs = y.view(y.shape[0], -1).max(dim=1).values.view(-1, 1, 1)
        elif y.ndim == 3 and layout == "HWC":
            mins = y.view(-1, y.shape[-1]).min(dim=0).values.view(1, 1, -1)
            maxs = y.view(-1, y.shape[-1]).max(dim=0).values.view(1, 1, -1)
        else:
            # HWC or other, fallback to global
            mins = y.min()
            maxs = y.max()
        scale = 65535.0 / torch.clamp(maxs - mins, min=1e-12)
        y_scaled = (y - mins) * scale
    else:
        raise ConfigError(f"Unsupported output.uint16.range_mode: {range_mode}")

    if clamp:
        y_scaled = torch.clamp(y_scaled, 0.0, 65535.0)
    return y_scaled.to(torch.uint16)


class MerlinForge(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.get("merlin", {}).get("device", "cpu")
        self.merlin = MerlinDespeckleChannel(cfg)
        self.adaptive = AdaptiveQuantizationChannels(cfg)

    def preprocess(self, x: torch.Tensor, ctx: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if x.ndim != 3 or x.shape[-1] != 2:
            raise ConfigError(f"Input tensor must have shape [H, W, 2], got {tuple(x.shape)}")
        if not torch.is_floating_point(x):
            x = x.to(torch.float32)
        if self.device:
            x = x.to(self.device)

        ctx = ctx or {}
        ctx["re"] = x[..., 0]
        ctx["im"] = x[..., 1]
        if "amplitude" not in ctx:
            ctx["amplitude"] = _compute_amplitude(x)
        return x, ctx

    def forward(self, x: torch.Tensor, ctx: Dict[str, Any] | None = None):
        if ctx is None:
            ctx = {}
        x, ctx = self.preprocess(x, ctx)
        merlin_ch, ctx = self.merlin(x, ctx)
        (weak_ch, strong_ch), ctx = self.adaptive(x, ctx)

        order = self.cfg.get("output", {}).get("order", ["merlin", "weak", "strong"])
        channel_map = {"merlin": merlin_ch, "weak": weak_ch, "strong": strong_ch}
        try:
            channels = [channel_map[name] for name in order]
        except KeyError as exc:
            raise ConfigError(f"Invalid output.order entry: {exc}") from exc
        y = aggregate_channels(channels, self.cfg)
        y = _apply_dtype_policy(y, self.cfg)
        return y, ctx
