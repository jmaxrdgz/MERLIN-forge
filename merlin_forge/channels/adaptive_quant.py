from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from .base import BaseChannel
from ..config import ConfigError


def _percentile(t: torch.Tensor, p: float) -> torch.Tensor:
    if not (0.0 <= p <= 100.0):
        raise ConfigError(f"Percentile must be in [0,100], got {p}")
    return torch.quantile(t, p / 100.0)


def _hist_mode(t: torch.Tensor, bins: int, min_val: float, max_val: float) -> float:
    if max_val <= min_val:
        return float(min_val)
    hist = torch.histc(t, bins=bins, min=min_val, max=max_val)
    idx = int(torch.argmax(hist).item())
    bin_width = (max_val - min_val) / bins
    return float(min_val + (idx + 0.5) * bin_width)


def _piecewise_map(a: torch.Tensor, t: float, out_min: float, out_split: float, out_max: float, a_max: float) -> torch.Tensor:
    eps = 1e-12
    t = max(float(t), eps)
    a_max = max(float(a_max), t + eps)

    a_clamped = torch.clamp(a, 0.0, a_max)
    low = a_clamped <= t

    # Map [0, t] -> [out_min, out_split]
    low_scaled = out_min + (a_clamped / t) * (out_split - out_min)

    # Map [t, a_max] -> [out_split, out_max]
    high_scaled = out_split + ((a_clamped - t) / (a_max - t)) * (out_max - out_split)

    return torch.where(low, low_scaled, high_scaled)


class AdaptiveQuantizationChannels(BaseChannel):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        self.aq_cfg = cfg["adaptive_quantization"]

    def process(self, x: torch.Tensor, ctx: Dict[str, Any]):
        if "amplitude" not in ctx:
            raise ConfigError("Amplitude not found in ctx; ensure MerlinForge preprocess computes it")
        amp = ctx["amplitude"]
        if amp.ndim != 2:
            raise ConfigError(f"Amplitude must be 2D [H,W], got {tuple(amp.shape)}")

        amp_flat = amp.flatten()
        mode_range = self.aq_cfg.get("mode_range_percentiles", [1.0, 99.5])
        p_low = float(mode_range[0])
        p_high = float(mode_range[1])
        hist_min = float(_percentile(amp_flat, p_low).item())
        hist_max = float(_percentile(amp_flat, p_high).item())
        bins = int(self.aq_cfg.get("bins", 512))
        P = _hist_mode(amp_flat, bins=bins, min_val=hist_min, max_val=hist_max)

        alpha_weak = float(self.aq_cfg.get("alpha_weak", 0.7))
        alpha_strong = float(self.aq_cfg.get("alpha_strong", 1.5))
        T_weak = alpha_weak * P
        T_strong = alpha_strong * P

        clip_max = self.aq_cfg.get("clip_max")
        if clip_max is None:
            clip_pct = self.aq_cfg.get("clip_percentiles", [0.0, 99.9])
            clip_max = float(_percentile(amp_flat, float(clip_pct[1])).item())
        else:
            clip_max = float(clip_max)

        weak_map = self.aq_cfg.get("weak_mapping")
        strong_map = self.aq_cfg.get("strong_mapping")

        weak_ch = _piecewise_map(
            amp,
            t=T_weak,
            out_min=float(weak_map["out_min"]),
            out_split=float(weak_map["out_split"]),
            out_max=float(weak_map["out_max"]),
            a_max=clip_max,
        )
        strong_ch = _piecewise_map(
            amp,
            t=T_strong,
            out_min=float(strong_map["out_min"]),
            out_split=float(strong_map["out_split"]),
            out_max=float(strong_map["out_max"]),
            a_max=clip_max,
        )

        ctx.setdefault("stats", {})["P"] = P
        ctx["stats"]["T_weak"] = T_weak
        ctx["stats"]["T_strong"] = T_strong
        ctx["stats"]["clip_max"] = clip_max

        return (weak_ch, strong_ch), ctx
