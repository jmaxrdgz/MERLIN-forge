from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from torch import nn


class BaseChannel(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg

    def preprocess(self, x: torch.Tensor, ctx: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return x, ctx

    def process(self, x: torch.Tensor, ctx: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        raise NotImplementedError

    def postprocess(self, y: torch.Tensor, ctx: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return y, ctx

    def forward(self, x: torch.Tensor, ctx: Dict[str, Any] | None = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if ctx is None:
            ctx = {}
        x, ctx = self.preprocess(x, ctx)
        y, ctx = self.process(x, ctx)
        y, ctx = self.postprocess(y, ctx)
        return y, ctx
