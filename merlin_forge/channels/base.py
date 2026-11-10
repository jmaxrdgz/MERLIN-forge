from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from torch import nn


class BaseChannel(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg

    def process(self, x: torch.Tensor, ctx: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, ctx: Dict[str, Any] | None = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if ctx is None:
            ctx = {}
        return self.process(x, ctx)
