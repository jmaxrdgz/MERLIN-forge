import torch

from merlin_forge.aggregate import aggregate_channels
from merlin_forge.config import validate_config


def test_aggregate_chw():
    cfg = {"output": {"layout": "CHW"}}
    validate_config(cfg)
    ch = [torch.zeros(2, 3), torch.ones(2, 3), torch.full((2, 3), 2.0)]
    y = aggregate_channels(ch, cfg)
    assert y.shape == (3, 2, 3)


def test_aggregate_hwc():
    cfg = {"output": {"layout": "HWC"}}
    validate_config(cfg)
    ch = [torch.zeros(2, 3), torch.ones(2, 3), torch.full((2, 3), 2.0)]
    y = aggregate_channels(ch, cfg)
    assert y.shape == (2, 3, 3)
