import math

import torch

from merlin_forge.channels.adaptive_quant import _hist_mode, _piecewise_map


def test_hist_mode_unimodal():
    torch.manual_seed(0)
    data = torch.randn(10000) * 0.5 + 5.0
    mode = _hist_mode(data, bins=128, min_val=3.0, max_val=7.0)
    assert abs(mode - 5.0) < 0.2


def test_piecewise_map():
    a = torch.tensor([0.0, 2.5, 5.0, 7.5, 10.0])
    t = 5.0
    out = _piecewise_map(a, t, out_min=0.0, out_split=0.5, out_max=1.0, a_max=10.0)
    assert math.isclose(out[0].item(), 0.0, rel_tol=1e-5)
    assert math.isclose(out[2].item(), 0.5, rel_tol=1e-5)
    assert math.isclose(out[-1].item(), 1.0, rel_tol=1e-5)
