import pytest
import torch

from merlin_forge.channels.merlin import MerlinDenoiserAdapter
from merlin_forge.config import ConfigError


@pytest.mark.parametrize("leading_dims", [(), (1,), (1, 1)])
def test_normalize_output_preserves_spatial_singletons(leading_dims):
    adapter = MerlinDenoiserAdapter("spotlight", "cpu", "auto", "amplitude", True, None, None)
    amplitude = torch.arange(4, dtype=torch.float32).reshape(1, 4)
    output = amplitude.reshape(*leading_dims, 1, 4)
    result = adapter._normalize_output(output, amplitude)
    torch.testing.assert_close(result, amplitude)


@pytest.mark.parametrize("shape", [(2, 8, 8), (1, 2, 8, 8), (8,), (1, 1, 7, 8)])
def test_denoise_rejects_invalid_output_shapes(shape):
    adapter = MerlinDenoiserAdapter("spotlight", "cpu", "stacked", "amplitude", True, None, None)

    class Model:
        def denoise(self, image):
            return torch.zeros(shape)

    adapter._get_model = lambda: Model()
    amplitude = torch.zeros(8, 8)
    with pytest.raises(ConfigError):
        adapter.denoise(amplitude, amplitude, amplitude)
