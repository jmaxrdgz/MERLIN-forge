from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from torch import nn

from .base import BaseChannel
from ..config import ConfigError


class MerlinDenoiserAdapter:
    _model_cache: Dict[tuple, Any] = {}

    def __init__(
        self,
        model_name: str,
        device: str,
        input_format: str,
        output_kind: str,
        symetrise: bool | None,
        patch_size: int | None,
        stride_size: int | None,
    ):
        self.model_name = model_name
        self.device = device
        self.input_format = input_format
        self.output_kind = output_kind
        self.symetrise = symetrise
        self.patch_size = patch_size
        self.stride_size = stride_size
        self._detected_format: str | None = None

    def _get_model(self):
        key = (self.model_name, self.device, self.symetrise)
        if key in self._model_cache:
            return self._model_cache[key]
        if self.device and self.device.startswith("cuda") and not torch.cuda.is_available():
            raise ConfigError("merlin.device is set to cuda but CUDA is not available")
        try:
            from deepdespeckling.merlin.merlin_denoiser import MerlinDenoiser

        except Exception as exc:
            raise ConfigError(
                "deepdespeckling.MerlinDenoiser import failed. Ensure deepdespeckling is installed."
            ) from exc

        kwargs = {"model_name": self.model_name}
        if self.symetrise is not None:
            kwargs["symetrise"] = self.symetrise
        model = None
        last_err: Exception | None = None
        attempts: list[tuple[tuple[Any, ...], dict[str, Any]]] = [((), kwargs)]
        if self.symetrise is not None:
            attempts.append(((self.model_name, self.symetrise), {}))
        attempts.append(((self.model_name,), {}))
        for args, kw in attempts:
            try:
                model = MerlinDenoiser(*args, **kw)
                break
            except TypeError as exc:
                last_err = exc
                continue
        if model is None:
            raise ConfigError(
                "Unable to instantiate MerlinDenoiser. Check deepdespeckling API and config merlin.model_name/device/symetrise."
            ) from last_err
        if self.device:
            model.device = self.device
        if hasattr(model, "eval"):
            model.eval()
        self._model_cache[key] = model
        return model

    def _prepare_input(self, re: torch.Tensor, im: torch.Tensor, amp: torch.Tensor, fmt: str) -> torch.Tensor:
        if fmt == "complex":
            x = torch.complex(re, im)
            x = x.unsqueeze(0).unsqueeze(0)
            return x.to(self.device)
        if fmt == "stacked":
            x = torch.stack([re, im], dim=0).unsqueeze(0)
            return x.to(self.device)
        if fmt == "hw2":
            x = torch.stack([re, im], dim=-1).unsqueeze(0)
            return x.to(self.device)
        if fmt == "amplitude":
            x = amp.unsqueeze(0).unsqueeze(0)
            return x.to(self.device)
        raise ConfigError(f"Unsupported merlin.input_format: {fmt}")

    def _call_model(self, model: Any, x: torch.Tensor) -> torch.Tensor:
        if hasattr(model, "denoise"):
            return model.denoise(x)
        if hasattr(model, "forward"):
            return model(x)
        raise ConfigError("MerlinDenoiser has no denoise/forward method; cannot call model")

    def denoise(self, re: torch.Tensor, im: torch.Tensor, amp: torch.Tensor) -> torch.Tensor:
        model = self._get_model()
        if hasattr(model, "denoise_image"):
            return self._denoise_with_deepdespeckling(model, re, im, amp)
        formats = [self.input_format]
        if self.input_format == "auto":
            formats = ["stacked", "complex", "amplitude", "hw2"]

        last_err = None
        for fmt in formats:
            try:
                x = self._prepare_input(re, im, amp, fmt)
                with torch.no_grad():
                    y = self._call_model(model, x)
                self._detected_format = fmt
                return self._normalize_output(y, amp)
            except Exception as exc:
                last_err = exc
                continue
        raise ConfigError(
            "Merlin adapter failed to run model for all input formats. "
            "Set merlin.input_format to one of: complex, stacked, amplitude, hw2."
        ) from last_err

    def _denoise_with_deepdespeckling(
        self,
        model: Any,
        re: torch.Tensor,
        im: torch.Tensor,
        amp: torch.Tensor,
    ) -> torch.Tensor:
        try:
            from deepdespeckling.utils.constants import PATCH_SIZE, STRIDE_SIZE
        except Exception as exc:
            raise ConfigError(
                "deepdespeckling.utils.constants import failed; cannot set default patch/stride size"
            ) from exc

        patch_size = self.patch_size if self.patch_size is not None else PATCH_SIZE
        stride_size = self.stride_size if self.stride_size is not None else STRIDE_SIZE
        if patch_size <= 0:
            raise ConfigError("merlin.patch_size must be > 0")
        if stride_size <= 0:
            raise ConfigError("merlin.stride_size must be > 0")
        if stride_size >= patch_size:
            raise ConfigError("merlin.stride_size must be < merlin.patch_size")

        noisy = torch.stack([re, im], dim=-1).detach().cpu().numpy()
        height, width = noisy.shape[0], noisy.shape[1]
        if patch_size > height or patch_size > width:
            raise ConfigError(
                f"merlin.patch_size ({patch_size}) must be <= input spatial dims ({height}, {width})"
            )

        result = model.denoise_image(noisy, patch_size=patch_size, stride_size=stride_size)
        denoised = result["denoised"]["full"]
        out = torch.as_tensor(denoised, dtype=amp.dtype, device=amp.device)
        return self._normalize_output(out, amp)

    def _normalize_output(self, y: Any, amp: torch.Tensor) -> torch.Tensor:
        if isinstance(y, torch.Tensor):
            out = y
        else:
            try:
                out = torch.as_tensor(y)
            except Exception as exc:
                raise ConfigError("Merlin output could not be converted to torch.Tensor") from exc

        while out.ndim > 2 and out.shape[0] == 1:
            out = out.squeeze(0)
        if out.ndim != 2:
            raise ConfigError(f"Merlin output must be 2D [H,W], got shape {tuple(out.shape)}")

        if torch.is_complex(out):
            raise ConfigError("Merlin output is complex; expected amplitude or intensity")

        if self.output_kind == "intensity":
            if (out < 0).any():
                raise ConfigError("Merlin output has negative values; cannot treat as intensity")
            out = torch.sqrt(out)
        else:
            if (out < 0).any():
                raise ConfigError(
                    "Merlin output has negative values; cannot treat as amplitude. "
                    "If model returns intensity, set merlin.output_kind='intensity'."
                )
        if out.shape != amp.shape:
            # allow broadcasting if shapes are compatible
            if out.shape[-2:] != amp.shape[-2:]:
                raise ConfigError(f"Merlin output shape {tuple(out.shape)} does not match input amplitude shape {tuple(amp.shape)}")
        return out


class MerlinDespeckleChannel(BaseChannel):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        merlin_cfg = cfg["merlin"]
        self.adapter = MerlinDenoiserAdapter(
            model_name=merlin_cfg["model_name"],
            device=merlin_cfg["device"],
            input_format=merlin_cfg.get("input_format", "auto"),
            output_kind=merlin_cfg.get("output_kind", "amplitude"),
            symetrise=merlin_cfg.get("symetrise", True),
            patch_size=merlin_cfg.get("patch_size"),
            stride_size=merlin_cfg.get("stride_size"),
        )

    def process(self, x: torch.Tensor, ctx: Dict[str, Any]):
        if "amplitude" not in ctx:
            raise ConfigError("Amplitude not found in ctx; ensure MerlinForge preprocess computes it")
        re = ctx.get("re")
        im = ctx.get("im")
        if re is None or im is None:
            raise ConfigError("re/im not found in ctx; ensure input is complex with last dim=2")
        amp = ctx["amplitude"]
        out = self.adapter.denoise(re, im, amp)
        ctx.setdefault("debug", {})["merlin_input_format"] = self.adapter._detected_format
        return out, ctx
