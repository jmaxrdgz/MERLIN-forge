from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable

import yaml


class ConfigError(ValueError):
    pass


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ConfigError(f"Config at {path} is empty")
    validate_config(cfg)
    return cfg


def _ensure_in(value: Any, allowed: Iterable[Any], path: str) -> Any:
    if value not in allowed:
        raise ConfigError(f"Invalid value for {path}: {value}. Allowed: {list(allowed)}")
    return value


def _ensure_type(value: Any, expected_type: type, path: str) -> Any:
    if not isinstance(value, expected_type):
        raise ConfigError(f"Invalid type for {path}: {type(value).__name__}, expected {expected_type.__name__}")
    return value


def validate_config(cfg: Dict[str, Any]) -> None:
    if not isinstance(cfg, dict):
        raise ConfigError("Config root must be a mapping")

    input_cfg = cfg.setdefault("input", {})
    _ensure_type(input_cfg, dict, "input")
    input_cfg.setdefault("expected_shape_last_dim", 2)
    input_cfg.setdefault("dtype", "float32")

    merlin_cfg = cfg.setdefault("merlin", {})
    _ensure_type(merlin_cfg, dict, "merlin")
    merlin_cfg.setdefault("model_name", "spotlight")
    merlin_cfg.setdefault("device", "cpu")
    merlin_cfg.setdefault("output_kind", "amplitude")
    merlin_cfg.setdefault("input_format", "auto")
    merlin_cfg.setdefault("symetrise", True)
    merlin_cfg.setdefault("patch_size", None)
    merlin_cfg.setdefault("stride_size", None)
    _ensure_in(merlin_cfg["output_kind"], {"amplitude", "intensity"}, "merlin.output_kind")
    _ensure_type(merlin_cfg["symetrise"], bool, "merlin.symetrise")
    if merlin_cfg["patch_size"] is not None:
        _ensure_type(merlin_cfg["patch_size"], int, "merlin.patch_size")
    if merlin_cfg["stride_size"] is not None:
        _ensure_type(merlin_cfg["stride_size"], int, "merlin.stride_size")

    aq_cfg = cfg.setdefault("adaptive_quantization", {})
    _ensure_type(aq_cfg, dict, "adaptive_quantization")
    aq_cfg.setdefault("bins", 512)
    aq_cfg.setdefault("mode_range_percentiles", [1.0, 99.5])
    aq_cfg.setdefault("alpha_weak", 0.7)
    aq_cfg.setdefault("alpha_strong", 1.5)
    aq_cfg.setdefault("clip_percentiles", [0.0, 99.9])
    aq_cfg.setdefault("clip_max", None)
    aq_cfg.setdefault("weak_mapping", {"out_min": 0.0, "out_split": 0.85, "out_max": 1.0})
    aq_cfg.setdefault("strong_mapping", {"out_min": 0.0, "out_split": 0.15, "out_max": 1.0})

    output_cfg = cfg.setdefault("output", {})
    _ensure_type(output_cfg, dict, "output")
    output_cfg.setdefault("layout", "CHW")
    output_cfg.setdefault("dtype", "float32")
    output_cfg.setdefault("uint16", {"range_mode": "use_output_endpoints", "clamp": True, "endpoints": None})
    _ensure_in(output_cfg["layout"], {"CHW", "HWC"}, "output.layout")
    _ensure_in(output_cfg["dtype"], {"float32", "uint16"}, "output.dtype")

    viz_cfg = cfg.setdefault("viz", {})
    _ensure_type(viz_cfg, dict, "viz")
    viz_cfg.setdefault("show_hist", True)
    viz_cfg.setdefault("overlay_channel_hists", False)

    bench_cfg = cfg.setdefault("bench", {})
    _ensure_type(bench_cfg, dict, "bench")
    bench_cfg.setdefault("warmup", 5)

    # Basic sanity checks
    if input_cfg["expected_shape_last_dim"] != 2:
        raise ConfigError("input.expected_shape_last_dim must be 2 for complex re/imag inputs")

    if aq_cfg["bins"] <= 1:
        raise ConfigError("adaptive_quantization.bins must be > 1")
    if len(aq_cfg["mode_range_percentiles"]) != 2:
        raise ConfigError("adaptive_quantization.mode_range_percentiles must have 2 values")
    if len(aq_cfg["clip_percentiles"]) != 2:
        raise ConfigError("adaptive_quantization.clip_percentiles must have 2 values")

    for key in ("weak_mapping", "strong_mapping"):
        mapping = aq_cfg.get(key, {})
        if not isinstance(mapping, dict):
            raise ConfigError(f"adaptive_quantization.{key} must be a mapping")
        for mkey in ("out_min", "out_split", "out_max"):
            if mkey not in mapping:
                raise ConfigError(f"adaptive_quantization.{key}.{mkey} is required")

    uint16_cfg = output_cfg.get("uint16", {})
    if output_cfg["dtype"] == "uint16":
        _ensure_type(uint16_cfg, dict, "output.uint16")
        uint16_cfg.setdefault("range_mode", "use_output_endpoints")
        uint16_cfg.setdefault("clamp", True)
        uint16_cfg.setdefault("endpoints", None)
        _ensure_in(uint16_cfg["range_mode"], {"use_output_endpoints", "minmax_per_channel"}, "output.uint16.range_mode")
        if uint16_cfg["range_mode"] == "use_output_endpoints" and uint16_cfg["endpoints"] is None:
            raise ConfigError(
                "output.uint16.endpoints is required when range_mode='use_output_endpoints'. "
                "Provide per-channel [min,max] endpoints matching output order."
            )

    # Fill back possibly updated nested configs
    cfg["input"] = input_cfg
    cfg["merlin"] = merlin_cfg
    cfg["adaptive_quantization"] = aq_cfg
    cfg["output"] = output_cfg
    cfg["viz"] = viz_cfg
    cfg["bench"] = bench_cfg
