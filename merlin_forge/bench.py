from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from .config import load_config
from .forge import MerlinForge
from .io import load_npy


def _iter_inputs(input_dir: Path):
    for p in sorted(input_dir.glob("*.npy")):
        yield p


def _rss_mb():
    try:
        import psutil

        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="MERLIN-forge benchmark")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()

    cfg = load_config(args.config)
    input_dir = Path(args.input_dir)
    paths = list(_iter_inputs(input_dir))
    if not paths:
        raise RuntimeError(f"No .npy files found in {input_dir}")

    model = MerlinForge(cfg)
    warmup = cfg.get("bench", {}).get("warmup", 5)
    iters = min(args.n, len(paths))

    def run_once(path: Path):
        x = load_npy(path, cfg)
        y, _ = model(x)
        return y

    # Warmup
    for i in range(min(warmup, len(paths))):
        run_once(paths[i])

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    rss_start = _rss_mb()
    times = []
    for i in range(iters):
        start = time.perf_counter()
        run_once(paths[i])
        end = time.perf_counter()
        times.append((end - start) * 1000.0)

    rss_end = _rss_mb()
    rss_peak = None
    if rss_start is not None and rss_end is not None:
        rss_peak = max(rss_start, rss_end)

    if torch.cuda.is_available():
        cuda_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        cuda_peak = None

    avg_ms = sum(times) / len(times)
    throughput = 1000.0 / avg_ms if avg_ms > 0 else 0.0

    print(f"samples: {iters}")
    print(f"avg_ms: {avg_ms:.3f}")
    print(f"throughput_sps: {throughput:.2f}")
    if rss_peak is not None:
        print(f"rss_peak_mb: {rss_peak:.2f}")
    if cuda_peak is not None:
        print(f"cuda_peak_mb: {cuda_peak:.2f}")

    # Report output
    y = run_once(paths[0])
    print(f"output_dtype: {y.dtype}")
    print(f"output_shape: {tuple(y.shape)}")


if __name__ == "__main__":
    main()
