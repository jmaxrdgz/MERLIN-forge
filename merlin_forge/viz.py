from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .config import load_config
from .forge import MerlinForge
from .io import load_npy


def _to_numpy(t: torch.Tensor):
    return t.detach().cpu().numpy()


def _prep_display(t: torch.Tensor, cfg: dict, label: str):
    arr = _to_numpy(t)
    viz_cfg = cfg.get("viz", {})
    if viz_cfg.get("warn_nonfinite", True):
        if not np.isfinite(arr).all():
            raise RuntimeError(f"{label} contains NaN/Inf values; check input scaling or merlin denoiser normalization.")
    if viz_cfg.get("log_scale", False):
        arr = np.log1p(np.clip(arr, 0.0, None))
    clip = viz_cfg.get("clip_percentiles")
    vmin = vmax = None
    if clip is not None:
        lo, hi = float(clip[0]), float(clip[1])
        finite = arr[np.isfinite(arr)]
        if finite.size:
            vmin, vmax = np.percentile(finite, [lo, hi]).tolist()
    return arr, vmin, vmax


def main():
    parser = argparse.ArgumentParser(description="MERLIN-forge visualization")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    x = load_npy(args.input, cfg)
    model = MerlinForge(cfg)
    y, ctx = model(x)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for visualization") from exc

    amp = ctx["amplitude"]
    layout = cfg["output"].get("layout", "CHW")

    if layout == "CHW":
        merlin_ch = y[0]
        weak_ch = y[1]
        strong_ch = y[2]
    else:
        merlin_ch = y[..., 0]
        weak_ch = y[..., 1]
        strong_ch = y[..., 2]

    show_hist = cfg.get("viz", {}).get("show_hist", True)
    if show_hist:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        ax_img = axes.flatten()
    else:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        ax_img = axes.flatten()

    amp_img, amp_vmin, amp_vmax = _prep_display(amp, cfg, "Amplitude")
    merlin_img, merlin_vmin, merlin_vmax = _prep_display(merlin_ch, cfg, "MERLIN")
    weak_img, weak_vmin, weak_vmax = _prep_display(weak_ch, cfg, "Weak")
    strong_img, strong_vmin, strong_vmax = _prep_display(strong_ch, cfg, "Strong")

    ax_img[0].imshow(amp_img, cmap="gray", vmin=amp_vmin, vmax=amp_vmax)
    ax_img[0].set_title("Amplitude")
    ax_img[1].imshow(merlin_img, cmap="gray", vmin=merlin_vmin, vmax=merlin_vmax)
    ax_img[1].set_title("MERLIN")
    ax_img[2].imshow(weak_img, cmap="gray", vmin=weak_vmin, vmax=weak_vmax)
    ax_img[2].set_title("Weak")
    ax_img[3].imshow(strong_img, cmap="gray", vmin=strong_vmin, vmax=strong_vmax)
    ax_img[3].set_title("Strong")

    for ax in ax_img:
        ax.axis("off")

    if show_hist:
        fig_hist, axh = plt.subplots(1, 1, figsize=(6, 4))
        amp_flat = _to_numpy(amp).ravel()
        viz_cfg = cfg.get("viz", {})
        hist_clip = viz_cfg.get("hist_clip_percentiles", viz_cfg.get("clip_percentiles"))
        if hist_clip is not None:
            lo, hi = float(hist_clip[0]), float(hist_clip[1])
            finite = amp_flat[np.isfinite(amp_flat)]
            if finite.size:
                vmin, vmax = np.percentile(finite, [lo, hi]).tolist()
                amp_hist = np.clip(amp_flat, vmin, vmax)
            else:
                amp_hist = amp_flat
                vmin = vmax = None
        else:
            amp_hist = amp_flat
            vmin = vmax = None
        bins = cfg["adaptive_quantization"]["bins"]
        hist_range = (vmin, vmax) if vmin is not None and vmax is not None else None
        axh.hist(amp_hist, bins=bins, range=hist_range, color="gray", alpha=0.8, label="Amplitude")
        if vmin is not None and vmax is not None:
            axh.set_xlim(vmin, vmax)
        stats = ctx.get("stats", {})
        for label, key, color in (
            ("P", "P", "orange"),
            ("T_weak", "T_weak", "blue"),
            ("T_strong", "T_strong", "red"),
        ):
            if key in stats:
                axh.axvline(stats[key], color=color, linestyle="--", label=label)
        axh.legend()
        hist_path = Path(args.out).with_suffix(".hist.png")
        fig_hist.tight_layout()
        fig_hist.savefig(hist_path, dpi=150)
        plt.close(fig_hist)

        if viz_cfg.get("overlay_channel_hists", False):
            fig_ch, axc = plt.subplots(1, 1, figsize=(6, 4))
            for label, data, color in (
                ("Weak", _to_numpy(weak_ch).ravel(), "blue"),
                ("Strong", _to_numpy(strong_ch).ravel(), "red"),
            ):
                data = data[np.isfinite(data)]
                axc.hist(
                    data,
                    bins=bins,
                    range=(0.0, 1.0),
                    histtype="step",
                    linewidth=1.2,
                    color=color,
                    label=label,
                )
            axc.set_xlim(0.0, 1.0)
            axc.legend()
            channels_hist_path = Path(args.out).with_suffix(".channels.hist.png")
            fig_ch.tight_layout()
            fig_ch.savefig(channels_hist_path, dpi=150)
            plt.close(fig_ch)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
