#!/usr/bin/env python
"""Step 2: Turn the calibration traces into the tutorial figures.

Inputs  (produced by 01_ptq_with_histogram_trace.py):
    calib_trace/hist_trace.pkl
    calib_trace/amax_trace.json
    calib_trace/method_comparison.json
    calib_trace/final_amax_all_quantizers.pkl
    checkpoints/original_release_amax.pth   (from 00_reconstruct_fp_checkpoint.py)

Outputs (figures/):
    hist_evolution_<layer>.png     histogram snapshots at selected sample counts
    hist_heatmap_<layer>.png       full per-sample histogram evolution as a heatmap
    amax_trajectory.png            MSE-amax vs #calibration samples (several layers)
    method_comparison_<layer>.png  final histogram + amax candidates of each method
    amax_repro_vs_release.png      reproduced amax vs original release amax
    weight_amax_per_channel.png    per-channel weight amax example

Run in the container (matplotlib + torch required):
    python work_dirs/centerpoint_tutorial/scripts/02_plot_calibration.py
"""

import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]  # work_dirs/centerpoint_tutorial
TRACE = ROOT / "calib_trace"
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)

# Layers featured in the tutorial figures: shallow → deep → head.
FEATURED = [
    "pts_backbone.blocks.1.0._input_quantizer",
    "pts_backbone.blocks.2.0._input_quantizer",
    "pts_neck.deblocks.0.0._input_quantizer",
    "pts_bbox_head.shared_conv.conv._input_quantizer",
]


def short(name: str) -> str:
    return (
        name.replace("pts_backbone.", "backbone.")
        .replace("pts_neck.", "neck.")
        .replace("pts_bbox_head.", "head.")
        .replace("._input_quantizer", "")
    )


def load_traces():
    with open(TRACE / "hist_trace.pkl", "rb") as f:
        hist_trace = pickle.load(f)
    with open(TRACE / "amax_trace.json") as f:
        amax_trace = json.load(f)
    with open(TRACE / "method_comparison.json") as f:
        methods = json.load(f)
    return hist_trace, amax_trace, methods


def centers(edges):
    return 0.5 * (edges[:-1] + edges[1:])


def plot_hist_evolution(hist_trace, layer):
    """Small multiples: the histogram after 1 / 5 / 15 / 30 / all samples."""
    n = len(hist_trace)
    picks = sorted({0, 1, 4, 14, 29, n - 1} & set(range(n)))
    fig, axes = plt.subplots(1, len(picks), figsize=(3.2 * len(picks), 3.0), sharey=True)
    for ax, idx in zip(np.atleast_1d(axes), picks):
        snap = hist_trace[idx].get(layer)
        if snap is None:
            continue
        h, e = snap["hist"], snap["edges"]
        ax.semilogy(centers(e), np.maximum(h, 0.5), lw=0.8)
        ax.set_title(f"after sample {idx + 1}\nrange [0, {e[-1]:.1f}]", fontsize=9)
        ax.set_xlabel("|activation|")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("count (log)")
    fig.suptitle(f"Histogram evolution — {short(layer)}", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / f"hist_evolution_{short(layer).replace('.', '_')}.png", dpi=140)
    plt.close(fig)


def plot_hist_heatmap(hist_trace, layer):
    """Heatmap: x = calibration sample, y = |activation| bin, color = log10(count)."""
    # Re-bin every snapshot onto the FINAL bin grid so rows are comparable.
    final = hist_trace[-1][layer]
    fe = final["edges"]
    grid = np.zeros((len(hist_trace), len(fe) - 1))
    for i, snap in enumerate(hist_trace):
        s = snap.get(layer)
        if s is None:
            continue
        h, e = s["hist"], s["edges"]
        c = centers(e)
        idx = np.clip(np.searchsorted(fe, c) - 1, 0, len(fe) - 2)
        np.add.at(grid[i], idx, h)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    im = ax.imshow(
        np.log10(grid.T + 1),
        aspect="auto",
        origin="lower",
        extent=[1, len(hist_trace), fe[0], fe[-1]],
        cmap="viridis",
    )
    ax.set_xlabel("calibration sample #")
    ax.set_ylabel("|activation|")
    ax.set_title(f"Histogram fill-up over calibration — {short(layer)}")
    fig.colorbar(im, label="log10(count+1)")
    fig.tight_layout()
    fig.savefig(FIGS / f"hist_heatmap_{short(layer).replace('.', '_')}.png", dpi=140)
    plt.close(fig)


def plot_amax_trajectory(amax_trace):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    steps = [t["step"] + 1 for t in amax_trace]
    layers = sorted(amax_trace[-1]["amax"].keys())
    for layer in layers:
        ys = [t["amax"].get(layer) for t in amax_trace]
        if all(y is None for y in ys):
            continue
        featured = layer in FEATURED
        ax.plot(
            steps,
            [np.nan if y is None else y for y in ys],
            lw=1.8 if featured else 0.6,
            alpha=1.0 if featured else 0.35,
            label=short(layer) if featured else None,
        )
    ax.set_xlabel("calibration samples seen")
    ax.set_ylabel("amax chosen by MSE criterion")
    ax.set_title("amax convergence during calibration (all input quantizers; featured layers bold)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "amax_trajectory.png", dpi=140)
    plt.close(fig)


def plot_method_comparison(hist_trace, methods, layer):
    final = hist_trace[-1][layer]
    h, e = final["hist"], final["edges"]
    m = methods[layer]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogy(centers(e), np.maximum(h, 0.5), lw=0.8, color="gray", label="final histogram")
    for key, color in [
        ("mse", "tab:red"),
        ("entropy", "tab:blue"),
        ("percentile_99.9", "tab:green"),
        ("percentile_99.99", "tab:olive"),
        ("max", "tab:purple"),
    ]:
        v = m.get(key)
        if isinstance(v, (int, float)):
            ax.axvline(v, color=color, ls="--", lw=1.5, label=f"{key}: {v:.2f}")
    ax.set_xlabel("|activation|")
    ax.set_ylabel("count (log)")
    ax.set_title(f"Where each calibration method puts amax — {short(layer)}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / f"method_comparison_{short(layer).replace('.', '_')}.png", dpi=140)
    plt.close(fig)


def plot_repro_vs_release():
    release = torch.load(ROOT / "checkpoints" / "original_release_amax.pth", map_location="cpu", weights_only=True)
    with open(TRACE / "final_amax_all_quantizers.pkl", "rb") as f:
        ours = pickle.load(f)
    xs, ys, labels = [], [], []
    for k, v in release.items():
        name = k.replace("._amax", "")
        if name in ours and "_input_quantizer" in name:
            xs.append(float(v.flatten()[0]))
            ys.append(float(np.asarray(ours[name]).flatten()[0]))
            labels.append(short(name))
    xs, ys = np.array(xs), np.array(ys)
    fig, ax = plt.subplots(figsize=(6.4, 6))
    lim = max(xs.max(), ys.max()) * 1.1
    ax.plot([0, lim], [0, lim], color="gray", lw=1, ls=":")
    ax.scatter(xs, ys, s=28)
    for x, y, l in zip(xs, ys, labels):
        if abs(y - x) / max(x, 1e-6) > 0.25:
            ax.annotate(l, (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("release amax (400 full-dataset samples)")
    ax.set_ylabel("reproduced amax (60 local samples)")
    ax.set_title("Activation amax: reproduction vs release calibration")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "amax_repro_vs_release.png", dpi=140)
    plt.close(fig)
    rel = np.abs(ys - xs) / np.maximum(xs, 1e-6)
    print(f"amax repro vs release: median rel diff {np.median(rel):.1%}, max {rel.max():.1%}")


def plot_weight_amax():
    with open(TRACE / "final_amax_all_quantizers.pkl", "rb") as f:
        ours = pickle.load(f)
    name = "pts_backbone.blocks.1.0._weight_quantizer"
    if name not in ours:
        cands = [k for k in ours if "_weight_quantizer" in k]
        if not cands:
            return
        name = cands[0]
    a = np.asarray(ours[name]).flatten()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(np.arange(len(a)), a, width=0.9)
    ax.set_xlabel("output channel")
    ax.set_ylabel("weight amax")
    ax.set_title(
        f"Per-channel weight amax (MaxCalibrator) — {short(name).replace('._weight_quantizer','')} [{len(a)} ch]"
    )
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGS / "weight_amax_per_channel.png", dpi=140)
    plt.close(fig)


def main():
    hist_trace, amax_trace, methods = load_traces()
    print(f"snapshots: {len(hist_trace)}; layers: {len(hist_trace[-1])}")
    available = set(hist_trace[-1].keys())
    featured = [l for l in FEATURED if l in available]
    if not featured:  # fall back to whatever exists
        featured = sorted(available)[:4]
    for layer in featured:
        plot_hist_evolution(hist_trace, layer)
        plot_hist_heatmap(hist_trace, layer)
        plot_method_comparison(hist_trace, methods, layer)
    plot_amax_trajectory(amax_trace)
    plot_repro_vs_release()
    plot_weight_amax()
    print(f"figures written to {FIGS}/")


if __name__ == "__main__":
    main()
