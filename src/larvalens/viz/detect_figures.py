"""Figures 7, 8 and 9: detection overlay, count agreement and training curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter

from larvalens import viz

MODEL_COLORS = {"yolov8n-seg": "#CC79A7", "yolov8s-seg": "#009E73", "yolov8m-seg": "#222222"}
MODEL_LABELS = {"yolov8n-seg": "YOLOv8n", "yolov8s-seg": "YOLOv8s", "yolov8m-seg": "YOLOv8m"}
GT_COLOR = "#009E73"
PRED_COLOR = "#CC79A7"


def read_yolo_polygons(path: Path, size: int = 640) -> list[np.ndarray]:
    """Read YOLO segmentation polygons as pixel coordinates."""
    polys = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) > 6:
            polys.append(np.array(parts[1:], float).reshape(-1, 2) * size)
    return polys


def _show(ax, image: np.ndarray, polys: list[np.ndarray], color: str) -> None:
    ax.imshow(image)
    if polys:
        ax.add_collection(
            PolyCollection(polys, facecolors=color + "55", edgecolors=color, linewidths=0.8)
        )
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)


def fig07_detection_overlay(
    images: list[Path], label_dir: Path, preds: dict, model: str, conf: float = 0.25
) -> Figure:
    """Labelled larvae (top row) above predicted larvae (bottom row) for a few test images."""
    n = len(images)
    fig, axes = viz.new_figure("double", 7.2 * 2 / n + 0.5, 2, n)
    axes = np.atleast_2d(axes)
    for col, path in enumerate(images):
        image = mpimg.imread(path)
        gt = read_yolo_polygons(label_dir / f"{path.stem}.txt")
        p = preds[model][path.name]
        keep = [
            np.array(poly)
            for poly, c in zip(p["polygons"], p["conf"], strict=True)
            if c >= conf and len(poly) >= 3
        ]
        _show(axes[0, col], image, gt, GT_COLOR)
        _show(axes[1, col], image, keep, PRED_COLOR)
        axes[0, col].set_xlabel(f"{len(gt)} labelled", fontsize=8, color=GT_COLOR)
        axes[1, col].set_xlabel(f"{len(keep)} predicted", fontsize=8, color=PRED_COLOR)
    axes[0, 0].set_ylabel("Labelled", color=GT_COLOR, fontweight="bold")
    axes[1, 0].set_ylabel(
        f"Predicted ({MODEL_LABELS.get(model, model)})", color=PRED_COLOR, fontweight="bold"
    )
    fig.suptitle(
        "The model finds the labelled larvae, and also ones the labellers skipped",
        x=0.01,
        ha="left",
        fontweight="bold",
        fontsize=9,
    )
    viz.panel_letters(axes.flatten())
    return fig


def fig08_count_agreement(counts: pd.DataFrame, models: list[str]) -> Figure:
    """Predicted vs labelled larvae per test image, one panel per model, with a 1:1 line."""
    fig, axes = viz.new_figure("double", 2.7, 1, len(models), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    top = max(counts["true_count"].max(), *(counts[f"pred_{m}"].max() for m in models)) * 1.5
    low = 0.8
    for ax, m in zip(axes, models, strict=True):
        y, p = counts["true_count"], counts[f"pred_{m}"]
        err = p - y
        ax.plot([low, top], [low, top], color="#999999", lw=0.8, ls="--")
        ax.scatter(y, p, s=14, color=MODEL_COLORS.get(m, "#222222"), alpha=0.8)
        ax.set_title(
            f"{MODEL_LABELS.get(m, m)}\n"
            f"median error {err.median():+.0f}, MAE {err.abs().mean():.1f}",
            loc="left",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(FixedLocator([1, 3, 10, 30, 100, 300]))
            axis.set_major_formatter(ScalarFormatter())
            axis.set_minor_formatter(NullFormatter())
        ax.set_xlim(low, top)
        ax.set_ylim(low, top)
        ax.set_aspect("equal")
        ax.set_xlabel("Labelled larvae per image")
    axes[0].set_ylabel("Predicted larvae per image")
    viz.panel_letters(axes)
    return fig


def _spread(values: dict[str, float], gap: float) -> dict[str, float]:
    """Nudge label positions apart so no two are closer than `gap`."""
    order = sorted(values, key=values.get)
    out: dict[str, float] = {}
    prev = -np.inf
    for k in order:
        out[k] = max(values[k], prev + gap)
        prev = out[k]
    return out


def fig09_training_curves(epochs: pd.DataFrame, models: list[str]) -> Figure:
    """Segmentation loss and mask mAP50-95 per epoch for each model."""
    fig, (ax, bx) = viz.new_figure("double", 2.6, 1, 2)
    ends = {}
    for m in models:
        e = epochs[epochs["model"] == m]
        c = MODEL_COLORS.get(m, "#222222")
        ax.plot(e["epoch"], e["train/seg_loss"], color=c, lw=1.0, alpha=0.45)
        ax.plot(e["epoch"], e["val/seg_loss"], color=c, lw=1.3)
        bx.plot(e["epoch"], e["metrics/mAP50-95(M)"], color=c, lw=1.3)
        ends[m] = float(e["metrics/mAP50-95(M)"].iloc[-5:].mean())
    for m, y in _spread(ends, 0.05).items():
        viz.label_line(
            bx, epochs["epoch"].max(), y, MODEL_LABELS.get(m, m), MODEL_COLORS.get(m, "#222222")
        )
    late = epochs[epochs["epoch"] > 10]
    ax.set_ylim(0, max(late["val/seg_loss"].max(), late["train/seg_loss"].max()) * 1.3)
    ax.set_xlabel("Epoch (faint: train, solid: validation; early spikes are cut off)")
    ax.set_ylabel("Segmentation loss")
    bx.set_xlabel("Epoch")
    bx.set_ylabel("Validation mask mAP50-95")
    bx.set_xlim(right=epochs["epoch"].max() * 1.2)
    bx.set_ylim(0, max(_spread(ends, 0.05).values()) + 0.06)
    ax.set_title("Loss settles after about 20 epochs", loc="left")
    bx.set_title("All three sizes reach similar mask accuracy", loc="left")
    viz.panel_letters([ax, bx])
    return fig
