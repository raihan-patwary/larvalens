"""Figures 7, 8 and 9: detection overlay, count agreement and training curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure

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
    """Ground truth beside prediction for a few test images (one row per image)."""
    n = len(images)
    fig, axes = viz.new_figure("double", 3.5 * n / 2 + 0.4, n, 2)
    axes = np.atleast_2d(axes)
    for row, path in enumerate(images):
        image = mpimg.imread(path)
        gt = read_yolo_polygons(label_dir / f"{path.stem}.txt")
        p = preds[model][path.name]
        keep = [
            np.array(poly)
            for poly, c in zip(p["polygons"], p["conf"], strict=True)
            if c >= conf and len(poly) >= 3
        ]
        _show(axes[row, 0], image, gt, GT_COLOR)
        _show(axes[row, 1], image, keep, PRED_COLOR)
        axes[row, 0].set_ylabel(f"{len(gt)} labelled", fontsize=8)
        axes[row, 1].set_ylabel(f"{len(keep)} predicted", fontsize=8)
    axes[0, 0].set_title("Labelled (ground truth)", color=GT_COLOR, loc="left")
    axes[0, 1].set_title(
        f"Predicted by {MODEL_LABELS.get(model, model)}", color=PRED_COLOR, loc="left"
    )
    viz.panel_letters(axes.flatten())
    return fig


def fig08_count_agreement(counts: pd.DataFrame, models: list[str]) -> Figure:
    """Predicted vs labelled larvae per test image, one panel per model, with a 1:1 line."""
    fig, axes = viz.new_figure("double", 2.6, 1, len(models), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    top = max(counts["true_count"].max(), *(counts[f"pred_{m}"].max() for m in models)) * 1.08
    for ax, m in zip(axes, models, strict=True):
        y, p = counts["true_count"], counts[f"pred_{m}"]
        ax.plot([0, top], [0, top], color="#999999", lw=0.8, ls="--")
        ax.scatter(y, p, s=14, color=MODEL_COLORS.get(m, "#222222"), alpha=0.8)
        mae = (p - y).abs().mean()
        bias = (p - y).mean()
        ax.text(
            0.04,
            0.96,
            f"MAE {mae:.1f} larvae\nbias {bias:+.1f}",
            transform=ax.transAxes,
            va="top",
            fontsize=7.5,
        )
        ax.set_title(MODEL_LABELS.get(m, m), loc="left")
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.set_aspect("equal")
        ax.set_xlabel("Labelled larvae per image")
    axes[0].set_ylabel("Predicted larvae per image")
    viz.panel_letters(axes)
    return fig


def fig09_training_curves(epochs: pd.DataFrame, models: list[str]) -> Figure:
    """Segmentation loss and mask mAP50-95 per epoch for each model."""
    fig, (ax, bx) = viz.new_figure("double", 2.6, 1, 2)
    for m in models:
        e = epochs[epochs["model"] == m]
        c = MODEL_COLORS.get(m, "#222222")
        ax.plot(e["epoch"], e["train/seg_loss"], color=c, lw=1.0, alpha=0.45)
        ax.plot(e["epoch"], e["val/seg_loss"], color=c, lw=1.3)
        bx.plot(e["epoch"], e["metrics/mAP50-95(M)"], color=c, lw=1.3)
        last = e.iloc[-1]
        viz.label_line(bx, last["epoch"], last["metrics/mAP50-95(M)"], MODEL_LABELS.get(m, m), c)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Segmentation loss")
    ax.text(
        0.98,
        0.95,
        "faint: train, solid: validation",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color="#555555",
    )
    bx.set_xlabel("Epoch")
    bx.set_ylabel("Validation mask mAP50-95")
    bx.set_xlim(right=epochs["epoch"].max() * 1.18)
    ax.set_title("Loss falls and levels off", loc="left")
    bx.set_title("Mask accuracy on validation photos", loc="left")
    viz.panel_letters([ax, bx])
    return fig
