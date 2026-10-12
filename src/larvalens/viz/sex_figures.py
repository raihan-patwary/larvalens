"""Figures 16 and 17: sex classification results and Grad-CAM gallery."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from PIL import Image
from sklearn.metrics import precision_recall_curve, roc_curve

from larvalens import viz
from larvalens.sex import normalised_confusion


def fig16_sex_results(cv: pd.DataFrame, scores: pd.Series) -> Figure:
    """Normalised confusion matrix, ROC curve and precision-recall curve (pooled out-of-fold)."""
    fig, axes = viz.new_figure("double", 2.7, 1, 3)
    cm = normalised_confusion(cv["sex"], cv["p_male"])
    ax = axes[0]
    ax.imshow(cm, cmap="viridis", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{cm[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if cm[i, j] < 0.6 else "#222222",
                fontweight="bold",
            )
    ax.set_xticks([0, 1], ["female", "male"])
    ax.set_yticks([0, 1], ["female", "male"])
    for t, c in zip(ax.get_yticklabels(), [viz.COLORS["female"], viz.COLORS["male"]], strict=True):
        t.set_color(c)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Balanced accuracy {scores['balanced_accuracy']:.2f}", loc="left")
    y = (cv["sex"] == "M").astype(int)
    fpr, tpr, _ = roc_curve(y, cv["p_male"])
    axes[1].plot([0, 1], [0, 1], color="#999999", ls="--", lw=0.8)
    axes[1].plot(fpr, tpr, color=viz.COLORS["male"], lw=1.5)
    axes[1].set_xlabel("False positive rate")
    axes[1].set_ylabel("True positive rate (male)")
    axes[1].set_title(f"ROC, AUC {scores['roc_auc']:.2f}", loc="left")
    prec, rec, _ = precision_recall_curve(y, cv["p_male"])
    axes[2].plot(rec, prec, color=viz.COLORS["male"], lw=1.5)
    axes[2].axhline(y.mean(), color="#999999", ls="--", lw=0.8)
    axes[2].set_xlabel("Recall (male); dashed line = chance")
    axes[2].set_ylabel("Precision (male)")
    axes[2].set_title(f"Precision-recall, AP {scores['pr_auc_male']:.2f}", loc="left")
    for a in axes[1:]:
        a.set_xlim(0, 1)
        a.set_ylim(0, 1.02)
    fig.suptitle(
        "Sex is barely predictable from one photo of a larva from an unseen batch",
        x=0.01,
        ha="left",
        fontweight="bold",
        fontsize=9,
    )
    viz.panel_letters(axes)
    return fig


def fig17_gradcam(images: dict[str, np.ndarray], picks: pd.DataFrame, cams: dict) -> Figure:
    """Grad-CAM overlays: rows are male right, male wrong, female right, female wrong."""
    groups = [
        ("M", True, "male, right"),
        ("M", False, "male, wrong"),
        ("F", True, "female, right"),
        ("F", False, "female, wrong"),
    ]
    ncols = 8
    fig, axes = viz.new_figure(
        "double", 4.2, 4, ncols, gridspec_kw={"wspace": 0.04, "hspace": 0.12}
    )
    for row, (sex, right, label) in enumerate(groups):
        g = picks[(picks["sex"] == sex) & (picks["right"] == right)].head(ncols)
        color = viz.COLORS["male"] if sex == "M" else viz.COLORS["female"]
        axes[row, 0].set_ylabel(label, color=color, fontweight="bold", fontsize=7.5)
        for col in range(ncols):
            ax = axes[row, col]
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if col >= len(g):
                continue
            rec = g.iloc[col]
            img = images[rec["file"]]
            cam = (
                np.asarray(
                    Image.fromarray((cams[rec["key"]] * 255).astype(np.uint8)).resize(
                        img.shape[1::-1], Image.BILINEAR
                    )
                )
                / 255
            )
            ax.imshow(img)
            ax.imshow(cam, cmap="viridis", alpha=0.45, vmin=0, vmax=1)
            ax.text(
                0.03,
                0.03,
                f"p(male) {rec['p_male']:.2f}",
                transform=ax.transAxes,
                fontsize=5.5,
                color="white",
            )
    fig.suptitle(
        "Where the network looks (Grad-CAM, brighter = more influence)",
        x=0.01,
        ha="left",
        fontweight="bold",
        fontsize=9,
    )
    return fig
