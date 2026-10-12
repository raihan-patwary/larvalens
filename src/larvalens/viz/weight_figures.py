"""Figures 2, 6 and 10 to 14: crops, image QC, morphometrics, calibration and weight models."""

from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from scipy import ndimage
from skimage.morphology import skeletonize

from larvalens import viz
from larvalens.morphometrics import largest_component, longest_skeleton_path

PAPER = {"Morphometric (paper)": 0.80, "CNN regression (paper)": 0.71}


def _thumb(ax, path: Path, border: str | None = None) -> None:
    ax.imshow(mpimg.imread(path))
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(border is not None)
        if border:
            side.set_edgecolor(border)
            side.set_linewidth(2)


def fig02_mosaic(df: pd.DataFrame, crop_dir: Path, n: int = 60, ncols: int = 10) -> Figure:
    """60 fixed-size crops from lightest to heaviest; border colour shows weight."""
    d = df.dropna(subset=["weight_g"]).sort_values("weight_g").reset_index(drop=True)
    picks = d.iloc[np.linspace(0, len(d) - 1, n).round().astype(int)]
    norm = Normalize(d["weight_g"].min(), d["weight_g"].max())
    nrows = int(np.ceil(n / ncols))
    fig, axes = viz.new_figure(
        "double",
        7.2 * nrows / ncols + 0.9,
        nrows,
        ncols,
        gridspec_kw={"wspace": 0.04, "hspace": 0.04},
    )
    for ax, rec in zip(axes.flat, picks.itertuples(), strict=False):
        _thumb(ax, crop_dir / f"{Path(rec.copy).stem}.jpg", cm.viridis(norm(rec.weight_g)))
        ax.text(
            0.04,
            0.04,
            f"{rec.weight_g * 1000:.0f} mg",
            transform=ax.transAxes,
            fontsize=5.5,
            color="#222222",
        )
    for ax in axes.flat[len(picks) :]:
        ax.axis("off")
    fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap="viridis"),
        ax=axes,
        orientation="horizontal",
        fraction=0.03,
        pad=0.02,
        label="Weight (g)",
    )
    fig.suptitle(
        "Heavier larvae look longer and wider in the same fixed window",
        x=0.01,
        ha="left",
        fontweight="bold",
        fontsize=9,
    )
    return fig


def fig06_image_qc(qc: pd.DataFrame, crop_dir: Path, k: int = 3) -> Figure:
    """Blur vs brightness for every photo, with the most extreme photos as thumbnails."""
    fig = viz.new_figure("double", 3.4)[0]
    fig.clf()
    gs = fig.add_gridspec(2, 2 + k, width_ratios=[1.6, 1.6] + [1] * k)
    ax = fig.add_subplot(gs[:, :2])
    z = (qc[["blur", "brightness"]] - qc[["blur", "brightness"]].median()) / qc[
        ["blur", "brightness"]
    ].std()
    qc = qc.assign(score=np.hypot(z["blur"], z["brightness"]))
    out = qc.nlargest(2 * k, "score")
    ax.scatter(qc["blur"], qc["brightness"], s=5, color="#999999", alpha=0.6, lw=0)
    ax.scatter(out["blur"], out["brightness"], s=18, color=viz.COLORS["purple"], zorder=3)
    for i, rec in enumerate(out.itertuples()):
        ax.annotate(
            str(i + 1),
            (rec.blur, rec.brightness),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
            color=viz.COLORS["purple"],
        )
        tax = fig.add_subplot(gs[i // k, 2 + i % k])
        _thumb(tax, crop_dir / f"{Path(rec.copy).stem}.jpg", viz.COLORS["purple"])
        tax.set_title(f"{i + 1}", fontsize=7, loc="left")
    ax.set_xlabel("Sharpness (variance of the Laplacian)")
    ax.set_ylabel("Mean brightness (0 to 255)")
    ax.set_title("Image quality is even; the most unusual photos still look usable", loc="left")
    return fig


def fig10_morphometrics(crop: np.ndarray, mask: np.ndarray, mm_per_px: float) -> Figure:
    """One larva: photo, mask, skeleton with length, and the width map."""
    mask = largest_component(mask)
    dt = ndimage.distance_transform_edt(mask)
    path = longest_skeleton_path(skeletonize(mask))
    ys, xs = np.nonzero(mask)
    pad = 20
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, mask.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, mask.shape[1])
    win = (slice(y0, y1), slice(x0, x1))
    fig, axes = viz.new_figure("double", 2.3, 1, 4)
    axes[0].imshow(crop[win])
    axes[1].imshow(mask[win], cmap="Greys")
    axes[2].imshow(crop[win], alpha=0.35)
    axes[2].plot(path[:, 1] - x0, path[:, 0] - y0, color=viz.COLORS["female"], lw=1.6)
    ends = path[[0, -1]]
    tips = dt[ends[:, 0].round().astype(int), ends[:, 1].round().astype(int)]
    length_px = np.linalg.norm(np.diff(path, axis=0), axis=1).sum() + tips.sum()
    axes[2].text(
        0.03,
        0.03,
        f"length {length_px * mm_per_px:.1f} mm",
        transform=axes[2].transAxes,
        fontsize=7.5,
        color=viz.COLORS["female"],
        fontweight="bold",
    )
    im = axes[3].imshow(np.where(mask, 2 * dt * mm_per_px, np.nan)[win], cmap="viridis")
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.03, label="Local width (mm)")
    titles = ["Photo (fixed window)", "YOLO mask", "Skeleton, longest path", "Width map"]
    for ax, t in zip(axes, titles, strict=True):
        ax.set_title(t, loc="left", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
    viz.panel_letters(axes)
    return fig


def fig11_calibration(df: pd.DataFrame, paper_mm_per_px: float) -> Figure:
    """Image length vs caliper length (out of fold), and a Bland-Altman plot."""
    x, y = df["length_mm"], df["img_length_mm"]
    fig, (ax, bx) = viz.new_figure("double", 3.0, 1, 2)
    lo, hi = min(x.min(), y.min()) - 0.5, max(x.max(), y.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], color="#999999", lw=0.8, ls="--")
    ax.scatter(x, y, s=5, color=viz.COLORS["ink"], alpha=0.35, lw=0)
    r = np.corrcoef(x, y)[0, 1]
    ax.text(
        0.04, 0.96, f"r = {r:.2f}\nn = {len(df):,}", transform=ax.transAxes, va="top", fontsize=7.5
    )
    ax.set_xlabel("Caliper length (mm)")
    ax.set_ylabel("Image length (mm)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_title("Image length tracks caliper length", loc="left")
    mean = (x + y) / 2
    diff = y - x
    bias, sd = diff.mean(), diff.std()
    bx.scatter(mean, diff, s=5, color=viz.COLORS["ink"], alpha=0.35, lw=0)
    for v, lab in [
        (bias, f"bias {bias:+.2f} mm"),
        (bias + 1.96 * sd, f"+1.96 SD {bias + 1.96 * sd:+.2f}"),
        (bias - 1.96 * sd, f"-1.96 SD {bias - 1.96 * sd:+.2f}"),
    ]:
        bx.axhline(
            v,
            color=viz.COLORS["female"] if v == bias else "#999999",
            lw=0.9,
            ls="-" if v == bias else "--",
        )
        bx.text(mean.max(), v, " " + lab, va="center", fontsize=7)
    bx.set_xlim(right=mean.max() + (mean.max() - mean.min()) * 0.35)
    bx.set_xlabel("Mean of the two lengths (mm)")
    bx.set_ylabel("Image minus caliper (mm)")
    bx.set_title("Bland-Altman agreement", loc="left")
    fitted = df["mm_per_px"].median()
    fig.text(
        0.01,
        -0.02,
        f"mm per pixel fitted inside each fold on training rows: median "
        f"{fitted:.4f} ({1 / fitted:.2f} px per mm on the 2048 px copies). The paper's 28 px "
        f"per mm on full-size photos equals {1 / paper_mm_per_px:.2f} px per mm here.",
        fontsize=7,
        color="#555555",
    )
    viz.panel_letters([ax, bx])
    return fig


def fig12_allometry(labels: pd.DataFrame) -> Figure:
    """Log weight against log caliper length with the fitted power law."""
    d = labels.dropna(subset=["weight_g", "length_mm"])
    lx, ly = np.log(d["length_mm"]), np.log(d["weight_g"])
    b, a = np.polyfit(lx, ly, 1)
    boot = np.random.default_rng(42).integers(0, len(d), (1000, len(d)))
    bs = np.array([np.polyfit(lx.to_numpy()[i], ly.to_numpy()[i], 1)[0] for i in boot])
    fig, ax = viz.new_figure("single", 3.0)
    ax.scatter(d["length_mm"], d["weight_g"], s=5, color="#999999", alpha=0.5, lw=0)
    grid = np.linspace(d["length_mm"].min(), d["length_mm"].max(), 100)
    ax.plot(grid, np.exp(a) * grid**b, color=viz.COLORS["female"], lw=1.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Caliper length (mm, log scale)")
    ax.set_ylabel("Weight (g, log scale)")
    lo, hi = np.percentile(bs, [2.5, 97.5])
    ax.text(
        0.04,
        0.96,
        f"W = {np.exp(a):.2e} L^{b:.2f}\nb 95% CI {lo:.2f} to {hi:.2f}",
        transform=ax.transAxes,
        va="top",
        fontsize=7.5,
        color=viz.COLORS["female"],
    )
    ax.set_title(f"Weight grows as length to the power {b:.1f}", loc="left")
    return fig


def fig13_pred_vs_measured(preds: pd.DataFrame, models: list[str]) -> Figure:
    """Out-of-fold predicted vs measured weight, one hexbin panel per model."""
    n = len(models)
    fig, axes = viz.new_figure("double", 7.2 / n + 0.7, 1, n, sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    lo, hi = preds[["weight_g", "pred_g"]].min().min(), preds[["weight_g", "pred_g"]].max().max()
    for ax, m in zip(axes, models, strict=True):
        d = preds[preds["model"] == m]
        ax.hexbin(
            d["weight_g"],
            d["pred_g"],
            gridsize=28,
            cmap="viridis",
            mincnt=1,
            extent=(lo, hi, lo, hi),
            linewidths=0,
        )
        ax.plot([lo, hi], [lo, hi], color="#999999", lw=0.8, ls="--")
        ss = ((d["weight_g"] - d["pred_g"]) ** 2).sum()
        r2 = 1 - ss / ((d["weight_g"] - d["weight_g"].mean()) ** 2).sum()
        mae = (d["weight_g"] - d["pred_g"]).abs().mean() * 1000
        ax.text(
            0.04,
            0.96,
            f"R² {r2:.2f}\nMAE {mae:.0f} mg",
            transform=ax.transAxes,
            va="top",
            fontsize=7.5,
        )
        ax.set_title(m, loc="left", fontsize=8)
        ax.set_aspect("equal")
        ax.set_xlabel("Measured weight (g)")
    axes[0].set_ylabel("Predicted weight (g)")
    viz.panel_letters(axes)
    return fig


def fig14_model_comparison(summary: pd.DataFrame) -> Figure:
    """Cross-validated R² and MAE per model, with the random split and the paper's values."""
    s = summary.sort_values("cv_r2_mean")
    fig, (ax, bx) = viz.new_figure("double", 0.5 + 0.38 * len(s), 1, 2, sharey=True)
    y = np.arange(len(s))
    ax.errorbar(
        s["cv_r2_mean"],
        y,
        xerr=s["cv_r2_std"],
        fmt="o",
        color=viz.COLORS["ink"],
        capsize=2,
        ms=5,
        label="batch-grouped CV",
    )
    ax.scatter(s["random_r2"], y, facecolors="none", edgecolors=viz.COLORS["ink"], s=28, zorder=3)
    for name, val in PAPER.items():
        ax.axvline(val, color=viz.COLORS["29C"], lw=1, ls="--")
        ax.text(
            val,
            len(s) - 0.4,
            name,
            rotation=90,
            fontsize=6.5,
            color=viz.COLORS["29C"],
            ha="right",
            va="top",
        )
    ax.set_yticks(y, s["model"])
    ax.set_xlabel("R² (higher is better)")
    ax.set_title("Filled: batch folds (mean ± SD). Hollow: random 80/20", loc="left", fontsize=8)
    bx.errorbar(
        s["cv_mae_mg_mean"],
        y,
        xerr=s["cv_mae_mg_std"],
        fmt="o",
        color=viz.COLORS["ink"],
        capsize=2,
        ms=5,
    )
    bx.scatter(
        s["random_mae_mg"], y, facecolors="none", edgecolors=viz.COLORS["ink"], s=28, zorder=3
    )
    bx.set_xlabel("Mean absolute error (mg, lower is better)")
    for a in (ax, bx):
        a.grid(axis="x", color=viz.COLORS["grid"], lw=0.5)
        a.set_axisbelow(True)
    viz.panel_letters([ax, bx])
    return fig
