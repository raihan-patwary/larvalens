"""Figure 1: the README hero banner, in light and dark versions (1280 x 640 px)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from scipy import ndimage
from skimage.morphology import skeletonize

from larvalens import viz
from larvalens.morphometrics import largest_component, longest_skeleton_path

THEMES = {
    "light": {"bg": "#FFFFFF", "ink": "#1F2328", "muted": "#59636E", "mask": "#D0D7DE"},
    "dark": {"bg": "#0D1117", "ink": "#E6EDF3", "muted": "#9198A1", "mask": "#30363D"},
}


def hero(crop: np.ndarray, mask: np.ndarray, mm_per_px: float, info: dict, theme: str) -> Figure:
    """Photo, mask, skeleton with width map and the predicted traits, in one strip."""
    t = THEMES[theme]
    viz.use_style()
    plt.rcParams["figure.constrained_layout.use"] = False
    plt.rcParams["savefig.bbox"] = "standard"  # keep the exact 1280 x 640 canvas
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100, facecolor=t["bg"])
    mask = largest_component(mask)
    ys, xs = np.nonzero(mask)
    pad = 40
    win = (
        slice(max(ys.min() - pad, 0), ys.max() + pad),
        slice(max(xs.min() - pad, 0), xs.max() + pad),
    )
    dt = ndimage.distance_transform_edt(mask)
    path = longest_skeleton_path(skeletonize(mask))
    y0, x0 = win[0].start, win[1].start
    fig.text(0.04, 0.88, "LarvaLens", fontsize=34, fontweight="bold", color=t["ink"])
    fig.text(
        0.04,
        0.80,
        "From photos to phenotypes: counting, measuring, weighing and sexing "
        "black soldier fly larvae",
        fontsize=13.5,
        color=t["muted"],
    )
    panels = [0.04, 0.24, 0.44]
    titles = ["Photo", "Outline (YOLOv8m-seg)", "Skeleton and width"]
    for k, (left, title) in enumerate(zip(panels, titles, strict=True)):
        ax = fig.add_axes([left, 0.12, 0.18, 0.58], facecolor=t["bg"])
        if k == 0:
            ax.imshow(crop[win])
        elif k == 1:
            ax.imshow(np.where(mask[win], 1.0, np.nan), cmap="Greys", vmin=0, vmax=1.4)
        else:
            ax.imshow(np.where(mask, 2 * dt * mm_per_px, np.nan)[win], cmap="viridis")
            ax.plot(path[:, 1] - x0, path[:, 0] - y0, color=viz.COLORS["29C"], lw=2.2)
        ax.set_title(title, color=t["ink"], fontsize=12, loc="left")
        ax.set_axis_off()
    lines = [
        ("Length", f"{info['length_mm']:.1f} mm", None),
        ("Width", f"{info['width_mm']:.1f} mm", None),
        ("Weight, predicted", f"{info['pred_mg']:.0f} mg", f"measured {info['true_mg']:.0f} mg"),
        ("Sex, predicted", info["sex_pred"], f"recorded as adult: {info['sex_true']}"),
    ]
    y = 0.64
    for label, value, note in lines:
        fig.text(0.64, y, label, fontsize=12, color=t["muted"])
        fig.text(0.64, y - 0.065, value, fontsize=24, fontweight="bold", color=t["ink"])
        if note:
            fig.text(0.82, y - 0.06, note, fontsize=10.5, color=t["muted"])
        y -= 0.145
    fig.text(
        0.04,
        0.04,
        "Data: Geissmann (2024), Zenodo 10.5281/zenodo.13359376, CC BY 4.0. "
        "Paper: Nawoya et al. (2025).",
        fontsize=9.5,
        color=t["muted"],
    )
    return fig
