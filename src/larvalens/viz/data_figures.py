"""Figures 3, 4 and 5: data audit, batch timeline and weight ridgeline."""

from __future__ import annotations

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from scipy.stats import gaussian_kde

from larvalens import viz

NO_BATCH_LABEL = "No batch"


def batch_order(labels: pd.DataFrame) -> list[str]:
    """Named batches in order, then the unbatched group last."""
    named = sorted(labels["batch"].dropna().unique())
    return named + ([NO_BATCH_LABEL] if labels["batch"].isna().any() else [])


def fig03_data_audit(
    labels: pd.DataFrame, sex_crops: pd.DataFrame, photo_ids: set[int] | None = None
) -> Figure:
    """Missing-value matrix with rows sorted by batch, plus how many rows have images.

    Args:
        labels: cleaned labels (one row per larva).
        sex_crops: the sex crop manifest (file, larva_id, usable).
        photo_ids: larva IDs that have a single-larva photo, or None if not yet known.
    """
    df = labels.copy()
    df["batch_label"] = df["batch"].fillna(NO_BATCH_LABEL)
    order = batch_order(labels)
    df["batch_rank"] = df["batch_label"].map({b: i for i, b in enumerate(order)})
    df = df.sort_values(["batch_rank", "larva_id"]).reset_index(drop=True)
    crop_ids = set(sex_crops.loc[sex_crops["readable"], "larva_id"].dropna().astype(int))
    cols = {
        "Weight": df["weight_g"].notna(),
        "Length": df["length_mm"].notna(),
        "Width": df["width_mm"].notna(),
        "Sex": df["sex"].notna(),
        "Batch": df["batch"].notna(),
        "Hatch date": df["age_days"].notna(),
        "Temperature": df["temp_c"].notna(),
        "Sex crop": df["larva_id"].isin(crop_ids),
    }
    if photo_ids is not None:
        cols["Photo"] = df["larva_id"].isin(photo_ids)
    matrix = np.column_stack([v.to_numpy() for v in cols.values()]).astype(float)

    fig, (ax, bx) = viz.new_figure("double", 4.2, 1, 2, gridspec_kw={"width_ratios": [1.6, 1]})
    cmap = ListedColormap(["#F2F2F2", "#3B528B"])
    ax.imshow(matrix, aspect="auto", cmap=cmap, interpolation="nearest", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)), list(cols), rotation=45, ha="right")
    starts = df.groupby("batch_label", sort=False).head(1).index
    ax.set_yticks(
        [(s + e) / 2 for s, e in zip(starts, list(starts[1:]) + [len(df)], strict=True)],
        df.loc[starts, "batch_label"],
    )
    for s in starts[1:]:
        ax.axhline(s - 0.5, color="white", lw=1.2)
    ax.tick_params(length=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title("Most gaps sit in the unbatched rows", loc="left")
    ax.set_xlabel("dark = present, light = missing", fontsize=7, color="#555555")

    n = len(df)
    bars = {"Spreadsheet rows": n}
    if photo_ids is not None:
        bars["with a photo"] = int(df["larva_id"].isin(photo_ids).sum())
    bars["with a readable sex crop"] = int(df["larva_id"].isin(crop_ids).sum())
    usable = sex_crops.loc[sex_crops["usable"], "larva_id"].nunique()
    bars["usable for the sex model"] = int(usable)
    names = list(bars)[::-1]
    vals = [bars[k] for k in names]
    bx.barh(
        names,
        vals,
        color=["#3B528B" if k == "Spreadsheet rows" else "#21918C" for k in names],
        height=0.6,
    )
    for y, v in enumerate(vals):
        bx.text(v + n * 0.01, y, f"{v:,}", va="center", fontsize=8)
    bx.set_xlim(0, n * 1.18)
    bx.set_xlabel("Larvae")
    bx.spines["left"].set_visible(False)
    bx.tick_params(axis="y", length=0)
    bx.set_title("Images per spreadsheet row", loc="left")
    viz.panel_letters([ax, bx])
    return fig


def fig04_batch_timeline(labels: pd.DataFrame, batches: pd.DataFrame) -> Figure:
    """Hatch date to measurement days for each batch, coloured by rearing temperature."""
    order = batch_order(labels)
    info = batches.set_index("batch")
    fig, ax = viz.new_figure("double", 3.4)
    for y, b in enumerate(order):
        rows = (
            labels[labels["batch"].isna()] if b == NO_BATCH_LABEL else labels[labels["batch"] == b]
        )
        temp = info["temp_c"].get(b) if b in info.index else None
        color = viz.temp_color(temp)
        days = rows.groupby("date").size()
        hatch = info["hatch_date"].get(b) if b in info.index else None
        if hatch is not None and pd.notna(hatch):
            ax.plot(
                [hatch, days.index.max()],
                [y, y],
                color=color,
                lw=2,
                solid_capstyle="round",
                alpha=0.55,
            )
            ax.plot(hatch, y, marker="o", mfc="white", mec=color, mew=1.3, ms=5)
        ax.scatter(
            days.index,
            [y] * len(days),
            s=days.to_numpy() * 0.9,
            color=color,
            zorder=3,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.text(
            days.index.max() + pd.Timedelta(days=2),
            y,
            f"n = {len(rows)}",
            va="center",
            fontsize=7,
            color="#555555",
        )
    ax.set_yticks(range(len(order)), order)
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(axis="x", color=viz.COLORS["grid"], lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    xmax = labels["date"].max() + pd.Timedelta(days=14)
    ax.set_xlim(right=xmax)
    for i, (txt, col) in enumerate(
        [("29 °C", viz.COLORS["29C"]), ("25 °C", viz.COLORS["25C"]), ("unknown", "#999999")]
    ):
        ax.text(
            0.99,
            0.97 - i * 0.07,
            txt,
            color=col,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontweight="bold",
        )
    ax.text(
        0.99,
        0.75,
        "open circle = hatch date\ndot size = larvae measured that day",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color="#555555",
    )
    ax.set_title("Larvae were measured 10 to 20 days after hatching, July to November 2023")
    return fig


def fig05_weight_ridgeline(labels: pd.DataFrame, batches: pd.DataFrame) -> Figure:
    """Weight distribution per batch, coloured by rearing temperature."""
    info = batches.set_index("batch")["temp_c"]
    df = labels.assign(batch_label=labels["batch"].fillna(NO_BATCH_LABEL))
    df["temp_c"] = df["batch_label"].map(info)
    key = {29.0: 0, 25.0: 1}
    order = sorted(
        df["batch_label"].unique(),
        key=lambda b: (key.get(info.get(b, np.nan), 2), b == NO_BATCH_LABEL, b),
    )
    grid = np.linspace(0.03, 0.38, 300)
    fig, ax = viz.new_figure("single", 4.6)
    step = 1.0
    for i, b in enumerate(order):
        w = df.loc[df["batch_label"] == b, "weight_g"].to_numpy()
        dens = gaussian_kde(w, bw_method=0.3)(grid)
        dens = dens / dens.max() * 1.35
        y0 = -i * step
        color = viz.temp_color(info.get(b, np.nan))
        ax.fill_between(grid, y0, y0 + dens, color=color, alpha=0.75, lw=0, zorder=len(order) - i)
        ax.plot(grid, y0 + dens, color="white", lw=0.8, zorder=len(order) - i)
        med = np.median(w)
        ax.plot(
            [med, med],
            [y0, y0 + np.interp(med, grid, dens)],
            color="#222222",
            lw=0.8,
            zorder=len(order) - i,
        )
        ax.text(
            -0.02,
            y0,
            b,
            fontsize=7.5,
            va="bottom",
            ha="right",
            zorder=50,
            transform=ax.get_yaxis_transform(),
        )
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0.03, 0.38)
    ax.set_xlabel("Weight (g)")
    for j, (txt, col) in enumerate(
        [("29 °C", viz.COLORS["29C"]), ("25 °C", viz.COLORS["25C"]), ("unknown", "#999999")]
    ):
        ax.text(
            0.98,
            0.97 - j * 0.045,
            txt,
            color=col,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontweight="bold",
        )
    ax.set_title("Weight varies more between batches\nthan between temperatures")
    return fig
