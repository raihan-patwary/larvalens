"""Figure helpers that apply the LarvaLens figure rules.

Rules: Okabe-Ito colours with fixed meanings, two widths (3.5 in and 7.2 in),
no top or right spines, direct labels over legends, panel letters, and every
figure exported as SVG plus a 300 dpi PNG.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from larvalens.paths import FIGURES

STYLE = Path(__file__).with_name("bsf.mplstyle")

COLORS: dict[str, str] = {
    "female": "#D55E00",
    "male": "#0072B2",
    "25C": "#56B4E9",
    "29C": "#E69F00",
    "unknown": "#999999",
    "green": "#009E73",
    "purple": "#CC79A7",
    "ink": "#222222",
    "grid": "#E6E6E6",
}

WIDTHS: dict[str, float] = {"single": 3.5, "double": 7.2}

CREDIT = (
    "Data: Geissmann (2024), Zenodo 10.5281/zenodo.13359376, CC BY 4.0; "
    "Nawoya et al. (2025), Smart Agric. Technol. 11, 100953."
)


def use_style() -> None:
    """Apply the house style globally."""
    mpl.style.use(STYLE)


def temp_color(temp: float | None) -> str:
    """Return the fixed colour for a rearing temperature in °C (grey if unknown)."""
    if temp is None or temp != temp:  # None or NaN
        return COLORS["unknown"]
    return COLORS["25C"] if round(temp) == 25 else COLORS["29C"]


def new_figure(
    width: str = "single", height: float = 2.6, nrows: int = 1, ncols: int = 1, **kwargs
) -> tuple[Figure, Axes | list[Axes]]:
    """Create a figure at one of the two allowed widths, in the house style."""
    if width not in WIDTHS:
        raise ValueError(f"width must be one of {sorted(WIDTHS)}, got {width!r}")
    use_style()
    return plt.subplots(nrows, ncols, figsize=(WIDTHS[width], height), **kwargs)


def panel_letters(axes: Iterable[Axes], start: str = "a") -> None:
    """Write bold panel letters (a, b, c...) at the top left of each axis."""
    for i, ax in enumerate(axes):
        ax.text(
            -0.02,
            1.02,
            chr(ord(start) + i),
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="bottom",
            ha="right",
        )


def label_line(ax: Axes, x: float, y: float, text: str, color: str, **kwargs) -> None:
    """Label a line or group directly instead of using a legend."""
    ax.annotate(
        text,
        (x, y),
        xytext=(4, 0),
        textcoords="offset points",
        color=color,
        va="center",
        fontweight="bold",
        **kwargs,
    )


def save(fig: Figure, name: str, outdir: Path | None = None) -> list[Path]:
    """Save a figure as SVG and 300 dpi PNG; return both paths."""
    outdir = outdir or FIGURES
    outdir.mkdir(parents=True, exist_ok=True)
    paths = [outdir / f"{name}.svg", outdir / f"{name}.png"]
    for path in paths:
        fig.savefig(path)
    plt.close(fig)
    return paths
