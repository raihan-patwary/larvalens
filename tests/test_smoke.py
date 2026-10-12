"""Smoke tests: the package imports, the CLI runs and the figure helper works."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import larvalens
from larvalens import viz
from larvalens.__main__ import main


def test_version(capsys) -> None:
    main(["version"])
    assert capsys.readouterr().out.strip() == larvalens.__version__


def test_figure_saves_svg_and_png(tmp_path) -> None:
    fig, ax = viz.new_figure("single")
    ax.plot([0, 1], [0, 1], color=viz.COLORS["female"])
    viz.panel_letters([ax])
    paths = viz.save(fig, "smoke", tmp_path)
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)
    assert {p.suffix for p in paths} == {".svg", ".png"}


def test_temp_color() -> None:
    assert viz.temp_color(25) == viz.COLORS["25C"]
    assert viz.temp_color(29.0) == viz.COLORS["29C"]
    assert viz.temp_color(float("nan")) == viz.COLORS["unknown"]
