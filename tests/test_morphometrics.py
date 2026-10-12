"""Tests for morphometrics.py on synthetic shapes with known answers."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from larvalens import morphometrics as mm

RADIUS = 15


def capsule(centreline: np.ndarray, shape=(400, 400), radius: int = RADIUS) -> np.ndarray:
    """Stamp discs along a dense centreline to draw a capsule (a sausage shape)."""
    mask = np.zeros(shape, np.uint8)
    for x, y in centreline:
        cv2.circle(mask, (round(x), round(y)), radius, 1, -1)
    return mask.astype(bool)


def straight(length: float = 200.0) -> tuple[np.ndarray, float]:
    xs = np.linspace(100, 100 + length, 2000)
    return capsule(np.column_stack([xs, np.full_like(xs, 200)])), length + 2 * RADIUS


def bent(r: float = 100.0) -> tuple[np.ndarray, float]:
    t = np.linspace(0, np.pi / 2, 2000)
    line = np.column_stack([80 + r * np.cos(t), 80 + r * np.sin(t)])
    return capsule(line), r * np.pi / 2 + 2 * RADIUS


@pytest.mark.parametrize("shape_fn", [straight, bent])
def test_length_matches_centreline_plus_caps(shape_fn) -> None:
    mask, true_length = shape_fn()
    m = mm.measure(mask)
    assert m.length_px == pytest.approx(true_length, rel=0.03)


@pytest.mark.parametrize("shape_fn", [straight, bent])
def test_width(shape_fn) -> None:
    mask, _ = shape_fn()
    m = mm.measure(mask)
    assert m.width_max_px == pytest.approx(2 * RADIUS, abs=2.5)
    assert m.width_mid_px == pytest.approx(2 * RADIUS, abs=2.5)


def test_area_and_perimeter_of_straight_capsule() -> None:
    mask, total = straight()
    body = total - 2 * RADIUS
    r = RADIUS + 0.5  # a drawn disc of radius 15 spans 31 pixels
    m = mm.measure(mask)
    assert m.area_px == pytest.approx(body * 2 * r + np.pi * r**2, rel=0.02)
    assert m.perimeter_px == pytest.approx(2 * body + 2 * np.pi * r, rel=0.05)


def test_keeps_largest_component_and_measures_colour() -> None:
    mask, _ = straight()
    mask[5:9, 5:9] = True  # a speck of noise
    image = np.zeros((*mask.shape, 3), np.uint8)
    image[mask] = (200, 160, 100)
    m = mm.measure(mask, image)
    clean = mm.measure(straight()[0])
    assert m.area_px == clean.area_px
    assert m.lab_l > 50 and m.lab_b > 0  # light and yellowish


def test_empty_mask() -> None:
    m = mm.measure(np.zeros((50, 50), bool))
    assert m.area_px == 0 and m.length_px == 0
