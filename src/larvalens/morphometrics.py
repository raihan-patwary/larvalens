"""Shape measurements from a binary larva mask.

Length follows the skeleton's longest path (found with skan) and adds the
distance-transform value at both ends, because thinning stops about half a
body width short of each tip. Width is twice the distance transform: its
maximum, and its mean along the middle half of the skeleton path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from scipy import ndimage
from scipy.sparse.csgraph import dijkstra
from skan import Skeleton
from skimage.color import rgb2lab
from skimage.morphology import skeletonize


@dataclass
class Measurements:
    """Morphometrics of one mask, in pixels (area in px²)."""

    area_px: float
    perimeter_px: float
    length_px: float
    width_max_px: float
    width_mid_px: float
    lab_l: float = float("nan")
    lab_a: float = float("nan")
    lab_b: float = float("nan")

    def as_dict(self) -> dict[str, float]:
        """Return the measurements as a plain dict."""
        return asdict(self)


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected region of a binary mask, with holes filled."""
    labels, n = ndimage.label(mask > 0)
    if n == 0:
        return np.zeros(mask.shape, dtype=bool)
    sizes = ndimage.sum(np.ones_like(labels), labels, index=range(1, n + 1))
    keep = labels == (int(np.argmax(sizes)) + 1)
    return ndimage.binary_fill_holes(keep)


def perimeter(mask: np.ndarray) -> float:
    """Length of the outer contour in pixels."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return float(max(cv2.arcLength(c, closed=True) for c in contours)) if contours else 0.0


def longest_skeleton_path(skeleton: np.ndarray) -> np.ndarray:
    """Return the (row, col) pixel coordinates of the skeleton's longest path, in order."""
    if skeleton.sum() < 2:
        return np.argwhere(skeleton).astype(float)
    skel = Skeleton(skeleton)
    graph = skel.graph
    coords = skel.coordinates
    start = 0
    for _ in range(2):  # two sweeps find the diameter of a tree
        dist, pred = dijkstra(graph, indices=start, return_predecessors=True)
        dist[~np.isfinite(dist)] = -1
        end = int(np.argmax(dist))
        if _ == 0:
            start = end
    path = [end]
    while pred[path[-1]] >= 0:
        path.append(int(pred[path[-1]]))
    return coords[path[::-1]]


def path_length(path: np.ndarray) -> float:
    """Sum of the step lengths along an ordered path."""
    if len(path) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def measure(mask: np.ndarray, image_rgb: np.ndarray | None = None) -> Measurements:
    """Measure one larva mask.

    Args:
        mask: 2D array, nonzero inside the larva.
        image_rgb: optional RGB image of the same size, for mean Lab colour.

    Returns:
        Measurements in pixels.
    """
    mask = largest_component(mask)
    if not mask.any():
        return Measurements(0.0, 0.0, 0.0, 0.0, 0.0)
    dt = ndimage.distance_transform_edt(mask)
    path = longest_skeleton_path(skeletonize(mask))
    ends = path[[0, -1]].round().astype(int)
    tips = float(dt[ends[0][0], ends[0][1]] + dt[ends[1][0], ends[1][1]])
    length = path_length(path) + tips
    idx = path.round().astype(int)
    radius = dt[idx[:, 0], idx[:, 1]]
    q = len(radius) // 4
    middle = radius[q : len(radius) - q] if len(radius) >= 4 else radius
    out = Measurements(
        area_px=float(mask.sum()),
        perimeter_px=perimeter(mask),
        length_px=length,
        width_max_px=float(2 * dt.max()),
        width_mid_px=float(2 * middle.mean()),
    )
    if image_rgb is not None:
        lab = rgb2lab(image_rgb[..., :3])[mask]
        out.lab_l, out.lab_a, out.lab_b = (float(v) for v in lab.mean(axis=0))
    return out
