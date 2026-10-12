"""Describe the unzipped detection and sex-crop sets."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

from larvalens.splits import source_photo

ID_RE = re.compile(r"^(\d+)\.jpe?g$", re.IGNORECASE)


def read_yolo_polygons(path: Path) -> list[tuple[int, int]]:
    """Return (class, n_points) for each polygon in a YOLO segmentation label file."""
    out = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if parts:
            out.append((int(parts[0]), (len(parts) - 1) // 2))
    return out


def detect_stats(root: Path) -> dict[str, object]:
    """Summarise the detection set: images, source photos, instances and classes."""
    images = sorted((root / "Images").glob("*.jpg"))
    labels = sorted((root / "Labels").glob("*.txt"))
    per_image = {p.stem: read_yolo_polygons(p) for p in labels}
    classes = Counter(c for polys in per_image.values() for c, _ in polys)
    sizes = Counter(Image.open(p).size for p in images)
    sources = {source_photo(p.name) for p in images}
    # Source photos whose augmented copies disagree on the class ID.
    by_source: dict[str, set[int]] = {}
    for stem, polys in per_image.items():
        by_source.setdefault(source_photo(stem + ".jpg"), set()).update(c for c, _ in polys)
    return {
        "images": len(images),
        "label_files": len(labels),
        "source_photos": len(sources),
        "instances": sum(classes.values()),
        "instances_by_class": dict(sorted(classes.items())),
        "empty_label_files": sum(1 for polys in per_image.values() if not polys),
        "image_sizes": {f"{w}x{h}": n for (w, h), n in sizes.items()},
        "sources_with_both_classes": sum(1 for c in by_source.values() if len(c) > 1),
        "max_instances_per_image": max(len(p) for p in per_image.values()),
    }


def sex_crop_manifest(root: Path, labels: pd.DataFrame) -> pd.DataFrame:
    """List every sex crop with its larva ID, readability, size and label.

    A crop is usable when it opens as an image, its larva ID is unique among the
    crops, and the spreadsheet gives a sex for that larva.
    """
    rows = []
    for path in sorted(root.glob("*")):
        match = ID_RE.match(path.name)
        row = {"file": path.name, "larva_id": int(match.group(1)) if match else None}
        try:
            with Image.open(path) as im:
                im.load()
                row.update(readable=True, width=im.width, height=im.height)
        except (UnidentifiedImageError, OSError):
            row.update(readable=False, width=None, height=None)
        rows.append(row)
    df = pd.DataFrame(rows)
    df["larva_id"] = df["larva_id"].astype("Int64")
    dup = df["larva_id"].duplicated(keep=False) & df["larva_id"].notna()
    df = df.merge(labels[["larva_id", "sex", "batch"]], on="larva_id", how="left")
    reason = pd.Series("", index=df.index)
    reason[df["sex"].isna()] = "no sex label"
    reason[dup] = "two crops share this larva ID"
    reason[~df["readable"]] = "not an image file"
    df["exclude_reason"] = reason
    df["usable"] = reason.eq("")
    return df
