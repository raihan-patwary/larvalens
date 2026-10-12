"""Tests for splits.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

from larvalens import splits


def synthetic_labels(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    temps = {f"B{i:03d}": t for i, t in enumerate([29, 29, 25, 29, 25, 25, 29, 25, 29, None], 1)}
    temps |= {"B011": None, "B013": None}
    rows = []
    lid = 1
    for batch, temp in temps.items():
        for _ in range(int(rng.integers(50, 190))):
            rows.append({"larva_id": lid, "batch": batch, "temp_c": temp})
            lid += 1
    for _ in range(200):
        rows.append({"larva_id": lid, "batch": None, "temp_c": None})
        lid += 1
    return pd.DataFrame(rows)


def test_batches_never_split_across_folds() -> None:
    s = splits.make_larvae_splits(synthetic_labels())
    batched = s[s["fold"] != splits.NO_BATCH]
    assert batched.groupby("batch")["fold"].nunique().max() == 1
    assert set(batched["fold"]) == {"0", "1", "2", "3", "4"}


def test_unbatched_larvae_are_left_out() -> None:
    labels = synthetic_labels()
    s = splits.make_larvae_splits(labels)
    assert (s.loc[s["batch"].isna(), "fold"] == splits.NO_BATCH).all()
    assert (s.loc[s["batch"].notna(), "fold"] != splits.NO_BATCH).all()


def test_temperatures_spread_across_folds() -> None:
    s = splits.make_larvae_splits(synthetic_labels())
    batched = s[s["fold"] != splits.NO_BATCH]
    per_fold = batched.groupby(["temp_group", "fold"])["batch"].nunique().unstack(fill_value=0)
    # five 29 °C batches over five folds: one each
    assert (per_fold.loc["29C"] == 1).all()
    # four 25 °C batches: no fold gets two
    assert per_fold.loc["25C"].max() == 1


def test_splits_are_deterministic_and_sized() -> None:
    labels = synthetic_labels()
    a = splits.make_larvae_splits(labels, seed=42)
    b = splits.make_larvae_splits(labels, seed=42)
    pd.testing.assert_frame_equal(a, b)
    assert (a["random_split"] == "test").sum() == round(len(labels) * 0.2)


def test_source_photo_strips_roboflow_suffix() -> None:
    assert splits.source_photo("001_jpg.rf.2ad9471a1bef3fa760fee5f46b4bc92b.jpg") == "001"
    name = "image_1686238089670_png_jpg.rf.34e1e0f74dea93aa6821bbeef1e895a2.jpg"
    assert splits.source_photo(name) == "image_1686238089670_png"


def test_detect_split_groups_copies() -> None:
    names = [f"{i:03d}_jpg.rf.{h}.jpg" for i in range(40) for h in ("aa", "bb", "cc")]
    d = splits.make_detect_split(names)
    assert d.groupby("source")["split"].nunique().max() == 1
    assert set(d["split"]) == {"train", "val", "test"}
