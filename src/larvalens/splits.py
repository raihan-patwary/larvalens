"""Build the fixed splits that every model reuses.

- larvae: 5-fold cross-validation grouped by batch and balanced on rearing
  temperature; larvae with no batch are a separate bonus test set; plus one
  random 80/20 split of all larvae for comparison with the paper.
- detect: train, val and test grouped by source photo, because the Roboflow
  export holds several augmented copies of the same photo.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

NO_BATCH = "no_batch"
ROBOFLOW_RE = re.compile(r"_(?:jpg|png|jpeg)\.rf\.[0-9a-f]+$", re.IGNORECASE)


def temp_group(temp: float | None) -> str:
    """Return '25C', '29C' or 'unknown' for a rearing temperature."""
    if temp is None or pd.isna(temp):
        return "unknown"
    return f"{round(temp)}C"


def assign_batch_folds(batches: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> dict[str, int]:
    """Assign whole batches to folds, spreading each temperature group across folds.

    Args:
        batches: one row per batch with columns batch, n (larvae) and temp_group.
        n_folds: number of folds.
        seed: random seed used to break ties.

    Returns:
        Mapping from batch to fold index (0 to n_folds - 1).
    """
    rng = np.random.default_rng(seed)
    order = batches.assign(tiebreak=rng.random(len(batches)))
    group_size = order.groupby("temp_group")["batch"].transform("count")
    order = order.assign(group_size=group_size).sort_values(
        ["group_size", "temp_group", "n", "tiebreak"], ascending=[False, True, False, True]
    )
    rows = np.zeros(n_folds)
    per_group: dict[str, np.ndarray] = {}
    out: dict[str, int] = {}
    for rec in order.itertuples(index=False):
        counts = per_group.setdefault(rec.temp_group, np.zeros(n_folds))
        noise = rng.random(n_folds) * 1e-6
        fold = int(np.lexsort((noise, rows, counts))[0])
        out[rec.batch] = fold
        counts[fold] += 1
        rows[fold] += rec.n
    return out


def make_larvae_splits(
    labels: pd.DataFrame, n_folds: int = 5, test_fraction: float = 0.2, seed: int = 42
) -> pd.DataFrame:
    """Return one row per larva with its CV fold and its random-split role."""
    df = labels[["larva_id", "batch", "temp_c"]].copy()
    df["temp_group"] = df["temp_c"].map(temp_group)
    batched = df[df["batch"].notna()]
    per_batch = (
        batched.groupby("batch")
        .agg(n=("larva_id", "size"), temp_group=("temp_group", "first"))
        .reset_index()
    )
    folds = assign_batch_folds(per_batch, n_folds=n_folds, seed=seed)
    df["fold"] = df["batch"].map(folds).astype("Int64").astype("string").fillna(NO_BATCH)
    rng = np.random.default_rng(seed)
    n_test = round(len(df) * test_fraction)
    test_ids = set(rng.choice(df["larva_id"].to_numpy(), size=n_test, replace=False))
    df["random_split"] = np.where(df["larva_id"].isin(test_ids), "test", "train")
    return df[["larva_id", "batch", "temp_group", "fold", "random_split"]]


def source_photo(filename: str) -> str:
    """Strip the Roboflow suffix so augmented copies share one source name."""
    stem = filename.rsplit(".", 1)[0]
    return ROBOFLOW_RE.sub("", stem)


def make_detect_split(
    filenames: list[str], fractions: tuple[float, float, float] = (0.7, 0.15, 0.15), seed: int = 42
) -> pd.DataFrame:
    """Split detection images into train, val and test, grouped by source photo."""
    df = pd.DataFrame({"image": sorted(filenames)})
    df["source"] = df["image"].map(source_photo)
    sources = np.array(sorted(df["source"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(sources)
    n_train = round(len(sources) * fractions[0])
    n_val = round(len(sources) * fractions[1])
    role = {s: "train" for s in sources[:n_train]}
    role |= {s: "val" for s in sources[n_train : n_train + n_val]}
    role |= {s: "test" for s in sources[n_train + n_val :]}
    df["split"] = df["source"].map(role)
    return df
