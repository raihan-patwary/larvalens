"""End-to-end steps behind the CLI commands."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd

from larvalens import audit, checks, clean, download, splits
from larvalens.paths import DATA, OUTPUTS, RAW, RESULTS, SPLITS, load_config

EXTRACTED = DATA / "extracted"
DETECT_DIR = EXTRACTED / "detect" / "BSF_Larvae_v1"
SEX_DIR = EXTRACTED / "sex" / "Larva_sex_classification"


def extract(zip_path: Path, dest: Path) -> Path:
    """Unzip once; skip if the destination already exists."""
    if not dest.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest)
    return dest


def photo_checks(labels: pd.DataFrame) -> list[checks.Check]:
    """Check the single-larva photos, using the Kaggle 01_photos output if present."""
    qc_path = OUTPUTS / "01_photos" / "qc.csv"
    if not qc_path.exists():
        return [checks.Check("Single-larva photos", "1,608", "not yet run", None)]
    qc = pd.read_csv(qc_path)
    ids = qc["larva_id"].dropna().astype(int)
    matched = int(ids.isin(labels["larva_id"]).sum())
    return [
        checks.Check(
            "Single-larva photos",
            "1,608",
            f"{len(qc):,} photos; {matched:,} match a spreadsheet row",
            len(qc) == 1608,
        )
    ]


def build_data() -> None:
    """Download, verify, clean, audit and split; write the reports."""
    cfg = load_config()
    download.fetch_all(RAW)
    batches, labels = clean.run()
    grouped = clean.load_sheets(RAW / clean.XLSX)["grouped"]
    extract(RAW / "BSF_Larvae_v1.zip", EXTRACTED / "detect")
    extract(RAW / "Larva_sex_classification.zip", EXTRACTED / "sex")

    det = audit.detect_stats(DETECT_DIR)
    manifest = audit.sex_crop_manifest(SEX_DIR, labels)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "detect_stats.json").write_text(json.dumps(det, indent=2), encoding="utf-8")

    result = checks.spreadsheet_checks(labels, batches, grouped)
    result.insert(
        0,
        checks.Check(
            "Detection set",
            "321 images, over 5,000 larva instances",
            f"{det['images']} images, {det['instances']:,} instances",
            det["images"] == 321 and det["instances"] > 5000,
            f"Only {det['source_photos']} distinct source photos (see findings).",
        ),
    )
    result[1:1] = photo_checks(labels)
    usable = manifest[manifest["usable"]]
    sex_counts = usable["sex"].value_counts()
    ages = labels["age_days"].dropna()
    findings = [
        f"Detection set: {det['images']} images come from only {det['source_photos']} source "
        "photos. It is a Roboflow export with up to 5 augmented copies per photo, so the image "
        "split is grouped by source photo to stop copies leaking between train and test.",
        f"Detection labels use two class IDs ({det['instances_by_class']}), but they mark the "
        f"same thing: in {det['sources_with_both_classes']} source photos, copies of one photo "
        "use class 0 in one copy and class 1 in another. Both are merged into one class, larva.",
        "Dense detection images are not exhaustively labelled: larvae at the edges and in "
        "clumps are often unlabelled, so predicted counts above the labelled count are expected.",
        f"Sex crops: {len(manifest):,} files, not sorted into male and female folders. Each is "
        "named by larva ID, so sex, batch and weight come from the spreadsheet and the sex model "
        "can use the batch folds.",
        f"{int((~manifest['readable']).sum())} sex crop files are not images (they hold browser "
        "log text) and are excluded.",
        "Larva IDs 152 and 172 each have two different crops (0152.jpg and 152.jpg; "
        "0172.jpg and 00172.jpg); both IDs are excluded from the sex model.",
        f"Usable sex crops: {len(usable):,} ({sex_counts.get('F', 0):,} female, "
        f"{sex_counts.get('M', 0):,} male), {int(usable['batch'].isna().sum())} of them with no "
        "batch.",
        f"Age at measurement ranges from {int(ages.min())} to {int(ages.max())} days after "
        "hatching for the batches with a hatch date. The paper says 11 to 14 days.",
        "The 114 unbatched larvae measured on 2023-11-18 share that date with batch B013, and "
        "the other 105 were measured on 2023-08-27, just after B011. This supports keeping "
        "unbatched larvae out of cross-validation.",
    ]
    checks.write_report(result, findings)

    SPLITS.mkdir(parents=True, exist_ok=True)
    splits.make_larvae_splits(
        labels, n_folds=cfg["n_folds"], test_fraction=cfg["random_test_fraction"], seed=cfg["seed"]
    ).to_csv(SPLITS / "larvae.csv", index=False)
    images = sorted(p.name for p in (DETECT_DIR / "Images").glob("*.jpg"))
    splits.make_detect_split(images, seed=cfg["seed"]).to_csv(SPLITS / "detect.csv", index=False)
    manifest.drop(columns=["batch"]).to_csv(SPLITS / "sex_crops.csv", index=False)
    print("data ready: data/processed, results/data_checks.md, splits/")


def build_figures() -> list[Path]:
    """Rebuild every figure whose inputs exist."""
    from larvalens import viz
    from larvalens.paths import PROCESSED
    from larvalens.viz import data_figures

    labels = pd.read_csv(PROCESSED / "labels.csv", parse_dates=["date", "hatch_date"])
    batches = pd.read_csv(PROCESSED / "batches.csv", parse_dates=["hatch_date"])
    sex_crops = pd.read_csv(SPLITS / "sex_crops.csv")
    qc_path = OUTPUTS / "01_photos" / "qc.csv"
    photo_ids = (
        set(pd.read_csv(qc_path)["larva_id"].dropna().astype(int)) if qc_path.exists() else None
    )
    made = []
    made += viz.save(
        data_figures.fig03_data_audit(labels, sex_crops, photo_ids), "fig03_data_audit"
    )
    made += viz.save(data_figures.fig04_batch_timeline(labels, batches), "fig04_batch_timeline")
    made += viz.save(data_figures.fig05_weight_ridgeline(labels, batches), "fig05_weight_ridgeline")
    return made
