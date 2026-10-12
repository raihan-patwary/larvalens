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


def photo_findings(labels: pd.DataFrame) -> list[str]:
    """Findings from the 01_photos QC table, if it has been pulled."""
    qc_path = OUTPUTS / "01_photos" / "qc.csv"
    if not qc_path.exists():
        return []
    qc = pd.read_csv(qc_path)
    missing = sorted(set(labels["larva_id"]) - set(qc["larva_id"].dropna().astype(int)))
    sizes = qc.groupby(["width", "height"]).size()
    near = int(qc["dhash"].duplicated(keep=False).sum())
    size_text = " and ".join(f"{w} x {h}" for w, h in sizes.index)
    return [
        f"Single-larva photos: {len(qc):,} JPEGs, all {size_text}"
        f" px, named by larva ID (0001.jpg). Every photo matches one spreadsheet row; "
        f"{len(missing)} rows have no photo (larva IDs {', '.join(map(str, missing))}).",
        f"No two photos are byte-identical (md5). {near} photos share a coarse 64-bit "
        "perceptual hash with another photo, but each pair differs in brightness and file size, "
        "so these look like similar empty backgrounds rather than repeated photos.",
        f"Image quality is even: blur (variance of the Laplacian) {qc['blur'].min():.1f} to "
        f"{qc['blur'].max():.1f}, mean brightness {qc['brightness'].min():.0f} to "
        f"{qc['brightness'].max():.0f} on a 0 to 255 scale. No photo is unreadable.",
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
    findings += photo_findings(labels)
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
    if (RESULTS / "detect_metrics_test.csv").exists():
        made += detect_figures()
    if (RESULTS / "weight_summary_shape.csv").exists():
        made += weight_figures()
    if (RESULTS / "sex_predictions.csv").exists() and (OUTPUTS / "03_measure" / "crops").exists():
        made += hero_banner()
    return made


DETECT_MODELS = ["yolov8n-seg", "yolov8s-seg", "yolov8m-seg"]
PAPER_MASK_AP50 = 0.93


def collect_detect() -> pd.DataFrame:
    """Copy the 02_detect metrics into results/ and add the paper's reference value."""
    out = OUTPUTS / "02_detect"
    metrics = pd.read_csv(out / "metrics_test.csv")
    epochs = pd.read_csv(out / "epochs.csv")
    counts = pd.read_csv(out / "counts_test.csv")
    best_val = epochs.groupby("model")["metrics/mAP50-95(M)"].max().rename("val_mask_map50_95_best")
    metrics = metrics.merge(best_val, on="model")
    for m in DETECT_MODELS:
        err = counts[f"pred_{m}"] - counts["true_count"]
        metrics.loc[metrics["model"] == m, "count_mae"] = err.abs().mean()
        metrics.loc[metrics["model"] == m, "count_median_error"] = err.median()
    metrics["test_images"] = len(counts)
    metrics["test_source_photos"] = counts["source"].nunique()
    metrics["paper_mask_ap50"] = PAPER_MASK_AP50
    RESULTS.mkdir(parents=True, exist_ok=True)
    metrics.round(4).to_csv(RESULTS / "detect_metrics_test.csv", index=False)
    counts.to_csv(RESULTS / "detect_counts_test.csv", index=False)
    keep = [
        "model",
        "epoch",
        "train/seg_loss",
        "val/seg_loss",
        "metrics/mAP50(M)",
        "metrics/mAP50-95(M)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]
    epochs[keep].round(5).to_csv(RESULTS / "detect_epochs.csv", index=False)
    return metrics


def detect_figures() -> list[Path]:
    """Figures 7, 8 and 9 from the saved detection results."""
    import json

    from larvalens import viz
    from larvalens.viz import detect_figures as df_

    counts = pd.read_csv(RESULTS / "detect_counts_test.csv")
    epochs = pd.read_csv(RESULTS / "detect_epochs.csv")
    metrics = pd.read_csv(RESULTS / "detect_metrics_test.csv")
    best = metrics.loc[metrics["val_mask_map50_95_best"].idxmax(), "model"]
    made = []
    pred_path = OUTPUTS / "02_detect" / "predictions_test.json"
    if pred_path.exists() and DETECT_DIR.exists():
        preds = json.loads(pred_path.read_text())
        first = counts.drop_duplicates("source").sort_values("true_count")
        picks = [first.iloc[len(first) // 4], first.iloc[(3 * len(first)) // 4], first.iloc[-1]]
        images = [DETECT_DIR / "Images" / p["image"] for p in picks]
        fig = df_.fig07_detection_overlay(images, DETECT_DIR / "Labels", preds, best)
        made += viz.save(fig, "fig07_detection_overlay")
    made += viz.save(df_.fig08_count_agreement(counts, DETECT_MODELS), "fig08_count_agreement")
    made += viz.save(df_.fig09_training_curves(epochs, DETECT_MODELS), "fig09_training_curves")
    return made


def measurement_table() -> pd.DataFrame:
    """Join the 03_measure output with the cleaned labels and the splits."""
    from larvalens.paths import PROCESSED

    meas = pd.read_csv(OUTPUTS / "03_measure" / "measurements.csv")
    labels = pd.read_csv(PROCESSED / "labels.csv")
    folds = pd.read_csv(SPLITS / "larvae.csv", dtype={"fold": str})
    df = meas.merge(labels, on="larva_id", how="inner").merge(
        folds[["larva_id", "fold", "random_split"]], on="larva_id", how="left"
    )
    return df[df["length_px"] > 0].reset_index(drop=True)


def run_weight_models() -> pd.DataFrame:
    """Calibrate and fit the weight-from-shape models; save metrics and predictions."""
    from larvalens import weights

    df = measurement_table()
    cal = weights.calibration_oof(df)
    keep = [
        "larva_id",
        "copy",
        "fold",
        "eval",
        "length_mm",
        "img_length_mm",
        "mm_per_px",
        "method",
        "conf",
        "touches_border",
        "fits_crop",
    ]
    cal[[c for c in keep if c in cal]].to_csv(RESULTS / "calibration_oof.csv", index=False)
    metrics, preds = [], []
    for name, fit_predict in weights.MODELS.items():
        m, p = weights.evaluate(df, fit_predict, name)
        metrics.append(m)
        preds.append(p)
    metrics = pd.concat(metrics, ignore_index=True)
    preds = pd.concat(preds, ignore_index=True).merge(
        df[["larva_id", "weight_g", "copy"]], on="larva_id"
    )
    metrics.round(4).to_csv(RESULTS / "weight_metrics_shape.csv", index=False)
    preds.round(5).to_csv(RESULTS / "weight_predictions_shape.csv", index=False)
    summary = weights.summarise(metrics)
    summary.round(4).to_csv(RESULTS / "weight_summary_shape.csv", index=False)
    qc = pd.read_csv(OUTPUTS / "01_photos" / "qc.csv")
    qc[["copy", "larva_id", "blur", "brightness", "contrast"]].round(3).to_csv(
        RESULTS / "photo_qc.csv", index=False
    )
    meas_cols = [
        "larva_id",
        "copy",
        "method",
        "conf",
        "n_detections",
        "area_px",
        "perimeter_px",
        "length_px",
        "width_max_px",
        "width_mid_px",
        "lab_l",
        "lab_a",
        "lab_b",
        "touches_border",
        "fits_crop",
    ]
    df[[c for c in meas_cols if c in df]].round(3).to_csv(RESULTS / "measurements.csv", index=False)
    return summary


def weight_figures() -> list[Path]:
    """Figures 2, 6 and 10 to 14 from saved results and the pulled crops."""
    import matplotlib.image as mpimg

    from larvalens import viz
    from larvalens.paths import PROCESSED
    from larvalens.viz import weight_figures as wf

    made = []
    crop_dir = OUTPUTS / "03_measure" / "crops"
    mask_dir = OUTPUTS / "03_measure" / "masks"
    labels = pd.read_csv(PROCESSED / "labels.csv")
    meas = pd.read_csv(RESULTS / "measurements.csv").merge(labels, on="larva_id")
    cal = pd.read_csv(RESULTS / "calibration_oof.csv")
    paper_mm_per_px = 1 / (28.0 * 2048 / 4056)
    if crop_dir.exists():
        made += viz.save(wf.fig02_mosaic(meas, crop_dir), "fig02_mosaic")
        made += viz.save(
            wf.fig06_image_qc(pd.read_csv(RESULTS / "photo_qc.csv"), crop_dir), "fig06_image_qc"
        )
        typical = meas[(meas["method"] == "yolo")].copy()
        typical["gap"] = (typical["weight_g"] - typical["weight_g"].median()).abs()
        stem = Path(typical.nsmallest(1, "gap")["copy"].iloc[0]).stem
        crop = mpimg.imread(crop_dir / f"{stem}.jpg")
        mask = mpimg.imread(mask_dir / f"{stem}.png") > 0.5
        mask = mask if mask.ndim == 2 else mask[..., 0]
        made += viz.save(
            wf.fig10_morphometrics(crop, mask, cal["mm_per_px"].median()), "fig10_morphometrics"
        )
    made += viz.save(
        wf.fig11_calibration(cal[cal["eval"] == "cv"], paper_mm_per_px), "fig11_calibration"
    )
    made += viz.save(wf.fig12_allometry(labels), "fig12_allometry")
    preds = pd.read_csv(RESULTS / "weight_predictions_shape.csv")
    extra = RESULTS / "weight_predictions_pixels.csv"
    if extra.exists():
        preds = pd.concat([preds, pd.read_csv(extra)], ignore_index=True)
    cv = preds[preds["eval"] == "cv"]
    made += viz.save(
        wf.fig13_pred_vs_measured(cv, list(dict.fromkeys(cv["model"]))), "fig13_pred_vs_measured"
    )
    summary = pd.read_csv(RESULTS / "weight_summary_shape.csv")
    extra_s = RESULTS / "weight_summary_pixels.csv"
    if extra_s.exists():
        summary = pd.concat([summary, pd.read_csv(extra_s)], ignore_index=True)
    made += viz.save(wf.fig14_model_comparison(summary), "fig14_model_comparison")
    if extra.exists():
        resnet_cv = cv[cv["model"] == "ResNet-18 (fine-tuned)"]
        made += viz.save(wf.fig15_failures(resnet_cv, crop_dir), "fig15_failures")
        um = pd.read_csv(RESULTS / "umap.csv")
        made += viz.save(
            wf.fig18_umap(um, labels, crop_dir if crop_dir.exists() else None), "fig18_umap"
        )
    return made


def run_pixel_models() -> pd.DataFrame:
    """Score the ResNet-18 predictions from 04_pixels and fit ridge on DINOv2 features."""
    import numpy as np

    from larvalens import weights
    from larvalens.paths import PROCESSED

    out = OUTPUTS / "04_pixels"
    labels = pd.read_csv(PROCESSED / "labels.csv")
    folds = pd.read_csv(SPLITS / "larvae.csv", dtype={"fold": str})
    ids = pd.read_csv(out / "dinov2_ids.csv")
    df = ids.merge(labels[["larva_id", "weight_g", "length_mm"]], on="larva_id").merge(
        folds[["larva_id", "fold", "random_split"]], on="larva_id"
    )
    feats = pd.DataFrame(np.load(out / "dinov2_features.npy"), index=ids["larva_id"])
    m_ridge, p_ridge = weights.evaluate(df, weights.ridge_on_features(feats), "DINOv2 + ridge")
    resnet = pd.read_csv(out / "resnet18_predictions.csv")
    m_resnet = weights.score_predictions(resnet, labels)
    p_resnet = resnet.drop(columns="run")
    metrics = pd.concat([m_resnet, m_ridge], ignore_index=True)
    preds = pd.concat([p_resnet, p_ridge], ignore_index=True).merge(
        df[["larva_id", "weight_g", "copy"]], on="larva_id"
    )
    metrics.round(4).to_csv(RESULTS / "weight_metrics_pixels.csv", index=False)
    preds.round(5).to_csv(RESULTS / "weight_predictions_pixels.csv", index=False)
    summary = weights.summarise(metrics)
    summary.round(4).to_csv(RESULTS / "weight_summary_pixels.csv", index=False)
    pd.read_csv(out / "umap.csv").round(4).to_csv(RESULTS / "umap.csv", index=False)
    pd.read_csv(out / "resnet18_training.csv").round(5).to_csv(
        RESULTS / "resnet18_training.csv", index=False
    )
    return summary


def run_sex() -> pd.DataFrame:
    """Score the 05_sex predictions and draw figures 16 and 17."""
    import io
    import zipfile

    import numpy as np
    from PIL import Image

    from larvalens import sex, viz
    from larvalens.viz import sex_figures

    out = OUTPUTS / "05_sex"
    preds = pd.read_csv(out / "sex_predictions.csv")
    scores = sex.score_runs(preds)
    scores.round(4).to_csv(RESULTS / "sex_scores.csv", index=False)
    preds.round(5).to_csv(RESULTS / "sex_predictions.csv", index=False)
    cv = preds[preds["eval"] == "cv"]
    pooled = scores.set_index("run").loc["pooled out-of-fold"]
    viz.save(sex_figures.fig16_sex_results(cv, pooled), "fig16_sex_results")
    z = np.load(out / "gradcam.npz")
    cams = dict(zip(z["keys"], z["cams"].astype(np.float32), strict=True))
    picks = cv.assign(
        right=(cv["p_male"] >= 0.5) == (cv["sex"] == "M"), key=cv["run"] + "|" + cv["file"]
    )
    # most confident right answers and most confident wrong answers
    picks["conf"] = (picks["p_male"] - 0.5).abs()
    picks = picks.sort_values("conf", ascending=False).groupby(["sex", "right"]).head(8)
    with zipfile.ZipFile(RAW / "Larva_sex_classification.zip") as zf:
        names = {n.rsplit("/", 1)[-1]: n for n in zf.namelist()}
        images = {}
        for f in picks["file"]:
            images[f] = _square(Image.open(io.BytesIO(zf.read(names[f]))))
    viz.save(sex_figures.fig17_gradcam(images, picks, cams), "fig17_gradcam")
    return scores


def _square(im, size: int = 224):
    """Pad to a square with the median border colour and resize (as in 05_sex)."""
    import numpy as np
    from PIL import Image

    a = np.asarray(im.convert("RGB"))
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    fill = tuple(int(v) for v in np.median(border, axis=0))
    h, w = a.shape[:2]
    side = max(h, w)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(Image.fromarray(a), ((side - w) // 2, (side - h) // 2))
    return np.asarray(canvas.resize((size, size), Image.BILINEAR))


def hero_banner() -> list[Path]:
    """Figure 1 from the typical-weight held-out larva used in figure 10."""
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    from larvalens.paths import FIGURES, PROCESSED
    from larvalens.viz.hero import hero

    labels = pd.read_csv(PROCESSED / "labels.csv")
    meas = pd.read_csv(RESULTS / "measurements.csv").merge(labels, on="larva_id")
    crop_dir = OUTPUTS / "03_measure" / "crops"
    t = meas[meas["method"] == "yolo"].copy()
    t["gap"] = (t["weight_g"] - t["weight_g"].median()).abs()
    rec = t.nsmallest(1, "gap").iloc[0]
    stem = Path(rec["copy"]).stem
    crop = mpimg.imread(crop_dir / f"{stem}.jpg")
    mask = mpimg.imread(OUTPUTS / "03_measure" / "masks" / f"{stem}.png")
    mask = (mask if mask.ndim == 2 else mask[..., 0]) > 0.5
    cal = pd.read_csv(RESULTS / "calibration_oof.csv").set_index("larva_id")
    k = cal.loc[rec["larva_id"], "mm_per_px"]
    preds = pd.read_csv(RESULTS / "weight_predictions_shape.csv")
    extra = RESULTS / "weight_predictions_pixels.csv"
    if extra.exists():
        preds = pd.concat([preds, pd.read_csv(extra)], ignore_index=True)
    summary = pd.read_csv(RESULTS / "weight_summary_shape.csv")
    if (RESULTS / "weight_summary_pixels.csv").exists():
        summary = pd.concat([summary, pd.read_csv(RESULTS / "weight_summary_pixels.csv")])
    best = summary.loc[summary["cv_r2_mean"].idxmax(), "model"]
    p = preds[
        (preds["model"] == best)
        & (preds["larva_id"] == rec["larva_id"])
        & (preds["eval"].isin(["cv", "bonus"]))
    ]["pred_g"].iloc[0]
    sexp = pd.read_csv(RESULTS / "sex_predictions.csv")
    s = sexp[(sexp["larva_id"] == rec["larva_id"]) & (sexp["eval"].isin(["cv", "bonus"]))]
    if len(s):
        pm = s["p_male"].iloc[0]
        sex_pred = f"{'male' if pm >= 0.5 else 'female'} ({max(pm, 1 - pm):.0%})"
    else:
        sex_pred = "n/a"
    sex_true = {"F": "female", "M": "male"}.get(rec["sex"], "unknown")
    info = {
        "length_mm": rec["length_px"] * k,
        "width_mm": rec["width_mid_px"] * k,
        "pred_mg": p * 1000,
        "true_mg": rec["weight_g"] * 1000,
        "sex_pred": sex_pred,
        "sex_true": sex_true,
    }
    made = []
    for theme in ("light", "dark"):
        fig = hero(crop, mask, k, info, theme)
        path = FIGURES / f"fig01_hero_{theme}.png"
        fig.savefig(path, dpi=100, facecolor=fig.get_facecolor(), bbox_inches=None)
        plt.close(fig)
        made.append(path)
    return made
