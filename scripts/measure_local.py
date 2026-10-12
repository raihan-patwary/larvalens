"""Run the 03_measure step on a local CPU (same logic as kernels/03_measure).

Inputs: outputs/01_photos (qc.csv and photos/) and outputs/02_detect (weights and
epochs.csv). Output: outputs/03_measure (measurements.csv, crops/, masks/).
Needs torch and ultralytics==8.4.175, which are not part of the core install.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

from larvalens.morphometrics import largest_component, measure  # noqa: E402

PHOTOS = ROOT / "outputs" / "01_photos"
DETECT = ROOT / "outputs" / "02_detect"
OUT = ROOT / "outputs" / "03_measure"
CROP = 896  # fixed window in photo-copy pixels (about 30 mm at the fitted ~30 px per mm)
IMGSZ = 640
MIN_AREA, MAX_AREA = 10_000, 300_000


def otsu_mask(rgb: np.ndarray) -> np.ndarray:
    """Dark-object threshold; keep the larva-sized region closest to the photo centre."""
    gray = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    _, m = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(m)
    h, w = m.shape
    best, best_d = 0, np.inf
    for i in range(1, n):
        if MIN_AREA <= stats[i, cv2.CC_STAT_AREA] <= MAX_AREA:
            d = np.hypot(cents[i][0] - w / 2, cents[i][1] - h / 2)
            if d < best_d:
                best, best_d = i, d
    return lab == best if best else np.zeros_like(m, bool)


def fixed_crop(img: np.ndarray, cy: float, cx: float, fill) -> np.ndarray:
    """Cut a CROP x CROP window centred on (cy, cx), padding with `fill` at the borders."""
    h, w = img.shape[:2]
    half = CROP // 2
    y0, x0 = int(round(cy)) - half, int(round(cx)) - half
    out = np.empty((CROP, CROP, *img.shape[2:]), img.dtype)
    out[...] = fill
    sy0, sx0 = max(0, y0), max(0, x0)
    sy1, sx1 = min(h, y0 + CROP), min(w, x0 + CROP)
    out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return out


def main(limit: int | None = None) -> None:
    t0 = time.time()
    ep = pd.read_csv(DETECT / "epochs.csv")
    best = ep.groupby("model")["metrics/mAP50-95(M)"].max().idxmax()
    model = YOLO(str(DETECT / f"{best}_best.pt"))
    print("model", best, flush=True)
    qc = pd.read_csv(PHOTOS / "qc.csv")
    photo_dir = next(p.parent for p in PHOTOS.rglob("*.jpg"))
    (OUT / "crops").mkdir(parents=True, exist_ok=True)
    (OUT / "masks").mkdir(parents=True, exist_ok=True)
    files = qc.dropna(subset=["copy"])["copy"].tolist()[:limit]
    rows = []
    for i, name in enumerate(files):
        bgr = cv2.imread(str(photo_dir / name))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = model.predict(bgr, imgsz=IMGSZ, conf=0.25, retina_masks=True, verbose=False)[0]
        row = {"copy": name, "model": best}
        if res.masks is not None and len(res.masks):
            conf = res.boxes.conf.cpu().numpy()
            areas = res.masks.data.cpu().numpy().reshape(len(conf), -1).sum(axis=1)
            # Label digits are also detected; keep the most confident larva-sized mask.
            k = int(np.argmax(np.where(areas >= MIN_AREA, conf, -1)))
            mask = res.masks.data[k].cpu().numpy() > 0.5
            if mask.shape != rgb.shape[:2]:
                mask = cv2.resize(mask.astype(np.uint8), rgb.shape[1::-1], cv2.INTER_NEAREST) > 0
            row.update(method="yolo", conf=float(conf[k]), n_detections=int(len(conf)))
        else:
            mask = otsu_mask(rgb)
            row.update(method="otsu", conf=np.nan, n_detections=0)
        mask = largest_component(mask)
        area = int(mask.sum())
        row["mask_area_ok"] = bool(MIN_AREA <= area <= MAX_AREA)
        if not row["mask_area_ok"]:
            row["area_px"] = area
            rows.append(row)
            continue
        row.update(measure(mask, rgb).as_dict())
        ys, xs = np.nonzero(mask)
        cy, cx = ys.mean(), xs.mean()
        bg = np.median(rgb[~mask], axis=0).astype(np.uint8)
        bh, bw = int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1)
        row.update(
            centroid_y=cy,
            centroid_x=cx,
            bbox_h=bh,
            bbox_w=bw,
            fits_crop=max(bh, bw) <= CROP,
            touches_border=bool(
                ys.min() == 0
                or xs.min() == 0
                or ys.max() == rgb.shape[0] - 1
                or xs.max() == rgb.shape[1] - 1
            ),
        )
        stem = Path(name).stem
        crop = fixed_crop(rgb, cy, cx, bg)
        cv2.imwrite(
            str(OUT / "crops" / f"{stem}.jpg"),
            cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )
        cv2.imwrite(
            str(OUT / "masks" / f"{stem}.png"), fixed_crop(mask.astype(np.uint8) * 255, cy, cx, 0)
        )
        rows.append(row)
        if i % 100 == 0:
            print(f"{i}/{len(files)} {(time.time() - t0) / 60:.1f} min", flush=True)
    cols = ["copy", "file", "larva_id", "scale", "width", "height"]
    df = pd.DataFrame(rows).merge(qc[cols], on="copy", how="left")
    df.to_csv(OUT / "measurements.csv", index=False)
    print(df["method"].value_counts().to_dict(), f"done {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
