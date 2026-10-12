"""LarvaLens 03: segment every single-larva photo and measure it.

Uses the best YOLO segmentation model from 02_detect (chosen on validation mask
mAP50-95) on the photo copies from 01_photos. When YOLO finds no larva, an Otsu
threshold is used instead and the row is flagged. Saves measurements.csv, a
fixed-size crop and mask centred on each larva (same window for every larva,
never resized), and 12 example panels.
"""

import glob
import os
import subprocess
import sys
import time

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.175", "skan==0.13.1"],
               check=True)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from ultralytics import YOLO, settings  # noqa: E402

settings.update({k: False for k in ("raytune", "wandb", "mlflow", "comet", "clearml",
                                    "tensorboard", "dvc", "neptune", "hub") if k in settings})

OUT = "/kaggle/working"
CROP = 512  # fixed window in photo-copy pixels (about 36 mm at the paper's scale)
# A larva covers about 5,000 to 60,000 px on the 2048 px copies. Larger masks are
# background leaks; they are flagged and not measured, which also keeps runtime bounded.
MIN_AREA, MAX_AREA = 2_000, 150_000
IMGSZ = 640


def find(pattern: str) -> list[str]:
    return sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))


def support_path() -> str:
    """Return a folder that contains the larvalens package, unzipping it if needed."""
    pkg = find("larvalens/morphometrics.py")
    if not pkg:
        import zipfile

        for z in find("larvalens.zip"):
            zipfile.ZipFile(z).extractall("/tmp/support/larvalens")
        pkg = glob.glob("/tmp/support/**/larvalens/morphometrics.py", recursive=True)
    if not pkg:
        raise FileNotFoundError("larvalens package not found in /kaggle/input")
    return os.path.dirname(os.path.dirname(sorted(pkg, key=len)[0]))


sys.path.insert(0, support_path())
from larvalens.morphometrics import largest_component, measure  # noqa: E402


def best_model() -> str:
    ep = pd.read_csv(find("epochs.csv")[0])
    best = ep.groupby("model")["metrics/mAP50-95(M)"].max().idxmax()
    print("best model on validation:", best, flush=True)
    return find(f"{best}_best.pt")[0], best


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


def main() -> None:
    t0 = time.time()
    weights, model_name = best_model()
    model = YOLO(weights)
    photo_dir = os.path.dirname(find("photos/*.jpg")[0])
    qc = pd.read_csv(find("qc.csv")[0])
    os.makedirs(f"{OUT}/crops", exist_ok=True)
    os.makedirs(f"{OUT}/masks", exist_ok=True)
    rows = []
    files = qc.dropna(subset=["copy"])["copy"].tolist()
    for i, name in enumerate(files):
        rgb = cv2.cvtColor(cv2.imread(f"{photo_dir}/{name}"), cv2.COLOR_BGR2RGB)
        res = model.predict(rgb, imgsz=IMGSZ, conf=0.25, retina_masks=True, verbose=False)[0]
        row = {"copy": name, "model": model_name}
        if res.masks is not None and len(res.masks):
            conf = res.boxes.conf.cpu().numpy()
            k = int(np.argmax(conf))
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
        m = measure(mask, rgb)
        row.update(m.as_dict())
        ys, xs = np.nonzero(mask)
        if len(ys):
            cy, cx = ys.mean(), xs.mean()
            bg = np.median(rgb[~mask], axis=0).astype(np.uint8)
            row.update(centroid_y=cy, centroid_x=cx,
                       bbox_h=int(ys.max() - ys.min() + 1), bbox_w=int(xs.max() - xs.min() + 1),
                       touches_border=bool(ys.min() == 0 or xs.min() == 0
                                           or ys.max() == rgb.shape[0] - 1
                                           or xs.max() == rgb.shape[1] - 1))
            crop = fixed_crop(rgb, cy, cx, bg)
            cmask = fixed_crop(mask.astype(np.uint8) * 255, cy, cx, 0)
            stem = os.path.splitext(name)[0]
            cv2.imwrite(f"{OUT}/crops/{stem}.jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(f"{OUT}/masks/{stem}.png", cmask)
            row["fits_crop"] = bool(max(row["bbox_h"], row["bbox_w"]) <= CROP)
        rows.append(row)
        if i % 100 == 0:
            print(f"{i}/{len(files)} {(time.time() - t0) / 60:.1f} min", flush=True)
    df = pd.DataFrame(rows).merge(qc[["copy", "file", "larva_id", "scale", "width", "height"]],
                                  on="copy", how="left")
    df.to_csv(f"{OUT}/measurements.csv", index=False)
    print(df["method"].value_counts().to_dict(), f"done {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
