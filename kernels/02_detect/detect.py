"""LarvaLens 02: larva detection and instance segmentation.

Downloads BSF_Larvae_v1.zip from Zenodo, checks its md5, merges the two class IDs
(both mean "larva"), builds train, val and test folders from splits/detect.csv
(grouped by source photo), and trains YOLOv8n-seg, YOLOv8s-seg and YOLOv8m-seg.
Saves test metrics, per-image counts, per-epoch curves, best weights and the
test predictions as polygons (figures are drawn later in the house style).
"""

import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.3.40"], check=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

URL = "https://zenodo.org/records/13359376/files/BSF_Larvae_v1.zip?download=1"
MD5 = "a5a104e2c10bbc499ad3c45b71d2ffd8"
WORK = "/kaggle/temp" if os.path.isdir("/kaggle/temp") else "/tmp/larvalens"
OUT = "/kaggle/working"
SEED = 42
EPOCHS = 100
MODELS = ["yolov8n-seg", "yolov8s-seg", "yolov8m-seg"]
CONF = 0.25  # Ultralytics default confidence for counting


def find_support_file(name: str) -> str:
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if hits:
        return hits[0]
    for z in glob.glob("/kaggle/input/**/splits.zip", recursive=True):
        zipfile.ZipFile(z).extractall(f"{WORK}/support")
        hits = glob.glob(f"{WORK}/support/**/{name}", recursive=True)
        if hits:
            return hits[0]
    raise FileNotFoundError(name)


def md5sum(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while b := fh.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def build_dataset() -> tuple[str, pd.DataFrame]:
    os.makedirs(WORK, exist_ok=True)
    zpath = f"{WORK}/det.zip"
    with requests.get(URL, timeout=300) as r:
        r.raise_for_status()
        open(zpath, "wb").write(r.content)
    found = md5sum(zpath)
    print("md5", found, flush=True)
    if found != MD5:
        sys.exit(f"md5 mismatch: expected {MD5}, found {found}")
    zipfile.ZipFile(zpath).extractall(f"{WORK}/raw")
    src = glob.glob(f"{WORK}/raw/**/Images", recursive=True)[0].rsplit("/", 1)[0]
    split = pd.read_csv(find_support_file("detect.csv"))
    root = f"{WORK}/yolo"
    for part in ("train", "val", "test"):
        os.makedirs(f"{root}/images/{part}", exist_ok=True)
        os.makedirs(f"{root}/labels/{part}", exist_ok=True)
    counts = []
    for rec in split.itertuples(index=False):
        stem = rec.image.rsplit(".", 1)[0]
        shutil.copy(f"{src}/Images/{rec.image}", f"{root}/images/{rec.split}/{rec.image}")
        lines = []
        for line in open(f"{src}/Labels/{stem}.txt"):
            parts = line.split()
            if parts:
                lines.append(" ".join(["0"] + parts[1:]))  # merge class 0 and 1
        open(f"{root}/labels/{rec.split}/{stem}.txt", "w").write("\n".join(lines) + "\n")
        counts.append({"image": rec.image, "source": rec.source, "split": rec.split,
                       "true_count": len(lines)})
    yaml = f"path: {root}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: larva\n"
    open(f"{root}/data.yaml", "w").write(yaml)
    return f"{root}/data.yaml", pd.DataFrame(counts)


def metrics_row(name: str, m) -> dict:
    return {
        "model": name,
        "box_precision": m.box.mp, "box_recall": m.box.mr,
        "box_map50": m.box.map50, "box_map50_95": m.box.map,
        "mask_precision": m.seg.mp, "mask_recall": m.seg.mr,
        "mask_map50": m.seg.map50, "mask_map50_95": m.seg.map,
    }


def main() -> None:
    t0 = time.time()
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none", flush=True)
    data, counts = build_dataset()
    counts[["image", "source", "split"]].to_csv(f"{OUT}/detect_split_used.csv", index=False)
    rows, epochs, preds = [], [], {}
    test_imgs = sorted(glob.glob(data.rsplit("/", 1)[0] + "/images/test/*.jpg"))
    for name in MODELS:
        model = YOLO(f"{name}.pt")
        model.train(data=data, epochs=EPOCHS, imgsz=640, batch=16, seed=SEED, deterministic=True,
                    project=f"{WORK}/runs", name=name, exist_ok=True, patience=0, plots=False,
                    verbose=False, workers=2)
        run = f"{WORK}/runs/{name}"
        ep = pd.read_csv(f"{run}/results.csv")
        ep.columns = [c.strip() for c in ep.columns]
        ep.insert(0, "model", name)
        epochs.append(ep)
        best = YOLO(f"{run}/weights/best.pt")
        test = best.val(data=data, split="test", imgsz=640, batch=16, plots=False, verbose=False)
        rows.append(metrics_row(name, test))
        shutil.copy(f"{run}/weights/best.pt", f"{OUT}/{name}_best.pt")
        preds[name] = {}
        for res in best.predict(test_imgs, imgsz=640, conf=0.05, verbose=False, stream=True):
            img = os.path.basename(res.path)
            confs = res.boxes.conf.cpu().numpy().tolist() if res.boxes is not None else []
            polys = [p.round(1).tolist() for p in res.masks.xy] if res.masks is not None else []
            preds[name][img] = {"conf": [round(c, 4) for c in confs], "polygons": polys}
            counts.loc[counts["image"] == img, f"pred_{name}"] = sum(c >= CONF for c in confs)
        print(name, rows[-1], f"{(time.time() - t0) / 60:.1f} min", flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/metrics_test.csv", index=False)
    pd.concat(epochs).to_csv(f"{OUT}/epochs.csv", index=False)
    counts[counts["split"] == "test"].to_csv(f"{OUT}/counts_test.csv", index=False)
    json.dump(preds, open(f"{OUT}/predictions_test.json", "w"))
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
