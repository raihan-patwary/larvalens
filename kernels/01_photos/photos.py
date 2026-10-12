"""LarvaLens 01: single-larva photos.

Downloads Single Larvae data.zip from Zenodo, checks its md5, and for every photo
records size, blur (variance of the Laplacian), brightness, a perceptual hash
and the larva ID parsed from the file name. Saves qc.csv, match.csv,
zip_listing.csv and copies of each photo with the long side at most 2048 px.
"""

import hashlib
import io
import os
import re
import sys
import time
import zipfile
from urllib.parse import quote

import cv2
import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageOps

NAME = "Single Larvae data.zip"
MD5 = "ea931d09ac8533f909d73371d45bf43d"
URL = f"https://zenodo.org/records/13359376/files/{quote(NAME)}?download=1"
ZIP = "/kaggle/tmp/single.zip" if os.path.isdir("/kaggle/tmp") else "/tmp/single.zip"
OUT = "/kaggle/working"
PHOTOS = os.path.join(OUT, "photos")
LONG_SIDE = 2048
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


def download(url: str, path: str, tries: int = 4) -> None:
    """Stream a large file to disk, resuming after a dropped connection."""
    for attempt in range(1, tries + 1):
        done = os.path.getsize(path) if os.path.exists(path) else 0
        headers = {"Range": f"bytes={done}-"} if done else {}
        try:
            with requests.get(url, stream=True, timeout=120, headers=headers) as r:
                if r.status_code == 416:
                    return
                r.raise_for_status()
                mode = "ab" if done and r.status_code == 206 else "wb"
                with open(path, mode) as fh:
                    for block in r.iter_content(8 << 20):
                        fh.write(block)
            return
        except Exception as exc:  # noqa: BLE001
            print(f"download attempt {attempt} failed: {exc}", flush=True)
            time.sleep(20 * attempt)
    raise RuntimeError("download failed")


def md5sum(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while b := fh.read(8 << 20):
            h.update(b)
    return h.hexdigest()


def dhash(gray: np.ndarray, size: int = 8) -> str:
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def parse_id(name: str):
    nums = re.findall(r"\d+", os.path.splitext(os.path.basename(name))[0])
    return int(nums[-1]) if nums else None


def main() -> None:
    os.makedirs(os.path.dirname(ZIP), exist_ok=True)
    os.makedirs(PHOTOS, exist_ok=True)
    t0 = time.time()
    download(URL, ZIP)
    size = os.path.getsize(ZIP)
    found = md5sum(ZIP)
    print(f"downloaded {size / 1e9:.2f} GB in {(time.time() - t0) / 60:.1f} min, md5 {found}", flush=True)
    if found != MD5:
        sys.exit(f"md5 mismatch: expected {MD5}, found {found}")

    zf = zipfile.ZipFile(ZIP)
    infos = zf.infolist()
    pd.DataFrame(
        [{"entry": i.filename, "bytes": i.file_size, "is_dir": i.is_dir()} for i in infos]
    ).to_csv(os.path.join(OUT, "zip_listing.csv"), index=False)

    rows = []
    for k, info in enumerate(infos):
        name = info.filename
        base = os.path.basename(name)
        if info.is_dir() or not base.lower().endswith(IMAGE_EXT) or base.startswith("._"):
            continue
        raw = zf.read(info)
        row = {"entry": name, "file": base, "bytes": info.file_size,
               "md5": hashlib.md5(raw).hexdigest(), "larva_id": parse_id(base)}
        try:
            im = Image.open(io.BytesIO(raw))
            row["exif_orientation"] = im.getexif().get(274)
            im = ImageOps.exif_transpose(im).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            row["error"] = str(exc)[:200]
            rows.append(row)
            continue
        w, h = im.size
        scale = min(1.0, LONG_SIDE / max(w, h))
        if scale < 1.0:
            im = im.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
        out_name = f"{os.path.splitext(base)[0]}.jpg"
        im.save(os.path.join(PHOTOS, out_name), quality=92)
        gray = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2GRAY)
        row.update(width=w, height=h, scale=scale, copy=out_name,
                   blur=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                   brightness=float(gray.mean()), contrast=float(gray.std()),
                   dhash=dhash(gray))
        rows.append(row)
        if k % 200 == 0:
            print(f"{k}/{len(infos)} entries, {(time.time() - t0) / 60:.1f} min", flush=True)

    qc = pd.DataFrame(rows)
    qc.to_csv(os.path.join(OUT, "qc.csv"), index=False)
    match = qc[["file", "copy", "larva_id"]].copy()
    match["duplicate_id"] = match["larva_id"].duplicated(keep=False) & match["larva_id"].notna()
    match.to_csv(os.path.join(OUT, "match.csv"), index=False)
    print(f"photos {len(qc)}, unreadable {qc.get('error', pd.Series(dtype=str)).notna().sum()}, "
          f"ids parsed {match['larva_id'].notna().sum()}, total {(time.time() - t0) / 60:.1f} min")
    os.remove(ZIP)


if __name__ == "__main__":
    main()
