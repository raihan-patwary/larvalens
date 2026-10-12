"""LarvaLens 05: larval sex from the image.

Downloads Larva_sex_classification.zip from Zenodo and checks its md5. Uses the
usable crops listed in splits/sex_crops.csv (readable, one crop per larva ID,
sex known). Each crop is padded to a square with its border colour and resized
to 224 px. A timm ResNet-18 is fine-tuned with class weights on the same batch
folds as the weight models, then on all batched larvae for the no-batch bonus
set, and on a stratified random 80/20 split for comparison with the paper.
Saves the probability of "male" for every held-out crop and a Grad-CAM map
(7 x 7, last convolutional block) for each, so overlays can be drawn locally.
"""

import glob
import hashlib
import io
import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import requests
import timm
import torch
import torch.nn.functional as F
from PIL import Image

T0 = time.time()
DEADLINE_MIN = 100
INPUT = os.environ.get("LARVALENS_INPUT", "/kaggle/input")
OUT = os.environ.get("LARVALENS_OUT", "/kaggle/working")
ZIP_PATH = os.environ.get("LARVALENS_SEX_ZIP", "/tmp/sex.zip")
URL = "https://zenodo.org/records/13359376/files/Larva_sex_classification.zip?download=1"
MD5 = "0e28870daeb6bb4ffb1142aeb4811206"
SEED, SIZE, BATCH, LR = 42, 224, 32, 3e-4
EPOCHS = int(os.environ.get("LARVALENS_EPOCHS", 20))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
print("started", timm.__version__, torch.__version__, DEV, flush=True)


def find(pattern):
    return sorted(glob.glob(f"{INPUT}/**/{pattern}", recursive=True))


def fetch_zip():
    if not os.path.exists(ZIP_PATH):
        with requests.get(URL, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(ZIP_PATH, "wb") as fh:
                for b in r.iter_content(8 << 20):
                    fh.write(b)
    h = hashlib.md5(open(ZIP_PATH, "rb").read()).hexdigest()
    print("md5", h, flush=True)
    if h != MD5:
        sys.exit(f"md5 mismatch: expected {MD5}, found {h}")
    return zipfile.ZipFile(ZIP_PATH)


def square(im):
    """Pad to a square with the median border colour, then resize to SIZE."""
    a = np.asarray(im.convert("RGB"))
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    fill = tuple(int(v) for v in np.median(border, axis=0))
    h, w = a.shape[:2]
    side = max(h, w)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(Image.fromarray(a), ((side - w) // 2, (side - h) // 2))
    return np.asarray(canvas.resize((SIZE, SIZE), Image.BILINEAR))


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_tensor(b):
    return (torch.from_numpy(b).permute(0, 3, 1, 2).float() / 255 - MEAN) / STD


def dihedral(x, k):
    if k >= 4:
        x = torch.flip(x, dims=[3])
    return torch.rot90(x, k % 4, dims=[2, 3])


def train_and_predict(imgs, y, tr, te):
    counts = np.bincount(y[tr], minlength=2)
    weight = torch.tensor(len(tr) / (2 * counts), dtype=torch.float32, device=DEV)
    model = timm.create_model("resnet18", pretrained=True, num_classes=2).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    steps = EPOCHS * int(np.ceil(len(tr) / BATCH))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=DEV == "cuda")
    rng = np.random.default_rng(SEED)
    for _ in range(EPOCHS):
        model.train()
        order = rng.permutation(tr)
        for s in range(0, len(order), BATCH):
            b = order[s : s + BATCH]
            x = dihedral(to_tensor(imgs[b]), int(rng.integers(8))).to(DEV)
            t = torch.from_numpy(y[b]).to(DEV)
            with torch.autocast(DEV, enabled=DEV == "cuda"):
                loss = F.cross_entropy(model(x).float(), t, weight=weight)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
    model.eval()
    # probabilities with 8-way test-time augmentation
    probs = []
    with torch.no_grad():
        for s in range(0, len(te), 64):
            x = to_tensor(imgs[te[s : s + 64]]).to(DEV)
            p = torch.stack([F.softmax(model(dihedral(x, k)).float(), 1)[:, 1] for k in range(8)])
            probs.append(p.mean(0).cpu().numpy())
    probs = np.concatenate(probs)
    # Grad-CAM on the last block for the predicted class (no augmentation)
    feats = {}
    layer = model.layer4
    h1 = layer.register_forward_hook(lambda m, i, o: feats.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: feats.__setitem__("g", go[0]))
    cams = []
    for s in range(0, len(te), 32):
        x = to_tensor(imgs[te[s : s + 32]]).to(DEV).requires_grad_(True)
        out = model(x).float()
        cls = out.argmax(1)
        model.zero_grad()
        out.gather(1, cls[:, None]).sum().backward()
        w = feats["g"].mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * feats["a"]).sum(1)).detach()
        cam = cam / (cam.flatten(1).max(1).values.view(-1, 1, 1) + 1e-8)
        cams.append(cam.cpu().numpy())
    h1.remove()
    h2.remove()
    return probs, np.concatenate(cams)


def main():
    zf = fetch_zip()
    crops = pd.read_csv(find("splits/sex_crops.csv")[0])
    crops = crops[crops["usable"]].reset_index(drop=True)
    if "LARVALENS_LIMIT" in os.environ:  # local smoke test only
        crops = crops.sample(int(os.environ["LARVALENS_LIMIT"]), random_state=SEED).reset_index(drop=True)
    folds = pd.read_csv(find("splits/larvae.csv")[0], dtype={"fold": str})
    df = crops.merge(folds[["larva_id", "fold"]], on="larva_id", how="left")
    names = {os.path.basename(n): n for n in zf.namelist()}
    imgs = np.stack([square(Image.open(io.BytesIO(zf.read(names[f])))) for f in df["file"]])
    y = (df["sex"] == "M").to_numpy().astype(np.int64)
    print(f"{len(df)} crops ({y.sum()} male) at {(time.time() - T0) / 60:.1f} min", flush=True)
    rng = np.random.default_rng(SEED)
    rand_test = np.zeros(len(df), bool)
    for c in (0, 1):  # stratified random 80/20
        idx = np.where(y == c)[0]
        rand_test[rng.choice(idx, size=round(0.2 * len(idx)), replace=False)] = True
    batched = (df["fold"] != "no_batch").to_numpy()
    runs = [(f"fold {f}", np.where(batched & (df["fold"] != f))[0], np.where(df["fold"] == f)[0], "cv")
            for f in sorted(df.loc[batched, "fold"].unique())]
    runs.append(("bonus", np.where(batched)[0], np.where(~batched)[0], "bonus"))
    runs.append(("random", np.where(~rand_test)[0], np.where(rand_test)[0], "random"))
    rows, cam_list, cam_keys = [], [], []
    for tag, tr, te, ev in runs:
        if (time.time() - T0) / 60 > DEADLINE_MIN:
            print("deadline reached; skipping", tag, flush=True)
            continue
        p, cams = train_and_predict(imgs, y, tr, te)
        rows.append(pd.DataFrame({"file": df.loc[te, "file"].to_numpy(),
                                  "larva_id": df.loc[te, "larva_id"].to_numpy(),
                                  "sex": df.loc[te, "sex"].to_numpy(), "eval": ev, "run": tag,
                                  "p_male": p}))
        cam_list.append(cams)
        cam_keys += [f"{tag}|{f}" for f in df.loc[te, "file"]]
        acc = ((p > 0.5).astype(int) == y[te]).mean()
        print(f"{tag}: accuracy {acc:.3f}, {(time.time() - T0) / 60:.1f} min", flush=True)
    pd.concat(rows).to_csv(f"{OUT}/sex_predictions.csv", index=False)
    np.savez_compressed(f"{OUT}/gradcam.npz", cams=np.concatenate(cam_list).astype(np.float16),
                        keys=np.array(cam_keys))
    print(f"done in {(time.time() - T0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
