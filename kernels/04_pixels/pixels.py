"""LarvaLens 04: weight straight from the pixels.

Fine-tunes a timm ResNet-18 on the fixed-size crops from 03_measure to predict
weight (Huber loss on standardised log weight; flips and 90 degree rotations
only, so absolute size is kept). Every crop is shrunk by the same factor
(896 to 224 px). Runs: the five batch folds, the no-batch bonus set (trained
on all batched larvae) and the random 80/20 split. The last epoch is always
used. Also extracts DINOv2 ViT-S/14 features for every crop and a 2D UMAP.
"""

import glob
import os
import subprocess
import sys
import time

T0 = time.time()
DEADLINE_MIN = 100
print("started", flush=True)
if "LARVALENS_INPUT" not in os.environ:  # on Kaggle; a local test brings its own packages
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "umap-learn==0.5.12"], check=True,
                   timeout=900)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import timm  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

INPUT = os.environ.get("LARVALENS_INPUT", "/kaggle/input")
OUT = os.environ.get("LARVALENS_OUT", "/kaggle/working")
SEED = 42
SIZE = 224
EPOCHS = int(os.environ.get("LARVALENS_EPOCHS", 25))
BATCH = 64
LR = 3e-4
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
np.random.seed(SEED)
print("timm", timm.__version__, "torch", torch.__version__, DEV, flush=True)


def find(pattern):
    return sorted(glob.glob(f"{INPUT}/**/{pattern}", recursive=True))


def load_table():
    meas = pd.read_csv(find("measurements.csv")[0])
    labels = pd.read_csv(find("labels/labels.csv")[0])
    folds = pd.read_csv(find("splits/larvae.csv")[0], dtype={"fold": str})
    df = meas[["copy", "larva_id"]].merge(labels[["larva_id", "weight_g"]], on="larva_id")
    return df.merge(folds[["larva_id", "fold", "random_split"]], on="larva_id")


def load_crops(df):
    crop_dir = os.path.dirname(find("crops/*.jpg")[0])
    imgs = np.zeros((len(df), SIZE, SIZE, 3), np.uint8)
    for i, name in enumerate(df["copy"]):
        im = Image.open(f"{crop_dir}/{os.path.splitext(name)[0]}.jpg").convert("RGB")
        imgs[i] = np.asarray(im.resize((SIZE, SIZE), Image.BILINEAR))  # same factor for all
    return imgs


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_tensor(batch):
    x = torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255
    return (x - MEAN) / STD


def dihedral(x, k):
    """One of the 8 flips and 90 degree rotations."""
    if k >= 4:
        x = torch.flip(x, dims=[3])
    return torch.rot90(x, k % 4, dims=[2, 3])


def train_and_predict(imgs, y_log, train_idx, test_idx, tag):
    mu, sd = y_log[train_idx].mean(), y_log[train_idx].std()
    target = torch.tensor((y_log - mu) / sd, dtype=torch.float32)
    model = timm.create_model("resnet18", pretrained=True, num_classes=1).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    steps = EPOCHS * int(np.ceil(len(train_idx) / BATCH))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=DEV == "cuda")
    rng = np.random.default_rng(SEED)
    log = []
    for epoch in range(EPOCHS):
        model.train()
        order = rng.permutation(train_idx)
        total = 0.0
        for s in range(0, len(order), BATCH):
            b = order[s : s + BATCH]
            x = dihedral(to_tensor(imgs[b]), int(rng.integers(8))).to(DEV)
            t = target[b].to(DEV)
            with torch.autocast(DEV, enabled=DEV == "cuda"):
                loss = F.huber_loss(model(x).squeeze(1).float(), t, delta=1.0)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(b)
        log.append({"run": tag, "epoch": epoch + 1, "train_loss": total / len(train_idx)})
    model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(test_idx), 128):
            x = to_tensor(imgs[test_idx[s : s + 128]]).to(DEV)
            out = torch.stack([model(dihedral(x, k)).squeeze(1).float() for k in range(8)]).mean(0)
            preds.append(out.cpu().numpy())
    pred_log = np.concatenate(preds) * sd + mu
    return np.exp(pred_log), log


def main():
    df = load_table().reset_index(drop=True)
    imgs = load_crops(df)
    print(f"{len(df)} crops loaded at {(time.time() - T0) / 60:.1f} min", flush=True)
    y_log = np.log(df["weight_g"].to_numpy())
    batched = df["fold"] != "no_batch"
    runs = [(f"fold {f}", np.where(batched & (df["fold"] != f))[0], np.where(df["fold"] == f)[0], "cv")
            for f in sorted(df.loc[batched, "fold"].unique())]
    runs.append(("bonus", np.where(batched)[0], np.where(~batched)[0], "bonus"))
    runs.append(("random", np.where(df["random_split"] == "train")[0],
                 np.where(df["random_split"] == "test")[0], "random"))
    preds, logs = [], []
    for tag, tr, te, ev in runs:
        if (time.time() - T0) / 60 > DEADLINE_MIN:
            print("deadline reached; skipping", tag, flush=True)
            continue
        p, log = train_and_predict(imgs, y_log, tr, te, tag)
        preds.append(pd.DataFrame({"larva_id": df.loc[te, "larva_id"].to_numpy(),
                                   "model": "ResNet-18 (fine-tuned)", "eval": ev, "run": tag,
                                   "pred_g": p}))
        logs += log
        err = np.abs(p - df.loc[te, "weight_g"].to_numpy()).mean() * 1000
        print(f"{tag}: MAE {err:.1f} mg, {(time.time() - T0) / 60:.1f} min", flush=True)
    pd.concat(preds).to_csv(f"{OUT}/resnet18_predictions.csv", index=False)
    pd.DataFrame(logs).to_csv(f"{OUT}/resnet18_training.csv", index=False)

    # DINOv2 features (frozen), then a 2D UMAP
    dino = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0,
                             img_size=SIZE).to(DEV).eval()
    feats = []
    with torch.no_grad():
        for s in range(0, len(df), 128):
            x = to_tensor(imgs[s : s + 128]).to(DEV)
            f = torch.stack([dino(dihedral(x, k)).float() for k in range(4)]).mean(0)
            feats.append(f.cpu().numpy())
    feats = np.concatenate(feats).astype(np.float32)
    np.save(f"{OUT}/dinov2_features.npy", feats)
    df[["larva_id", "copy"]].to_csv(f"{OUT}/dinov2_ids.csv", index=False)
    import umap

    xy = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=SEED).fit_transform(feats)
    pd.DataFrame({"larva_id": df["larva_id"], "copy": df["copy"], "umap_x": xy[:, 0],
                  "umap_y": xy[:, 1]}).to_csv(f"{OUT}/umap.csv", index=False)
    print(f"done in {(time.time() - T0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
