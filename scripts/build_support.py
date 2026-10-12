"""Assemble build/support: the files Kaggle notebooks need from this repo.

The folder becomes the private Kaggle dataset larvalens-support. It holds the
larvalens package source, the splits, the configs and the cleaned labels.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build" / "support"
OWNER = "mdraihanpatwary02"


def main() -> None:
    """Rebuild build/support from scratch."""
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(ROOT / "src" / "larvalens", OUT / "larvalens", ignore=ignore)
    shutil.copytree(ROOT / "configs", OUT / "configs", ignore=ignore)
    labels = ROOT / "data" / "processed" / "labels.csv"
    if labels.exists():
        (OUT / "labels").mkdir()
        shutil.copy(labels, OUT / "labels" / "labels.csv")
    if (ROOT / "splits").exists():
        shutil.copytree(ROOT / "splits", OUT / "splits", ignore=ignore)
    meta = {
        "title": "larvalens-support",
        "id": f"{OWNER}/larvalens-support",
        "licenses": [{"name": "other"}],
        "subtitle": "Support code for the LarvaLens Kaggle notebooks",
    }
    (OUT / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"built {OUT}")


if __name__ == "__main__":
    main()
