"""Project paths and config loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
SPLITS = ROOT / "splits"
OUTPUTS = ROOT / "outputs"
CONFIGS = ROOT / "configs"


def load_config(name: str = "default") -> dict[str, Any]:
    """Load configs/<name>.yaml as a dict."""
    with open(CONFIGS / f"{name}.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
