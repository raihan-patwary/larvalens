"""Clean the Larvae--size-data.xlsx spreadsheet into tidy tables.

Outputs:
- data/processed/batches.csv: batch, hatch date, rearing temperature, notes
- data/processed/labels.csv: one row per larva with cleaned measurements and age
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from larvalens.paths import PROCESSED, RAW

XLSX = "Larvae--size-data.xlsx"

MONTHS = {
    m: i
    for i, m in enumerate(
        [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ],
        start=1,
    )
}
DATE_RE = re.compile(r"(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)\s+(\d{4})")
TEMP_RE = re.compile(r"(\d{2})\s*degrees", re.IGNORECASE)
BATCH_RE = re.compile(r"^B\d{3}$")

LABEL_COLUMNS = {
    "Batch_ID": "batch",
    "Date": "date",
    "L_ID": "larva_id",
    "Weight": "weight_g",
    "Length": "length_mm",
    "Width": "width_mm",
    "Sex": "sex",
}


def parse_hatch_date(text: str) -> pd.Timestamp | None:
    """Return the first 'hatched on <day> <Month> <year>' date in a note, if any."""
    for day, month, year in DATE_RE.findall(text):
        month_num = MONTHS.get(month.lower())
        if month_num:
            return pd.Timestamp(int(year), month_num, int(day))
    return None


def parse_temperature(text: str) -> float | None:
    """Return the rearing temperature in °C named in a note, if any."""
    match = TEMP_RE.search(text)
    return float(match.group(1)) if match else None


def parse_batch_info(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn the free-text Batch_Info sheet into one row per batch.

    The sheet has a batch ID in the first column and one or more note lines in
    the second; continuation lines have an empty first column.
    """
    notes: dict[str, list[str]] = {}
    current: str | None = None
    for key, text in raw.iloc[:, :2].itertuples(index=False):
        if isinstance(key, str) and BATCH_RE.match(key.strip()):
            current = key.strip()
            notes[current] = []
        if current is not None and isinstance(text, str) and text.strip():
            notes[current].append(text.strip())
    rows = []
    for batch, lines in notes.items():
        joined = " ".join(lines)
        rows.append(
            {
                "batch": batch,
                "hatch_date": parse_hatch_date(joined),
                "temp_c": parse_temperature(joined),
                "notes": joined,
            }
        )
    out = pd.DataFrame(rows, columns=["batch", "hatch_date", "temp_c", "notes"])
    out["hatch_date"] = pd.to_datetime(out["hatch_date"])
    return out


def clean_labels(raw: pd.DataFrame, batches: pd.DataFrame) -> pd.DataFrame:
    """Clean the Larva Data sheet and join batch information.

    Zero widths are recording errors and become missing. Age is the number of
    days from hatch date to measurement date, where the hatch date is known.
    """
    df = raw[list(LABEL_COLUMNS)].rename(columns=LABEL_COLUMNS).copy()
    df["batch"] = df["batch"].astype("string").str.strip()
    df["sex"] = df["sex"].astype("string").str.strip().str.upper()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["width_zero"] = df["width_mm"].eq(0)
    df.loc[df["width_zero"], "width_mm"] = np.nan
    df = df.merge(batches[["batch", "hatch_date", "temp_c"]], on="batch", how="left")
    df["age_days"] = (df["date"] - df["hatch_date"]).dt.days.astype("Int64")
    return df.sort_values("larva_id").reset_index(drop=True)


def load_sheets(path: Path) -> dict[str, pd.DataFrame]:
    """Read the sheets that cleaning needs."""
    return {
        "larva": pd.read_excel(path, sheet_name="Larva Data"),
        "batch_info": pd.read_excel(path, sheet_name="Batch_Info", header=None),
        "grouped": pd.read_excel(path, sheet_name="Grouped_Larva"),
    }


def run(raw_dir: Path = RAW, out_dir: Path = PROCESSED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean the spreadsheet and write batches.csv and labels.csv."""
    sheets = load_sheets(raw_dir / XLSX)
    batches = parse_batch_info(sheets["batch_info"])
    labels = clean_labels(sheets["larva"], batches)
    out_dir.mkdir(parents=True, exist_ok=True)
    batches.to_csv(out_dir / "batches.csv", index=False)
    labels.to_csv(out_dir / "labels.csv", index=False)
    return batches, labels
