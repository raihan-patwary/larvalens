"""Compare the data with the expected values in CLAUDE.md and write a report.

Expected values are checked, never forced: a mismatch is reported and explained,
and the data is left as it is.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from larvalens.paths import RESULTS


@dataclass
class Check:
    """One expected-value check."""

    item: str
    expected: str
    found: str
    match: bool | None  # None means pending (data not yet available)
    note: str = ""


def pct(n: int, total: int) -> str:
    """Format n out of total as a whole percentage."""
    return f"{100 * n / total:.0f}%"


def spreadsheet_checks(
    labels: pd.DataFrame, batches: pd.DataFrame, grouped: pd.DataFrame
) -> list[Check]:
    """Run every expected-value check that needs only the spreadsheet."""
    n = len(labels)
    zeros = int(labels["width_zero"].sum())
    width_filled = int(labels["width_mm"].notna().sum()) + zeros  # zeros were filled in the file
    sex = labels["sex"].value_counts()
    named = labels["batch"].dropna().unique()
    no_notes = sorted(set(named) - set(batches["batch"]))
    both = labels["age_days"].notna() & labels["temp_c"].notna()
    r = labels["weight_g"].corr(labels["length_mm"])
    w, ln = labels["weight_g"], labels["length_mm"]
    return [
        Check("Larva Data rows", "1,613", f"{n:,}", n == 1613),
        Check(
            "Weight (g)",
            "complete, 0.058 to 0.353",
            f"{w.notna().sum():,}/{n:,}, {w.min():.3f} to {w.max():.3f}",
            bool(w.notna().all() and w.min() == 0.058 and w.max() == 0.353),
        ),
        Check(
            "Length (mm)",
            "complete, 11 to 26",
            f"{ln.notna().sum():,}/{n:,}, {ln.min():.2f} to {ln.max():.2f}",
            bool(ln.notna().all() and round(ln.min()) == 11 and int(ln.max()) == 26),
            "The maximum is 26.43 mm; the expected range was rounded.",
        ),
        Check(
            "Width (mm)",
            "92% filled, 52 zeros",
            f"{pct(width_filled, n)} filled ({width_filled:,}), {zeros} zeros",
            pct(width_filled, n) == "92%" and zeros == 52,
            "Zeros are set to missing in labels.csv.",
        ),
        Check(
            "Sex",
            "91% filled, 1,079 female, 383 male",
            f"{pct(int(sex.sum()), n)} filled, {sex.get('F', 0):,} female, {sex.get('M', 0)} male",
            pct(int(sex.sum()), n) == "91%" and sex.get("F") == 1079 and sex.get("M") == 383,
        ),
        Check(
            "Batch_ID",
            "86% filled, 219 blank, 12 named batches; B011 and B013 have no notes",
            f"{pct(int(labels['batch'].notna().sum()), n)} filled, "
            f"{int(labels['batch'].isna().sum())} blank, {len(named)} named; "
            f"no notes: {', '.join(no_notes)}",
            int(labels["batch"].isna().sum()) == 219
            and len(named) == 12
            and no_notes == ["B011", "B013"],
        ),
        Check(
            "Batch_Info",
            "hatch dates for 10 batches, temperature (25 or 29 °C) for 9",
            f"hatch dates {int(batches['hatch_date'].notna().sum())}, temperature "
            f"{int(batches['temp_c'].notna().sum())} "
            f"({', '.join(f'{t:.0f}' for t in sorted(batches['temp_c'].dropna().unique()))} °C)",
            int(batches["hatch_date"].notna().sum()) == 10
            and int(batches["temp_c"].notna().sum()) == 9,
        ),
        Check(
            "Rows with both age and temperature",
            "1,001",
            f"{int(both.sum()):,}",
            both.sum() == 1001,
        ),
        Check(
            "Grouped_Larva rows", "24", f"{len(grouped)}", len(grouped) == 24, "Kept out of models."
        ),
        Check("Pearson r, Weight vs Length", "0.86", f"{r:.2f}", round(r, 2) == 0.86),
    ]


def write_report(checks: list[Check], findings: list[str], path: Path | None = None) -> Path:
    """Write results/data_checks.md."""
    path = path or RESULTS / "data_checks.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    mark = {True: "yes", False: "**no**", None: "pending"}
    lines = [
        "# Data checks",
        "",
        "Expected values from CLAUDE.md, checked against the downloaded files. "
        "Mismatches are reported and explained; the data is never edited to match.",
        "",
        "| Item | Expected | Found | Match | Note |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {c.item} | {c.expected} | {c.found} | {mark[c.match]} | {c.note} |" for c in checks
    ]
    if findings:
        lines += ["", "## Other findings", ""] + [f"- {f}" for f in findings]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
