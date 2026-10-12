"""Tests for clean.py on small synthetic tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from larvalens import clean


@pytest.fixture
def batch_info() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [np.nan, np.nan],
            ["B001", "Larva belonging to this batch were hatched on 29th June 2023,"],
            [np.nan, "They have been kept at 29 degrees celcius"],
            [np.nan, np.nan],
            ["B002", "Larva were kept at 25 degrees, hatched on 2nd July 2023"],
            [
                "B003",
                "They were hatched from batch B001 flies. They were hatched on 8th August 2023",
            ],
        ]
    )


@pytest.fixture
def larva_sheet() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Batch_ID": ["B001", "B002", "B003", None, "B011"],
            "Date": pd.to_datetime(
                [
                    "2023-07-10 00:00",
                    "2023-07-14 10:30",
                    "2023-08-20 00:00",
                    "2023-08-27 00:00",
                    "2023-08-24 00:00",
                ]
            ),
            "L_ID": [1, 2, 3, 4, 5],
            "Weight": [0.1, 0.2, 0.15, 0.3, 0.25],
            "Length": [14.0, 20.0, 17.5, 22.0, 21.0],
            "Width": [np.nan, 0.0, 5.1, 5.5, 5.4],
            "Sex": ["F", "m ", None, "F", "M"],
        }
    )


def test_parse_hatch_date() -> None:
    assert clean.parse_hatch_date("hatched on 29th June 2023,") == pd.Timestamp(2023, 6, 29)
    assert clean.parse_hatch_date("hatched on 2nd July 2023") == pd.Timestamp(2023, 7, 2)
    assert clean.parse_hatch_date("Fed on chicken feed") is None


def test_parse_temperature() -> None:
    assert clean.parse_temperature("kept at 29 degrees celcius") == 29.0
    assert clean.parse_temperature("reared at 25 Degrees") == 25.0
    assert clean.parse_temperature("hatched on 8th August 2023") is None


def test_parse_batch_info_joins_continuation_lines(batch_info: pd.DataFrame) -> None:
    out = clean.parse_batch_info(batch_info).set_index("batch")
    assert list(out.index) == ["B001", "B002", "B003"]
    assert out.loc["B001", "hatch_date"] == pd.Timestamp(2023, 6, 29)
    assert out.loc["B001", "temp_c"] == 29.0
    assert out.loc["B002", "temp_c"] == 25.0
    assert np.isnan(out.loc["B003", "temp_c"])
    assert out.loc["B003", "hatch_date"] == pd.Timestamp(2023, 8, 8)


def test_clean_labels(batch_info: pd.DataFrame, larva_sheet: pd.DataFrame) -> None:
    labels = clean.clean_labels(larva_sheet, clean.parse_batch_info(batch_info))
    by_id = labels.set_index("larva_id")
    # zero width becomes missing and is flagged
    assert np.isnan(by_id.loc[2, "width_mm"]) and by_id.loc[2, "width_zero"]
    assert not by_id.loc[1, "width_zero"]
    # sex is tidied
    assert by_id.loc[2, "sex"] == "M"
    # age from hatch date to the measurement day (time of day ignored)
    assert by_id.loc[1, "age_days"] == 11
    assert by_id.loc[2, "age_days"] == 12
    # no batch, or a batch with no notes, gives unknown age and temperature
    assert pd.isna(by_id.loc[4, "age_days"]) and pd.isna(by_id.loc[5, "temp_c"])
    assert len(labels) == len(larva_sheet)
