"""Tests for weights.py on synthetic larvae with a known power law."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from larvalens import weights

K = 0.07  # mm per pixel used to make the synthetic data


def synthetic(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    length_mm = rng.uniform(12, 25, n)
    width_mm = 0.27 * length_mm * rng.normal(1, 0.03, n)
    weight = 2.5e-4 * length_mm**2.3 * rng.normal(1, 0.03, n)
    df = pd.DataFrame(
        {
            "larva_id": np.arange(n),
            "length_mm": length_mm,
            "weight_g": weight,
            "length_px": length_mm / K * rng.normal(1, 0.01, n),
            "width_mid_px": width_mm / K,
            "width_max_px": width_mm / K * 1.05,
            "area_px": 0.8 * length_mm * width_mm / K**2,
            "perimeter_px": 2.1 * length_mm / K,
        }
    )
    df["fold"] = (df["larva_id"] % 5).astype(str)
    df.loc[df.index[-50:], "fold"] = "no_batch"
    df["random_split"] = np.where(rng.random(n) < 0.2, "test", "train")
    return df


def test_fit_mm_per_px_recovers_scale() -> None:
    df = synthetic()
    assert weights.fit_mm_per_px(df["length_px"], df["length_mm"]) == pytest.approx(K, rel=0.01)


def test_power_law_fits_well() -> None:
    metrics, preds = weights.evaluate(synthetic(), weights.power_law_length, "power")
    cv = metrics[metrics["eval"].str.startswith("fold")]
    assert len(cv) == 5 and (cv["r2"] > 0.9).all()
    assert set(preds["eval"]) == {"cv", "bonus", "random"}
    # every batched larva gets exactly one out-of-fold prediction
    assert preds[preds["eval"] == "cv"]["larva_id"].is_unique


def test_all_models_run_and_summarise() -> None:
    df = synthetic()
    all_metrics = pd.concat(
        [weights.evaluate(df, f, name)[0] for name, f in weights.MODELS.items()]
    )
    summary = weights.summarise(all_metrics)
    assert set(summary["model"]) == set(weights.MODELS)
    assert {"cv_r2_mean", "cv_r2_std", "bonus_r2", "random_r2"} <= set(summary.columns)
