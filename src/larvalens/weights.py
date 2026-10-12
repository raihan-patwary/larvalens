"""Weight from shape: calibration, models and cross-validated evaluation.

Every model is a function fit_predict(train, test) -> predicted weight (g) for
the test rows. Anything fitted on data, including mm per pixel, is fitted on the
training rows only, so it is refitted inside each fold.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from larvalens.splits import NO_BATCH

FitPredict = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]


def fit_mm_per_px(length_px: np.ndarray, length_mm: np.ndarray) -> float:
    """Least-squares scale through the origin: length_mm ≈ k * length_px."""
    px = np.asarray(length_px, float)
    mm = np.asarray(length_mm, float)
    ok = np.isfinite(px) & np.isfinite(mm) & (px > 0)
    return float((px[ok] @ mm[ok]) / (px[ok] @ px[ok]))


def to_mm(df: pd.DataFrame, k: float) -> pd.DataFrame:
    """Convert pixel measurements to mm (and mm²) with scale k mm per pixel."""
    out = pd.DataFrame(index=df.index)
    out["img_length_mm"] = df["length_px"] * k
    out["img_width_mm"] = df["width_mid_px"] * k
    out["img_width_max_mm"] = df["width_max_px"] * k
    out["img_area_mm2"] = df["area_px"] * k**2
    out["img_perimeter_mm"] = df["perimeter_px"] * k
    return out


def _scaled(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    k = fit_mm_per_px(train["length_px"], train["length_mm"])
    return to_mm(train, k), to_mm(test, k)


def linear_area_length_width(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Linear regression on image area, length and width (the paper's best features)."""
    cols = ["img_area_mm2", "img_length_mm", "img_width_mm"]
    a, b = _scaled(train, test)
    model = LinearRegression().fit(a[cols], train["weight_g"])
    return model.predict(b[cols])


def power_law_length(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Allometric power law W = a * L^b, fitted as a line on log scales."""
    a, b = _scaled(train, test)
    slope, intercept = np.polyfit(np.log(a["img_length_mm"]), np.log(train["weight_g"]), 1)
    return np.exp(intercept) * b["img_length_mm"].to_numpy() ** slope


def gradient_boosting(train: pd.DataFrame, test: pd.DataFrame, seed: int = 42) -> np.ndarray:
    """Gradient boosting on every shape and colour feature."""
    a, b = _scaled(train, test)
    extra = [c for c in ("lab_l", "lab_a", "lab_b") if c in train]
    xa = pd.concat([a, train[extra]], axis=1)
    xb = pd.concat([b, test[extra]], axis=1)
    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=seed,
    )
    return model.fit(xa, train["weight_g"]).predict(xb)


MODELS: dict[str, FitPredict] = {
    "Linear (area, length, width)": linear_area_length_width,
    "Power law (length)": power_law_length,
    "Gradient boosting": gradient_boosting,
}


def scores(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """R², RMSE (g) and MAE (mg)."""
    return {
        "r2": float(r2_score(y, p)),
        "rmse_g": float(np.sqrt(mean_squared_error(y, p))),
        "mae_mg": float(mean_absolute_error(y, p) * 1000),
        "n": int(len(y)),
    }


def evaluate(
    df: pd.DataFrame, fit_predict: FitPredict, name: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run batch-fold CV, the no-batch bonus test and the random split for one model.

    Args:
        df: one row per larva with features, weight_g, length_mm, fold and random_split.
        fit_predict: the model.
        name: model name for the outputs.

    Returns:
        (metrics, predictions). Metrics has one row per fold plus 'bonus' and
        'random'; predictions holds the out-of-fold prediction for every batched
        larva and the bonus prediction for every unbatched larva.
    """
    rows, preds = [], []
    batched = df[df["fold"] != NO_BATCH]
    for fold in sorted(batched["fold"].unique()):
        train, test = batched[batched["fold"] != fold], batched[batched["fold"] == fold]
        p = fit_predict(train, test)
        rows.append({"model": name, "eval": f"fold {fold}", **scores(test["weight_g"], p)})
        preds.append(test[["larva_id"]].assign(model=name, eval="cv", pred_g=p))
    bonus = df[df["fold"] == NO_BATCH]
    if len(bonus):
        p = fit_predict(batched, bonus)
        rows.append({"model": name, "eval": "bonus", **scores(bonus["weight_g"], p)})
        preds.append(bonus[["larva_id"]].assign(model=name, eval="bonus", pred_g=p))
    train, test = df[df["random_split"] == "train"], df[df["random_split"] == "test"]
    p = fit_predict(train, test)
    rows.append({"model": name, "eval": "random", **scores(test["weight_g"], p)})
    preds.append(test[["larva_id"]].assign(model=name, eval="random", pred_g=p))
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def summarise(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD across folds, next to the bonus and random-split scores."""
    folds = metrics[metrics["eval"].str.startswith("fold")]
    agg = folds.groupby("model")[["r2", "rmse_g", "mae_mg"]].agg(["mean", "std"])
    agg.columns = [f"cv_{m}_{s}" for m, s in agg.columns]
    for ev in ("bonus", "random"):
        part = metrics[metrics["eval"] == ev].set_index("model")[["r2", "rmse_g", "mae_mg"]]
        agg = agg.join(part.add_prefix(f"{ev}_"))
    return agg.reset_index()
