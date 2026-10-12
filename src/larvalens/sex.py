"""Scores for the sex classifier, from saved out-of-fold probabilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

PAPER = {"accuracy": 0.74, "f1": 0.75, "fusion_accuracy": 0.69}


def classification_scores(sex: pd.Series, p_male: pd.Series, threshold: float = 0.5) -> dict:
    """Accuracy, balanced accuracy, per-class F1, ROC-AUC and PR-AUC (male = positive)."""
    y = (sex == "M").astype(int).to_numpy()
    pred = (np.asarray(p_male) >= threshold).astype(int)
    out = {
        "n": int(len(y)),
        "n_male": int(y.sum()),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1_female": float(f1_score(y, pred, pos_label=0, zero_division=0)),
        "f1_male": float(f1_score(y, pred, pos_label=1, zero_division=0)),
    }
    out["f1_macro"] = (out["f1_female"] + out["f1_male"]) / 2
    if 0 < y.sum() < len(y):
        out["roc_auc"] = float(roc_auc_score(y, p_male))
        out["pr_auc_male"] = float(average_precision_score(y, p_male))
    return out


def score_runs(preds: pd.DataFrame) -> pd.DataFrame:
    """One row per run (fold, bonus, random) plus a pooled out-of-fold row."""
    rows = [
        {"run": run, **classification_scores(g["sex"], g["p_male"])}
        for run, g in preds.groupby("run", sort=False)
    ]
    cv = preds[preds["eval"] == "cv"]
    rows.append({"run": "pooled out-of-fold", **classification_scores(cv["sex"], cv["p_male"])})
    return pd.DataFrame(rows)


def normalised_confusion(sex: pd.Series, p_male: pd.Series) -> np.ndarray:
    """Row-normalised confusion matrix, rows = true (female, male)."""
    y = (sex == "M").astype(int)
    cm = confusion_matrix(y, (np.asarray(p_male) >= 0.5).astype(int), labels=[0, 1])
    return cm / cm.sum(axis=1, keepdims=True)
