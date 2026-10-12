"""Tests for sex.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from larvalens import sex


def test_perfect_and_chance_scores() -> None:
    s = pd.Series(["F", "F", "M", "M"])
    perfect = sex.classification_scores(s, pd.Series([0.1, 0.2, 0.9, 0.8]))
    assert perfect["accuracy"] == 1 and perfect["roc_auc"] == 1 and perfect["f1_macro"] == 1
    all_female = sex.classification_scores(s, pd.Series([0.1] * 4))
    assert all_female["balanced_accuracy"] == pytest.approx(0.5)
    assert all_female["f1_male"] == 0


def test_score_runs_and_confusion() -> None:
    preds = pd.DataFrame(
        {
            "sex": ["F", "M", "F", "M"],
            "p_male": [0.2, 0.7, 0.6, 0.4],
            "eval": ["cv", "cv", "cv", "cv"],
            "run": ["fold 0", "fold 0", "fold 1", "fold 1"],
        }
    )
    out = sex.score_runs(preds).set_index("run")
    assert out.loc["fold 0", "accuracy"] == 1 and out.loc["fold 1", "accuracy"] == 0
    assert out.loc["pooled out-of-fold", "n"] == 4
    cm = sex.normalised_confusion(preds["sex"], preds["p_male"])
    np.testing.assert_allclose(cm.sum(axis=1), [1, 1])
