from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


EVALUATION_DIR = Path(__file__).resolve().parents[1] / "src" / "_01_NLP" / "_06_evaluation"
sys.path.insert(0, str(EVALUATION_DIR))

from _01_classification import ClassificationEvaluation, ClassificationMetrics


def test_classification_metrics_from_predictions():
    metrics = ClassificationMetrics.from_predictions(
        y_true=[1, 1, 0, 0],
        y_pred=[1, 0, 1, 0],
    )

    assert metrics.tp == 1
    assert metrics.tn == 1
    assert metrics.fp == 1
    assert metrics.fn == 1
    assert metrics.accuracy() == 0.5
    assert metrics.precision() == 0.5
    assert metrics.recall() == 0.5
    assert metrics.f1() == 0.5
    assert np.array_equal(metrics.confusion_matrix(), np.array([[1, 1], [1, 1]]))


def test_classification_evaluation_from_scores_builds_curves():
    evaluation = ClassificationEvaluation(
        y_true=[1, 1, 0, 0],
        y_score=[0.9, 0.8, 0.4, 0.1],
    )

    assert evaluation.metrics().as_dict()["accuracy"] == 1.0
    assert evaluation.auc("ROC") == 1.0
    assert evaluation.auc("PR") > 0.0
    assert {"threshold", "fpr", "tpr"}.issubset(evaluation.roc_curve().columns)
    assert {"threshold", "recall", "precision"}.issubset(evaluation.pr_curve().columns)


def test_zero_division_metrics_return_zero():
    metrics = ClassificationMetrics(tp=0, tn=2, fp=0, fn=2)

    assert metrics.precision() == 0.0
    assert metrics.recall() == 0.0
    assert metrics.f1() == 0.0
