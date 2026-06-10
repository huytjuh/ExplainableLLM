from __future__ import annotations

import pandas as pd
import numpy as np

from dataclasses import dataclass
from typing import Literal

from sklearn.metrics import roc_curve, precision_recall_curve


@dataclass(frozen=True)
class ClassificationMetrics:
    tp: int
    tn: int
    fp: int
    fn: int

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)
    
    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp)
    
    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn)
    
    @property
    def f1(self) -> float:
        return 2 * self.precision * self.recall / (self.precision + self.recall)

    @property
    def specificity(self) -> float:
        return self.tn / (self.tn + self.fp)

    @property
    def confusion_matrix(self) -> np.ndarray:
        return np.array([[self.tn, self.fp], [self.fn, self.tp]])

@dataclass(frozen=True)
class ClassificationCurves:
    PR: list[tuple[float, float]]
    ROC: list[tuple[float, float]]
    top_k_accuracy: list[tuple[str, float]]
    
    @property
    def auc_pr(self) -> float:
        return
    
    @property
    def auc_roc(self) -> float:
        return

@dataclass(frozen=True)
class OperationalMetrics:
    coverage: float
    abstention_rate: float


class ClassificationEvaluation():
    """Evaluates the performance of a classification model."""

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> None:
        """Initialize the ClassificationEvaluation class."""
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob

        self.classification_metrics = self._calculate_classification_metrics()
        self.classification_curves = self._calculate_classification_curves()
        self.operational_metrics = self._calculate_operational_metrics()

    def _calculate_classification_metrics(self) -> ClassificationMetrics:
        """Calculate the classification metrics."""
        tp = np.sum((self.y_true == 1) & (self.y_pred == 1))
        tn = np.sum((self.y_true == 0) & (self.y_pred == 0))
        fp = np.sum((self.y_true == 0) & (self.y_pred == 1))
        fn = np.sum((self.y_true == 1) & (self.y_pred == 0))

        return ClassificationMetrics(tp, tn, fp, fn)

    def _calculate_classification_curves(self) -> ClassificationCurves:
        """Calculate the classification curves."""
        y_true = self.y_true
        y_prob = self.y_prob

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ROC = list(zip(fpr.tolist(), tpr.tolist()))

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        PR = list(zip(recall.tolist(), precision.tolist()))

        top_k_accuracy = []
        k_values = [1, 3, 5]
        for k in k_values:
            if k == 1:
                acc = (y_true == self.y_pred).mean()
            else:
                acc = (y_true == self.y_pred).mean()
            top_k_accuracy.append((f"k={k}", float(acc)))

        return ClassificationCurves(PR, ROC, top_k_accuracy)
    
    def _calculate_operational_metrics(self, threshold: float=0.5) -> OperationalMetrics:
        """Calculate the operational metrics."""
        max_prob = np.max(self.y_prob, axis=1) if self.y_prob.shape[1] > 1 else np.maximum(self.y_prob, 1 - self.y_prob)
        abstained = np.sum(max_prob < threshold)
        
        abstention_rate = abstained / len(self.y_true)
        coverage = 1 - abstention_rate

        return OperationalMetrics(coverage, abstention_rate)