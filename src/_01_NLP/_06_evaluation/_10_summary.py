from __future__ import annotations

import pandas as pd
import numpy as np

from dataclasses import dataclass
from typing import Literal


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

    def __init__(self, y_true, y_pred) -> None:
        """Initialize the ClassificationEvaluation class."""
        self.y_true = y_true
        self.y_pred = y_pred

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
        PR = []
        ROC = []
        top_k_accuracy = []

        return ClassificationCurves(PR, ROC, top_k_accuracy)
    
    def _calculate_operational_metrics(self) -> OperationalMetrics:
        """Calculate the operational metrics."""
        coverage = 0.0
        abstention_rate = 0.0

        return OperationalMetrics(coverage, abstention_rate)