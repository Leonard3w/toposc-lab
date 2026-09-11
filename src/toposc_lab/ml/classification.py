"""Explicit, class-balanced binary classification baselines and metrics."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    sample_count: int
    positive_count: int
    prevalence: float
    threshold: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float
    brier_score: float


class BalancedLogisticClassifier:
    """Standardized binary logistic regression with inverse-frequency weights."""

    def __init__(
        self,
        *,
        learning_rate: float = 0.05,
        epochs: int = 600,
        l2: float = 1e-4,
    ) -> None:
        self.learning_rate = _positive_float(learning_rate, "learning_rate")
        self.epochs = _positive_integer(epochs, "epochs")
        self.l2 = _nonnegative_float(l2, "l2")
        self._mean: NDArray[np.float64] | None = None
        self._scale: NDArray[np.float64] | None = None
        self._weights: NDArray[np.float64] | None = None

    def fit(self, features: ArrayLike, labels: ArrayLike) -> Self:
        matrix = _matrix(features)
        targets = _binary_labels(labels)
        if matrix.shape[0] != len(targets):
            raise ValueError("features and labels must contain the same number of samples")
        positive_count = int(np.sum(targets))
        negative_count = len(targets) - positive_count
        if positive_count == 0 or negative_count == 0:
            raise ValueError("classification training requires both classes")

        mean = np.mean(matrix, axis=0)
        scale = np.std(matrix, axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        design = np.column_stack((np.ones(len(targets)), (matrix - mean) / scale))
        sample_weights = np.where(
            targets == 1,
            len(targets) / (2.0 * positive_count),
            len(targets) / (2.0 * negative_count),
        )
        weights = np.zeros(design.shape[1], dtype=float)
        for _ in range(self.epochs):
            probabilities = _sigmoid(design @ weights)
            gradient = design.T @ (sample_weights * (probabilities - targets)) / len(targets)
            gradient[1:] += self.l2 * weights[1:]
            weights -= self.learning_rate * gradient
        self._mean = mean
        self._scale = scale
        self._weights = weights
        return self

    def predict_proba(self, features: ArrayLike) -> NDArray[np.float64]:
        if self._mean is None or self._scale is None or self._weights is None:
            raise RuntimeError("classifier must be fitted before prediction")
        matrix = _matrix(features)
        if matrix.shape[1] != len(self._mean):
            raise ValueError("prediction feature count does not match fitted data")
        design = np.column_stack((np.ones(matrix.shape[0]), (matrix - self._mean) / self._scale))
        return _sigmoid(design @ self._weights)

    def predict(
        self, features: ArrayLike, *, threshold: float = 0.5
    ) -> NDArray[np.int64]:
        selected = _probability(threshold, "threshold")
        return np.asarray(self.predict_proba(features) >= selected, dtype=np.int64)


def classification_metrics(
    labels: ArrayLike,
    probabilities: ArrayLike,
    *,
    threshold: float,
) -> ClassificationMetrics:
    """Compute threshold-dependent and probability metrics without hidden defaults."""
    truth = _binary_labels(labels)
    probability_values = _probabilities(probabilities)
    if truth.shape != probability_values.shape:
        raise ValueError("labels and probabilities must have the same shape")
    selected = _probability(threshold, "threshold")
    predictions = probability_values >= selected
    positives = truth == 1
    negatives = ~positives
    true_positive = int(np.sum(predictions & positives))
    false_positive = int(np.sum(predictions & negatives))
    true_negative = int(np.sum(~predictions & negatives))
    false_negative = int(np.sum(~predictions & positives))
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    specificity = _safe_ratio(true_negative, true_negative + false_positive)
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    return ClassificationMetrics(
        sample_count=len(truth),
        positive_count=int(np.sum(positives)),
        prevalence=float(np.mean(truth)),
        threshold=selected,
        true_positive=true_positive,
        false_positive=false_positive,
        true_negative=true_negative,
        false_negative=false_negative,
        accuracy=(true_positive + true_negative) / len(truth),
        balanced_accuracy=(recall + specificity) / 2.0,
        precision=precision,
        recall=recall,
        specificity=specificity,
        f1=_safe_ratio(2.0 * precision * recall, precision + recall),
        brier_score=float(np.mean(np.square(probability_values - truth))),
    )


def select_classification_threshold(
    labels: ArrayLike,
    probabilities: ArrayLike,
    *,
    candidates: ArrayLike | None = None,
) -> float:
    """Select a threshold on validation data by F1, then balanced accuracy."""
    truth = _binary_labels(labels)
    values = _probabilities(probabilities)
    if truth.shape != values.shape:
        raise ValueError("labels and probabilities must have the same shape")
    grid = (
        np.linspace(0.05, 0.95, 19)
        if candidates is None
        else np.asarray(candidates, dtype=float)
    )
    if grid.ndim != 1 or not len(grid):
        raise ValueError("threshold candidates must be a non-empty vector")
    thresholds = tuple(_probability(value, "threshold candidate") for value in grid)
    scored = tuple(
        (classification_metrics(truth, values, threshold=value), value)
        for value in thresholds
    )
    return min(
        scored,
        key=lambda item: (-item[0].f1, -item[0].balanced_accuracy, abs(item[1] - 0.5), item[1]),
    )[1]


def _matrix(values: ArrayLike) -> NDArray[np.float64]:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or not len(matrix):
        raise ValueError("features must be a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("features must be finite")
    return matrix


def _binary_labels(values: ArrayLike) -> NDArray[np.int64]:
    labels = np.asarray(values)
    if labels.ndim != 1 or not len(labels):
        raise ValueError("labels must be a non-empty one-dimensional vector")
    if not np.all(np.isin(labels, (0, 1, False, True))):
        raise ValueError("labels must be binary")
    return labels.astype(np.int64)


def _probabilities(values: ArrayLike) -> NDArray[np.float64]:
    probabilities = np.asarray(values, dtype=float)
    if probabilities.ndim != 1 or not len(probabilities):
        raise ValueError("probabilities must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must be finite values in [0, 1]")
    return probabilities


def _sigmoid(values: NDArray[np.float64]) -> NDArray[np.float64]:
    clipped = np.clip(values, -40.0, 40.0)
    return np.asarray(1.0 / (1.0 + np.exp(-clipped)), dtype=float)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0.0 else numerator / denominator


def _probability(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _positive_float(value: object, name: str) -> float:
    result = _nonnegative_float(value, name)
    if result == 0.0:
        raise ValueError(f"{name} must be positive")
    return result
