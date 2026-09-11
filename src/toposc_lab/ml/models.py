"""Small reproducible surrogate baselines implemented with NumPy only."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Protocol, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray


class Regressor(Protocol):
    """Minimal common interface used by Phase-12 evaluation code."""

    def fit(self, features: ArrayLike, targets: ArrayLike) -> Self: ...

    def predict(self, features: ArrayLike) -> NDArray[np.float64]: ...


@dataclass(frozen=True, slots=True)
class RegressionMetrics:
    sample_count: int
    mae: float
    rmse: float
    r2: float


class MeanRegressor:
    """Trivial predictor used as the mandatory scientific sanity baseline."""

    def __init__(self) -> None:
        self._mean: float | None = None
        self.n_features_in_: int | None = None

    def fit(self, features: ArrayLike, targets: ArrayLike) -> Self:
        matrix, values = _training_arrays(features, targets)
        self._mean = float(np.mean(values))
        self.n_features_in_ = matrix.shape[1]
        return self

    def predict(self, features: ArrayLike) -> NDArray[np.float64]:
        matrix = _prediction_matrix(features, self.n_features_in_)
        if self._mean is None:
            raise RuntimeError("regressor must be fitted before prediction")
        return np.full(matrix.shape[0], self._mean, dtype=float)


class RidgeRegressor:
    """Standardized linear ridge regression with an unpenalized intercept."""

    def __init__(self, *, alpha: float = 1e-6) -> None:
        self.alpha = _nonnegative_float(alpha, "alpha")
        self._feature_mean: NDArray[np.float64] | None = None
        self._feature_scale: NDArray[np.float64] | None = None
        self._coefficients: NDArray[np.float64] | None = None

    def fit(self, features: ArrayLike, targets: ArrayLike) -> Self:
        matrix, values = _training_arrays(features, targets)
        mean = np.mean(matrix, axis=0)
        scale = np.std(matrix, axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        standardized = (matrix - mean) / scale
        design = np.column_stack((np.ones(len(values)), standardized))
        penalty = np.eye(design.shape[1], dtype=float) * self.alpha
        penalty[0, 0] = 0.0
        coefficients = np.linalg.pinv(design.T @ design + penalty) @ design.T @ values
        self._feature_mean = mean
        self._feature_scale = scale
        self._coefficients = coefficients
        return self

    def predict(self, features: ArrayLike) -> NDArray[np.float64]:
        if (
            self._feature_mean is None
            or self._feature_scale is None
            or self._coefficients is None
        ):
            raise RuntimeError("regressor must be fitted before prediction")
        matrix = _prediction_matrix(features, len(self._feature_mean))
        standardized = (matrix - self._feature_mean) / self._feature_scale
        design = np.column_stack((np.ones(matrix.shape[0]), standardized))
        return np.asarray(design @ self._coefficients, dtype=float)


@dataclass(frozen=True, slots=True)
class _DecisionStump:
    feature: int
    threshold: float
    left_value: float
    right_value: float

    def predict(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.where(
            features[:, self.feature] <= self.threshold,
            self.left_value,
            self.right_value,
        )


class GradientBoostingRegressor:
    """Squared-error gradient boosting with shallow, interpretable stumps."""

    def __init__(
        self,
        *,
        n_estimators: int = 64,
        learning_rate: float = 0.05,
        max_bins: int = 32,
        min_leaf: int = 2,
    ) -> None:
        self.n_estimators = _positive_integer(n_estimators, "n_estimators")
        self.learning_rate = _positive_float(learning_rate, "learning_rate")
        self.max_bins = _positive_integer(max_bins, "max_bins")
        self.min_leaf = _positive_integer(min_leaf, "min_leaf")
        self._base_value: float | None = None
        self._stumps: tuple[_DecisionStump, ...] = ()
        self.n_features_in_: int | None = None

    def fit(self, features: ArrayLike, targets: ArrayLike) -> Self:
        matrix, values = _training_arrays(features, targets)
        if matrix.shape[0] < 2 * self.min_leaf:
            raise ValueError("training data are too small for configured min_leaf")
        prediction = np.full(len(values), np.mean(values), dtype=float)
        stumps: list[_DecisionStump] = []
        for _ in range(self.n_estimators):
            stump = _best_stump(
                matrix,
                values - prediction,
                max_bins=self.max_bins,
                min_leaf=self.min_leaf,
            )
            if stump is None:
                break
            prediction += self.learning_rate * stump.predict(matrix)
            stumps.append(stump)
        self._base_value = float(np.mean(values))
        self._stumps = tuple(stumps)
        self.n_features_in_ = matrix.shape[1]
        return self

    def predict(self, features: ArrayLike) -> NDArray[np.float64]:
        matrix = _prediction_matrix(features, self.n_features_in_)
        if self._base_value is None:
            raise RuntimeError("regressor must be fitted before prediction")
        prediction = np.full(matrix.shape[0], self._base_value, dtype=float)
        for stump in self._stumps:
            prediction += self.learning_rate * stump.predict(matrix)
        return prediction

    @property
    def fitted_estimator_count(self) -> int:
        return len(self._stumps)


def regression_metrics(targets: ArrayLike, predictions: ArrayLike) -> RegressionMetrics:
    """Return finite regression metrics with an explicit constant-target R² rule."""
    truth = _vector(targets, "targets")
    estimate = _vector(predictions, "predictions")
    if truth.shape != estimate.shape:
        raise ValueError("targets and predictions must have the same shape")
    residuals = truth - estimate
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(np.square(residuals))))
    denominator = float(np.sum(np.square(truth - np.mean(truth))))
    numerator = float(np.sum(np.square(residuals)))
    r2 = 1.0 if denominator == 0.0 and numerator == 0.0 else (
        0.0 if denominator == 0.0 else 1.0 - numerator / denominator
    )
    return RegressionMetrics(len(truth), mae, rmse, r2)


def _best_stump(
    features: NDArray[np.float64],
    residuals: NDArray[np.float64],
    *,
    max_bins: int,
    min_leaf: int,
) -> _DecisionStump | None:
    best: _DecisionStump | None = None
    best_loss = float("inf")
    for feature in range(features.shape[1]):
        column = features[:, feature]
        unique = np.unique(column)
        if len(unique) < 2:
            continue
        thresholds = (unique[:-1] + unique[1:]) / 2.0
        if len(thresholds) > max_bins:
            indices = np.linspace(0, len(thresholds) - 1, max_bins, dtype=int)
            thresholds = thresholds[np.unique(indices)]
        for threshold in thresholds:
            left = column <= threshold
            left_count = int(np.sum(left))
            right_count = len(column) - left_count
            if left_count < min_leaf or right_count < min_leaf:
                continue
            left_value = float(np.mean(residuals[left]))
            right_value = float(np.mean(residuals[~left]))
            loss = float(
                np.sum(np.square(residuals[left] - left_value))
                + np.sum(np.square(residuals[~left] - right_value))
            )
            candidate = _DecisionStump(feature, float(threshold), left_value, right_value)
            if loss < best_loss - 1e-15 or (
                np.isclose(loss, best_loss, rtol=0.0, atol=1e-15)
                and best is not None
                and (candidate.feature, candidate.threshold) < (best.feature, best.threshold)
            ):
                best_loss = loss
                best = candidate
    return best


def _training_arrays(
    features: ArrayLike, targets: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    matrix = _matrix(features)
    values = _vector(targets, "targets")
    if matrix.shape[0] != len(values):
        raise ValueError("features and targets must contain the same number of samples")
    if not len(values):
        raise ValueError("training data must not be empty")
    return matrix, values


def _prediction_matrix(
    features: ArrayLike, expected_features: int | None
) -> NDArray[np.float64]:
    if expected_features is None:
        raise RuntimeError("regressor must be fitted before prediction")
    matrix = _matrix(features)
    if matrix.shape[1] != expected_features:
        raise ValueError("prediction feature count does not match fitted data")
    return matrix


def _matrix(values: ArrayLike) -> NDArray[np.float64]:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("features must be a two-dimensional matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("features must be finite")
    return matrix


def _vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or not len(vector):
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite")
    return vector


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
