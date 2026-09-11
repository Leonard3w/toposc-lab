"""Predictive uncertainty estimates and held-out usefulness diagnostics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from toposc_lab.ml.models import Regressor, RidgeRegressor


@dataclass(frozen=True, slots=True)
class UncertainPrediction:
    mean: NDArray[np.float64]
    standard_deviation: NDArray[np.float64]
    ensemble_standard_deviation: NDArray[np.float64]
    nearest_training_distance: NDArray[np.float64]

    def __post_init__(self) -> None:
        arrays = tuple(
            np.array(value, dtype=float, copy=True)
            for value in (
                self.mean,
                self.standard_deviation,
                self.ensemble_standard_deviation,
                self.nearest_training_distance,
            )
        )
        if any(array.ndim != 1 for array in arrays) or len(
            {array.shape for array in arrays}
        ) != 1:
            raise ValueError("uncertainty arrays must be aligned vectors")
        if any(not np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("uncertainty arrays must be finite")
        if any(np.any(array < 0.0) for array in arrays[1:]):
            raise ValueError("uncertainties/distances must be nonnegative")
        names = (
            "mean",
            "standard_deviation",
            "ensemble_standard_deviation",
            "nearest_training_distance",
        )
        for name, array in zip(names, arrays, strict=True):
            array.setflags(write=False)
            object.__setattr__(self, name, array)


@dataclass(frozen=True, slots=True)
class UncertaintyDiagnostics:
    sample_count: int
    mean_uncertainty: float
    mean_absolute_error: float
    uncertainty_error_rank_correlation: float
    low_uncertainty_mae: float
    high_uncertainty_mae: float
    high_uncertainty_has_larger_error: bool


class BootstrapUncertaintyRegressor:
    """Bootstrap ensemble plus standardized distance-based epistemic signal."""

    def __init__(
        self,
        *,
        model_factory: Callable[[], Regressor] | None = None,
        ensemble_size: int = 16,
        seed: int = 1209,
        distance_weight: float = 0.25,
    ) -> None:
        self.model_factory = model_factory or (lambda: RidgeRegressor(alpha=1e-4))
        self.ensemble_size = _positive_integer(ensemble_size, "ensemble_size")
        self.seed = _nonnegative_integer(seed, "seed")
        self.distance_weight = _nonnegative_float(distance_weight, "distance_weight")
        self._models: tuple[Regressor, ...] = ()
        self._feature_mean: NDArray[np.float64] | None = None
        self._feature_scale: NDArray[np.float64] | None = None
        self._training_features: NDArray[np.float64] | None = None
        self._residual_scale: float | None = None

    def fit(self, features: ArrayLike, targets: ArrayLike) -> Self:
        matrix, values = _training_arrays(features, targets)
        mean = np.mean(matrix, axis=0)
        scale = np.std(matrix, axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        standardized = (matrix - mean) / scale
        generator = np.random.default_rng(self.seed)
        models: list[Regressor] = []
        training_predictions: list[NDArray[np.float64]] = []
        for _ in range(self.ensemble_size):
            indices = generator.integers(0, len(values), size=len(values))
            model = self.model_factory()
            model.fit(matrix[indices], values[indices])
            models.append(model)
            training_predictions.append(model.predict(matrix))
        ensemble_mean = np.mean(np.vstack(training_predictions), axis=0)
        residual_scale = float(np.sqrt(np.mean(np.square(values - ensemble_mean))))
        self._models = tuple(models)
        self._feature_mean = mean
        self._feature_scale = scale
        self._training_features = standardized
        self._residual_scale = max(residual_scale, np.finfo(float).eps)
        return self

    def predict(self, features: ArrayLike) -> NDArray[np.float64]:
        return self.predict_with_uncertainty(features).mean

    def predict_with_uncertainty(self, features: ArrayLike) -> UncertainPrediction:
        if (
            not self._models
            or self._feature_mean is None
            or self._feature_scale is None
            or self._training_features is None
            or self._residual_scale is None
        ):
            raise RuntimeError("uncertainty regressor must be fitted before prediction")
        matrix = _matrix(features)
        if matrix.shape[1] != len(self._feature_mean):
            raise ValueError("prediction feature count does not match fitted data")
        members = np.vstack(tuple(model.predict(matrix) for model in self._models))
        mean = np.mean(members, axis=0)
        ensemble_std = np.std(members, axis=0, ddof=1 if len(self._models) > 1 else 0)
        standardized = (matrix - self._feature_mean) / self._feature_scale
        distances = np.sqrt(
            np.sum(
                np.square(standardized[:, None, :] - self._training_features[None, :, :]),
                axis=2,
            )
        ) / np.sqrt(max(1, matrix.shape[1]))
        nearest = np.min(distances, axis=1)
        distance_scale = self._residual_scale * (1.0 + self.distance_weight * nearest)
        total = np.sqrt(np.square(ensemble_std) + np.square(distance_scale))
        return UncertainPrediction(mean, total, ensemble_std, nearest)


def uncertainty_diagnostics(
    targets: ArrayLike, prediction: UncertainPrediction
) -> UncertaintyDiagnostics:
    """Check whether held-out errors tend to rise with predicted uncertainty."""
    truth = np.asarray(targets, dtype=float)
    if (
        truth.shape != prediction.mean.shape
        or not len(truth)
        or not np.all(np.isfinite(truth))
    ):
        raise ValueError("targets must be finite and aligned with predictions")
    errors = np.abs(truth - prediction.mean)
    uncertainty = prediction.standard_deviation
    correlation = _rank_correlation(uncertainty, errors)
    order = np.argsort(uncertainty, kind="stable")
    midpoint = max(1, len(order) // 2)
    low = order[:midpoint]
    high = order[midpoint:]
    low_mae = float(np.mean(errors[low]))
    high_mae = low_mae if not len(high) else float(np.mean(errors[high]))
    return UncertaintyDiagnostics(
        sample_count=len(truth),
        mean_uncertainty=float(np.mean(uncertainty)),
        mean_absolute_error=float(np.mean(errors)),
        uncertainty_error_rank_correlation=correlation,
        low_uncertainty_mae=low_mae,
        high_uncertainty_mae=high_mae,
        high_uncertainty_has_larger_error=high_mae >= low_mae,
    )


def _rank_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_ranks = _ranks(first)
    second_ranks = _ranks(second)
    if np.std(first_ranks) == 0.0 or np.std(second_ranks) == 0.0:
        return 0.0
    return float(np.corrcoef(first_ranks, second_ranks)[0, 1])


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    for value in np.unique(values):
        indices = np.flatnonzero(values == value)
        ranks[indices] = np.mean(ranks[indices])
    return ranks


def _training_arrays(
    features: ArrayLike, targets: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    matrix = _matrix(features)
    values = np.asarray(targets, dtype=float)
    if values.shape != (len(matrix),) or not np.all(np.isfinite(values)):
        raise ValueError("targets must be one finite value per feature row")
    return matrix, values


def _matrix(values: ArrayLike) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or not len(matrix) or not np.all(np.isfinite(matrix)):
        raise ValueError("features must be a non-empty finite matrix")
    return matrix


def _positive_integer(value: object, name: str) -> int:
    result = _nonnegative_integer(value, name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _nonnegative_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result
