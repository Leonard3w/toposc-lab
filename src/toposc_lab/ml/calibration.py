"""Validation-only conformal scaling and held-out interval calibration metrics."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from toposc_lab.ml.uncertainty import UncertainPrediction


@dataclass(frozen=True, slots=True)
class PredictionInterval:
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    nominal_coverage: float
    calibration_quantile: float

    def __post_init__(self) -> None:
        lower = np.array(self.lower, dtype=float, copy=True)
        upper = np.array(self.upper, dtype=float, copy=True)
        if lower.ndim != 1 or upper.shape != lower.shape or not len(lower):
            raise ValueError("interval bounds must be aligned non-empty vectors")
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("interval bounds must be finite")
        if np.any(lower > upper):
            raise ValueError("interval lower bounds cannot exceed upper bounds")
        lower.setflags(write=False)
        upper.setflags(write=False)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


@dataclass(frozen=True, slots=True)
class CalibrationMetrics:
    sample_count: int
    nominal_coverage: float
    empirical_coverage: float
    absolute_coverage_error: float
    mean_interval_width: float
    median_interval_width: float
    failure_modes: tuple[str, ...]


class IntervalCalibrator:
    """Scale raw uncertainty using validation residuals, never test residuals."""

    def __init__(self, *, coverage: float = 0.9) -> None:
        self.coverage = _coverage(coverage)
        self._quantile: float | None = None
        self._validation_count: int | None = None
        self._zero_scale_count: int | None = None

    def fit(self, targets: ArrayLike, prediction: UncertainPrediction) -> IntervalCalibrator:
        truth = _targets(targets, prediction)
        scale = prediction.standard_deviation
        positive = scale > np.finfo(float).eps
        safe_scale = np.maximum(scale, np.finfo(float).eps)
        scores = np.abs(truth - prediction.mean) / safe_scale
        rank = min(len(scores), int(np.ceil((len(scores) + 1) * self.coverage)))
        quantile = float(np.partition(scores, rank - 1)[rank - 1])
        self._quantile = max(quantile, np.finfo(float).eps)
        self._validation_count = len(scores)
        self._zero_scale_count = int(np.sum(~positive))
        return self

    def predict_interval(self, prediction: UncertainPrediction) -> PredictionInterval:
        if self._quantile is None:
            raise RuntimeError("calibrator must be fitted on validation data first")
        half_width = self._quantile * prediction.standard_deviation
        return PredictionInterval(
            lower=prediction.mean - half_width,
            upper=prediction.mean + half_width,
            nominal_coverage=self.coverage,
            calibration_quantile=self._quantile,
        )

    @property
    def validation_count(self) -> int:
        if self._validation_count is None:
            raise RuntimeError("calibrator has not been fitted")
        return self._validation_count

    @property
    def failure_modes(self) -> tuple[str, ...]:
        if self._validation_count is None or self._zero_scale_count is None:
            raise RuntimeError("calibrator has not been fitted")
        failures: list[str] = []
        if self._validation_count < 20:
            failures.append("fewer than 20 validation samples; coverage estimate is coarse")
        if self._zero_scale_count:
            failures.append(
                f"{self._zero_scale_count} validation uncertainties were numerically zero"
            )
        return tuple(failures)


def calibration_metrics(
    targets: ArrayLike,
    interval: PredictionInterval,
    *,
    inherited_failure_modes: tuple[str, ...] = (),
) -> CalibrationMetrics:
    """Measure interval coverage on data not used to fit the calibrator."""
    truth = np.asarray(targets, dtype=float)
    if truth.shape != interval.lower.shape or not np.all(np.isfinite(truth)):
        raise ValueError("targets must be finite and aligned with interval bounds")
    covered = (truth >= interval.lower) & (truth <= interval.upper)
    empirical = float(np.mean(covered))
    widths = interval.upper - interval.lower
    failures = list(inherited_failure_modes)
    if len(truth) < 20:
        failures.append("fewer than 20 evaluation samples; empirical coverage is coarse")
    if np.allclose(widths, 0.0):
        failures.append("all prediction intervals have zero width")
    return CalibrationMetrics(
        sample_count=len(truth),
        nominal_coverage=interval.nominal_coverage,
        empirical_coverage=empirical,
        absolute_coverage_error=abs(empirical - interval.nominal_coverage),
        mean_interval_width=float(np.mean(widths)),
        median_interval_width=float(np.median(widths)),
        failure_modes=tuple(failures),
    )


def _targets(values: ArrayLike, prediction: UncertainPrediction) -> np.ndarray:
    targets = np.asarray(values, dtype=float)
    if (
        targets.shape != prediction.mean.shape
        or not len(targets)
        or not np.all(np.isfinite(targets))
    ):
        raise ValueError("targets must be finite and aligned with predictions")
    return targets


def _coverage(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("coverage must be a real number")
    result = float(value)
    if not np.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError("coverage must be strictly between zero and one")
    return result
