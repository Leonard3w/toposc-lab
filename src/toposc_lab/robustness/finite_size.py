"""Explicit finite-size fits for robustness success fractions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from numbers import Real

from toposc_lab.robustness.uncertainty import RobustnessUncertaintyEstimate

FINITE_SIZE_SCALING_VERSION = 1


class FiniteSizeScalingMethod(str, Enum):
    """Supported numerical model for finite-size robustness data."""

    ORDINARY_LEAST_SQUARES_LEADING_POWER = (
        "ordinary_least_squares_leading_power"
    )


@dataclass(frozen=True, slots=True)
class FiniteSizeScalingSpec:
    """Caller-declared size meaning and leading correction exponent."""

    size_key: str
    size_description: str
    correction_exponent: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "size_key", _identifier(self.size_key, name="size_key"))
        object.__setattr__(
            self,
            "size_description",
            _description(self.size_description, name="size_description"),
        )
        object.__setattr__(
            self,
            "correction_exponent",
            _positive_finite_real(
                self.correction_exponent,
                name="correction_exponent",
            ),
        )


@dataclass(frozen=True, slots=True)
class FiniteSizeRobustnessPoint:
    """One explicit system size and its existing robustness uncertainty."""

    system_size: float
    uncertainty: RobustnessUncertaintyEstimate

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "system_size",
            _positive_finite_real(self.system_size, name="system_size"),
        )
        if not isinstance(self.uncertainty, RobustnessUncertaintyEstimate):
            raise TypeError("uncertainty must be RobustnessUncertaintyEstimate")


@dataclass(frozen=True, slots=True)
class FiniteSizeScalingResult:
    """Auditable leading-power fit over compatible robustness estimates."""

    spec: FiniteSizeScalingSpec
    points: tuple[FiniteSizeRobustnessPoint, ...]
    method: FiniteSizeScalingMethod = (
        FiniteSizeScalingMethod.ORDINARY_LEAST_SQUARES_LEADING_POWER
    )
    scaling_version: int = field(
        default=FINITE_SIZE_SCALING_VERSION,
        init=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.spec, FiniteSizeScalingSpec):
            raise TypeError("spec must be FiniteSizeScalingSpec")
        points = tuple(self.points)
        if any(not isinstance(point, FiniteSizeRobustnessPoint) for point in points):
            raise TypeError("points must contain only FiniteSizeRobustnessPoint values")
        if len(points) < 3:
            raise ValueError("finite-size scaling requires at least three size points")
        sizes = tuple(point.system_size for point in points)
        if any(current >= following for current, following in zip(sizes, sizes[1:])):
            raise ValueError("system sizes must be strictly increasing")
        if not isinstance(self.method, FiniteSizeScalingMethod):
            raise TypeError("method must be FiniteSizeScalingMethod")

        _validate_compatible_uncertainty(points)
        coordinates = tuple(
            _scaling_coordinate(size, exponent=self.spec.correction_exponent)
            for size in sizes
        )
        if len(set(coordinates)) != len(coordinates):
            raise ValueError("system sizes do not produce distinct scaling coordinates")
        _coordinate_moments(coordinates)

        object.__setattr__(self, "points", points)

    @property
    def system_sizes(self) -> tuple[float, ...]:
        """Explicit caller-supplied system sizes in ascending order."""
        return tuple(point.system_size for point in self.points)

    @property
    def scaling_coordinates(self) -> tuple[float, ...]:
        """Coordinates ``system_size ** (-correction_exponent)`` used by the fit."""
        return tuple(
            _scaling_coordinate(
                point.system_size,
                exponent=self.spec.correction_exponent,
            )
            for point in self.points
        )

    @property
    def observed_fractions(self) -> tuple[float, ...]:
        """Phase-8.10 success fractions retained by the input estimates."""
        return tuple(point.uncertainty.observed_fraction for point in self.points)

    @property
    def infinite_size_intercept(self) -> float:
        """Unclipped fitted intercept at a zero scaling coordinate."""
        intercept, _, _ = self._fit()
        return intercept

    @property
    def finite_size_coefficient(self) -> float:
        """Fitted coefficient multiplying the leading size correction."""
        _, coefficient, _ = self._fit()
        return coefficient

    @property
    def fitted_fractions(self) -> tuple[float, ...]:
        """Fitted values at the supplied finite sizes."""
        intercept, coefficient, _ = self._fit()
        return tuple(
            intercept + coefficient * coordinate
            for coordinate in self.scaling_coordinates
        )

    @property
    def residuals(self) -> tuple[float, ...]:
        """Observed minus fitted robustness fractions in point order."""
        _, _, residuals = self._fit()
        return residuals

    @property
    def residual_sum_squares(self) -> float:
        """Unweighted residual sum of squares."""
        return math.fsum(residual * residual for residual in self.residuals)

    @property
    def r_squared(self) -> float | None:
        """Coefficient of determination, or ``None`` for constant observations."""
        observations = self.observed_fractions
        mean_observation = math.fsum(observations) / len(observations)
        total_sum_squares = math.fsum(
            (observation - mean_observation) ** 2
            for observation in observations
        )
        if total_sum_squares == 0.0:
            return None
        return 1.0 - self.residual_sum_squares / total_sum_squares

    @property
    def infinite_size_intercept_in_unit_interval(self) -> bool:
        """Whether the raw extrapolated fraction lies in its physical value range."""
        return 0.0 <= self.infinite_size_intercept <= 1.0

    def _fit(self) -> tuple[float, float, tuple[float, ...]]:
        coordinates = self.scaling_coordinates
        observations = self.observed_fractions
        count = len(coordinates)
        mean_coordinate, centered_coordinate_sum = _coordinate_moments(coordinates)
        mean_observation = math.fsum(observations) / count
        coefficient = math.fsum(
            (coordinate - mean_coordinate) * (observation - mean_observation)
            for coordinate, observation in zip(
                coordinates,
                observations,
                strict=True,
            )
        ) / centered_coordinate_sum
        intercept = mean_observation - coefficient * mean_coordinate
        residuals = tuple(
            observation - (intercept + coefficient * coordinate)
            for coordinate, observation in zip(
                coordinates,
                observations,
                strict=True,
            )
        )
        return intercept, coefficient, residuals


def fit_finite_size_scaling(
    spec: FiniteSizeScalingSpec,
    *,
    points: tuple[FiniteSizeRobustnessPoint, ...],
    method: FiniteSizeScalingMethod = (
        FiniteSizeScalingMethod.ORDINARY_LEAST_SQUARES_LEADING_POWER
    ),
) -> FiniteSizeScalingResult:
    """Fit one explicit leading-power correction to size-labelled estimates."""
    return FiniteSizeScalingResult(
        spec=spec,
        points=points,
        method=method,
    )


def _validate_compatible_uncertainty(
    points: tuple[FiniteSizeRobustnessPoint, ...],
) -> None:
    reference = points[0].uncertainty
    for point in points[1:]:
        uncertainty = point.uncertainty
        if (
            uncertainty.metric.criterion_key != reference.metric.criterion_key
            or uncertainty.metric.criterion_description
            != reference.metric.criterion_description
        ):
            raise ValueError("all points must use the same robustness success criterion")
        if (
            uncertainty.confidence_level != reference.confidence_level
            or uncertainty.method is not reference.method
            or uncertainty.uncertainty_version != reference.uncertainty_version
        ):
            raise ValueError("all points must use the same uncertainty contract")


def _scaling_coordinate(system_size: float, *, exponent: float) -> float:
    try:
        coordinate = math.pow(system_size, -exponent)
    except OverflowError as error:
        raise ValueError("system size and exponent produce an invalid scaling coordinate") from error
    if not math.isfinite(coordinate) or coordinate <= 0.0:
        raise ValueError("system size and exponent produce an invalid scaling coordinate")
    return coordinate


def _coordinate_moments(coordinates: tuple[float, ...]) -> tuple[float, float]:
    count = len(coordinates)
    try:
        mean_coordinate = math.fsum(
            coordinate / count for coordinate in coordinates
        )
        centered_sum = math.fsum(
            (coordinate - mean_coordinate) ** 2 for coordinate in coordinates
        )
    except OverflowError as error:
        raise ValueError(
            "scaling coordinates are not numerically resolvable"
        ) from error
    if not math.isfinite(centered_sum) or centered_sum <= 0.0:
        raise ValueError("scaling coordinates are not numerically resolvable")
    return mean_coordinate, centered_sum


def _identifier(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value.isidentifier():
        raise ValueError(f"{name} must be a Python-style identifier")
    return value


def _description(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _positive_finite_real(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return result
