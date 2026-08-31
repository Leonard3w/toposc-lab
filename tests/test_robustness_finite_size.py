from __future__ import annotations

from dataclasses import FrozenInstanceError
import math

import pytest

from toposc_lab.robustness import (
    FINITE_SIZE_SCALING_VERSION,
    DisorderEnsembleRequest,
    FiniteSizeRobustnessPoint,
    FiniteSizeScalingMethod,
    FiniteSizeScalingResult,
    FiniteSizeScalingSpec,
    RobustnessFractionMetric,
    estimate_robustness_uncertainty,
    fit_finite_size_scaling,
)


def _uncertainty(
    successful_count: int,
    total_count: int,
    *,
    criterion_key: str = "fixed_success_policy",
    criterion_description: str = "One fixed success policy across sizes.",
    confidence_level: float = 0.95,
    execution_failure_indices: tuple[int, ...] = (),
):
    successes = (True,) * successful_count + (False,) * (
        total_count - successful_count
    )
    metric = RobustnessFractionMetric(
        criterion_key=criterion_key,
        criterion_description=criterion_description,
        request=DisorderEnsembleRequest(seeds=tuple(range(total_count))),
        successes=successes,
        execution_failure_indices=execution_failure_indices,
    )
    return estimate_robustness_uncertainty(
        metric,
        confidence_level=confidence_level,
    )


def _point(system_size: float, successful_count: int, total_count: int = 20):
    return FiniteSizeRobustnessPoint(
        system_size=system_size,
        uncertainty=_uncertainty(successful_count, total_count),
    )


def _spec(*, correction_exponent: float = 1.0) -> FiniteSizeScalingSpec:
    return FiniteSizeScalingSpec(
        size_key="site_count_fixture",
        size_description="Explicit fixture size; no geometry inference.",
        correction_exponent=correction_exponent,
    )


def test_exact_leading_inverse_size_scaling_is_recovered() -> None:
    points = (
        _point(2, 16),
        _point(4, 14),
        _point(8, 13),
    )

    result = fit_finite_size_scaling(_spec(), points=points)

    assert isinstance(result, FiniteSizeScalingResult)
    assert result.points == points
    assert result.method is (
        FiniteSizeScalingMethod.ORDINARY_LEAST_SQUARES_LEADING_POWER
    )
    assert result.scaling_version == FINITE_SIZE_SCALING_VERSION
    assert result.system_sizes == (2.0, 4.0, 8.0)
    assert result.scaling_coordinates == pytest.approx((0.5, 0.25, 0.125))
    assert result.observed_fractions == pytest.approx((0.8, 0.7, 0.65))
    assert result.infinite_size_intercept == pytest.approx(0.6)
    assert result.finite_size_coefficient == pytest.approx(0.4)
    assert result.fitted_fractions == pytest.approx(result.observed_fractions)
    assert result.residuals == pytest.approx((0.0, 0.0, 0.0), abs=1.0e-15)
    assert result.residual_sum_squares == pytest.approx(0.0, abs=1.0e-30)
    assert result.r_squared == pytest.approx(1.0)
    assert result.infinite_size_intercept_in_unit_interval


def test_correction_exponent_is_explicitly_applied() -> None:
    result = fit_finite_size_scaling(
        _spec(correction_exponent=2.0),
        points=(
            _point(2, 14),
            _point(4, 11),
            _point(8, 10),
        ),
    )

    assert result.scaling_coordinates == pytest.approx((0.25, 0.0625, 0.015625))


def test_constant_observations_have_no_defined_r_squared() -> None:
    result = fit_finite_size_scaling(
        _spec(),
        points=(_point(2, 10), _point(4, 10), _point(8, 10)),
    )

    assert result.infinite_size_intercept == pytest.approx(0.5)
    assert result.finite_size_coefficient == pytest.approx(0.0)
    assert result.residuals == pytest.approx((0.0, 0.0, 0.0))
    assert result.r_squared is None


def test_out_of_range_intercept_is_exposed_without_clipping() -> None:
    result = fit_finite_size_scaling(
        _spec(),
        points=(_point(2, 8, 10), _point(4, 9, 10), _point(8, 10, 10)),
    )

    assert result.infinite_size_intercept > 1.0
    assert not result.infinite_size_intercept_in_unit_interval


def test_execution_failures_remain_in_the_nested_metric() -> None:
    failed_uncertainty = _uncertainty(
        2,
        3,
        execution_failure_indices=(2,),
    )
    points = (
        FiniteSizeRobustnessPoint(2, failed_uncertainty),
        _point(4, 14),
        _point(8, 13),
    )

    result = fit_finite_size_scaling(_spec(), points=points)

    retained_metric = result.points[0].uncertainty.metric
    assert retained_metric.total_count == 3
    assert retained_metric.execution_failure_indices == (2,)
    assert result.observed_fractions[0] == pytest.approx(2.0 / 3.0)


@pytest.mark.parametrize(
    "points",
    [
        (_point(2, 16), _point(4, 14)),
        (_point(2, 16), _point(2, 14), _point(8, 13)),
        (_point(4, 14), _point(2, 16), _point(8, 13)),
    ],
)
def test_fit_requires_three_strictly_increasing_sizes(
    points: tuple[FiniteSizeRobustnessPoint, ...],
) -> None:
    match = "at least three" if len(points) < 3 else "strictly increasing"
    with pytest.raises(ValueError, match=match):
        fit_finite_size_scaling(_spec(), points=points)


def test_fit_rejects_numerically_unresolvable_scaling_coordinates() -> None:
    with pytest.raises(ValueError, match="not numerically resolvable"):
        fit_finite_size_scaling(
            _spec(),
            points=(
                _point(1.0e170, 16),
                _point(2.0e170, 14),
                _point(3.0e170, 13),
            ),
        )


def test_fit_requires_one_success_criterion_across_sizes() -> None:
    incompatible = FiniteSizeRobustnessPoint(
        8,
        _uncertainty(
            13,
            20,
            criterion_key="changed_success_policy",
        ),
    )

    with pytest.raises(ValueError, match="same robustness success criterion"):
        fit_finite_size_scaling(
            _spec(),
            points=(_point(2, 16), _point(4, 14), incompatible),
        )


def test_fit_requires_one_uncertainty_contract_across_sizes() -> None:
    incompatible = FiniteSizeRobustnessPoint(
        8,
        _uncertainty(13, 20, confidence_level=0.9),
    )

    with pytest.raises(ValueError, match="same uncertainty contract"):
        fit_finite_size_scaling(
            _spec(),
            points=(_point(2, 16), _point(4, 14), incompatible),
        )


@pytest.mark.parametrize("value", [0.0, -1.0, math.inf, math.nan])
def test_size_and_exponent_must_be_positive_and_finite(value: float) -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        FiniteSizeRobustnessPoint(value, _uncertainty(1, 2))
    with pytest.raises(ValueError, match="strictly positive"):
        _spec(correction_exponent=value)


@pytest.mark.parametrize("value", [True, "4", None])
def test_size_and_exponent_must_be_real(value: object) -> None:
    with pytest.raises(TypeError, match="real number"):
        FiniteSizeRobustnessPoint(
            value,  # type: ignore[arg-type]
            _uncertainty(1, 2),
        )
    with pytest.raises(TypeError, match="real number"):
        _spec(correction_exponent=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("size_key", ["", "site-count", "site count"])
def test_scaling_spec_requires_a_stable_size_key(size_key: str) -> None:
    with pytest.raises(ValueError, match="size_key"):
        FiniteSizeScalingSpec(
            size_key=size_key,
            size_description="Valid description.",
            correction_exponent=1.0,
        )


def test_scaling_contracts_are_immutable_and_validate_types() -> None:
    point = _point(2, 16)
    with pytest.raises(FrozenInstanceError):
        point.system_size = 3.0  # type: ignore[misc]
    with pytest.raises(TypeError, match="uncertainty must be"):
        FiniteSizeRobustnessPoint(2, None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="spec must be"):
        fit_finite_size_scaling(
            None,  # type: ignore[arg-type]
            points=(_point(2, 16), _point(4, 14), _point(8, 13)),
        )
    with pytest.raises(TypeError, match="method must be"):
        fit_finite_size_scaling(
            _spec(),
            points=(_point(2, 16), _point(4, 14), _point(8, 13)),
            method="ordinary_least_squares_leading_power",  # type: ignore[arg-type]
        )
