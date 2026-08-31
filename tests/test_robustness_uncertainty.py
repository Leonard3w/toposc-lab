from __future__ import annotations

from dataclasses import FrozenInstanceError
import math

import pytest

from toposc_lab.robustness import (
    ROBUSTNESS_UNCERTAINTY_VERSION,
    DisorderEnsembleRequest,
    RobustnessFractionMetric,
    RobustnessUncertaintyEstimate,
    RobustnessUncertaintyMethod,
    estimate_robustness_uncertainty,
)


def _metric(
    successes: tuple[bool, ...],
    *,
    execution_failure_indices: tuple[int, ...] = (),
) -> RobustnessFractionMetric:
    return RobustnessFractionMetric(
        criterion_key="phase_8_11_fixture",
        criterion_description="An explicit success criterion for uncertainty tests.",
        request=DisorderEnsembleRequest(seeds=tuple(range(10, 10 + len(successes)))),
        successes=successes,
        execution_failure_indices=execution_failure_indices,
    )


def test_wilson_estimate_retains_the_complete_robustness_metric() -> None:
    metric = _metric((True, False, True, False))

    estimate = estimate_robustness_uncertainty(metric)

    assert isinstance(estimate, RobustnessUncertaintyEstimate)
    assert estimate.metric is metric
    assert estimate.method is RobustnessUncertaintyMethod.WILSON_SCORE
    assert estimate.uncertainty_version == ROBUSTNESS_UNCERTAINTY_VERSION
    assert estimate.confidence_level == 0.95
    assert estimate.sample_size == 4
    assert estimate.successful_count == 2
    assert estimate.observed_fraction == 0.5
    assert estimate.standard_error == 0.25
    assert estimate.critical_value == pytest.approx(1.959963984540054)
    assert estimate.confidence_interval == pytest.approx(
        (0.15003898915214947, 0.8499610108478506)
    )
    assert estimate.lower_bound == estimate.confidence_interval[0]
    assert estimate.upper_bound == estimate.confidence_interval[1]


@pytest.mark.parametrize(
    ("successes", "expected_bound"),
    [
        ((False, False, False), 0.5614970317550454),
        ((True, True, True), 0.4385029682449546),
    ],
)
def test_wilson_interval_is_non_degenerate_at_fraction_endpoints(
    successes: tuple[bool, ...],
    expected_bound: float,
) -> None:
    estimate = estimate_robustness_uncertainty(_metric(successes))

    assert estimate.standard_error == 0.0
    assert 0.0 <= estimate.lower_bound <= estimate.observed_fraction
    assert estimate.observed_fraction <= estimate.upper_bound <= 1.0
    if estimate.observed_fraction == 0.0:
        assert estimate.lower_bound == 0.0
        assert estimate.upper_bound == pytest.approx(expected_bound)
    else:
        assert estimate.lower_bound == pytest.approx(expected_bound)
        assert estimate.upper_bound == 1.0


def test_execution_failures_keep_phase_8_10_denominator_semantics() -> None:
    metric = _metric(
        (True, False, True),
        execution_failure_indices=(1,),
    )

    estimate = estimate_robustness_uncertainty(metric)

    assert estimate.sample_size == 3
    assert estimate.successful_count == 2
    assert estimate.observed_fraction == pytest.approx(2.0 / 3.0)
    assert estimate.metric.execution_failure_indices == (1,)
    assert estimate.metric.execution_failure_seeds == (11,)


def test_higher_confidence_level_produces_a_wider_interval() -> None:
    metric = _metric((True, False, True, False, True))
    estimate_80 = estimate_robustness_uncertainty(
        metric,
        confidence_level=0.8,
    )
    estimate_99 = estimate_robustness_uncertainty(
        metric,
        confidence_level=0.99,
    )

    width_80 = estimate_80.upper_bound - estimate_80.lower_bound
    width_99 = estimate_99.upper_bound - estimate_99.lower_bound
    assert width_99 > width_80


@pytest.mark.parametrize("confidence_level", [0.0, 1.0, -0.1, 1.1, math.inf, math.nan])
def test_confidence_level_must_be_finite_and_inside_the_unit_interval(
    confidence_level: float,
) -> None:
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        estimate_robustness_uncertainty(
            _metric((True,)),
            confidence_level=confidence_level,
        )


@pytest.mark.parametrize("confidence_level", [True, "0.95", None])
def test_confidence_level_requires_a_real_number(confidence_level: object) -> None:
    with pytest.raises(TypeError, match="real number"):
        estimate_robustness_uncertainty(
            _metric((True,)),
            confidence_level=confidence_level,  # type: ignore[arg-type]
        )


def test_uncertainty_contract_rejects_wrong_metric_and_method_types() -> None:
    with pytest.raises(TypeError, match="metric must be"):
        estimate_robustness_uncertainty(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="method must be"):
        estimate_robustness_uncertainty(
            _metric((True,)),
            method="wilson_score",  # type: ignore[arg-type]
        )


def test_uncertainty_estimate_is_immutable() -> None:
    estimate = estimate_robustness_uncertainty(_metric((True, False)))

    with pytest.raises(FrozenInstanceError):
        estimate.confidence_level = 0.9  # type: ignore[misc]
