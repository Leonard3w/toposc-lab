from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from toposc_lab.geometry import BUILTIN_GEOMETRY_GENERATORS
from toposc_lab.robustness import (
    ROBUSTNESS_REPORT_VERSION,
    DisorderEnsembleRequest,
    DisorderRequest,
    DisorderTarget,
    FiniteSizeRobustnessPoint,
    FiniteSizeScalingResult,
    FiniteSizeScalingSpec,
    FunctionDisorderTransform,
    GeometryFamilySeedPolicy,
    GeometryFamilySpec,
    RobustnessFractionMetric,
    RobustnessReport,
    RobustnessReportEntry,
    RobustnessSuccessCriterion,
    apply_uniform_parameter_perturbation,
    compute_robustness_fraction,
    create_cross_size_geometry_family,
    create_robustness_report,
    estimate_robustness_uncertainty,
    execute_disorder_ensemble,
    fit_finite_size_scaling,
    realize_disorder,
)


def _seeds_with_even_success_count(
    successful_count: int,
    *,
    total_count: int = 10,
) -> tuple[int, ...]:
    return tuple(2 * index for index in range(successful_count)) + tuple(
        2 * index + 1 for index in range(total_count - successful_count)
    )


def _entry(
    successful_count: int,
    *,
    total_count: int = 10,
) -> RobustnessReportEntry:
    request = DisorderEnsembleRequest(
        seeds=_seeds_with_even_success_count(
            successful_count,
            total_count=total_count,
        )
    )
    ensemble = execute_disorder_ensemble(
        request,
        realization_factory=lambda seed: apply_uniform_parameter_perturbation(
            {"mass": 0.5},
            widths={"mass": 0.2},
            seed=seed,
        ),
    )
    metric = compute_robustness_fraction(
        ensemble,
        criterion=RobustnessSuccessCriterion(
            key="even_seed_success",
            description="The fixed report fixture accepts even ensemble seeds.",
            predicate=lambda member: member.seed % 2 == 0,
        ),
    )
    return RobustnessReportEntry(
        ensemble=ensemble,
        metric=metric,
        uncertainty=estimate_robustness_uncertainty(metric),
    )


def _cross_size_components(
    successful_counts: tuple[int, int, int] = (7, 8, 9),
    *,
    sizes: tuple[int, int, int] = (2, 3, 4),
):
    entries = tuple(_entry(count) for count in successful_counts)
    scaling = fit_finite_size_scaling(
        FiniteSizeScalingSpec(
            size_key="linear_size",
            size_description="Explicit square side length.",
            correction_exponent=1.0,
        ),
        points=tuple(
            FiniteSizeRobustnessPoint(size, entry.uncertainty)
            for size, entry in zip(sizes, entries, strict=True)
        ),
    )
    geometries = tuple(
        BUILTIN_GEOMETRY_GENERATORS.generate(
            "square",
            parameters={"n_x": size, "n_y": size, "spacing": 1.0},
        )
        for size in sizes
    )
    family = create_cross_size_geometry_family(
        GeometryFamilySpec(
            family_key="square_report_family",
            family_version=1,
            description="Square family fixture for a robustness report.",
            size_key="linear_size",
            generator_key="square",
            generator_version=1,
            varying_parameter_keys=("n_x", "n_y"),
            seed_policy=GeometryFamilySeedPolicy.NO_SEED,
        ),
        scaling=scaling,
        geometries=geometries,
    )
    return entries, scaling, family


def test_single_size_report_composes_without_copying_source_results() -> None:
    entry = _entry(7)

    report = create_robustness_report(
        "single_size_fixture",
        description="  One explicit single-size robustness report.  ",
        entries=(entry,),
    )

    assert isinstance(report, RobustnessReport)
    assert report.report_version == ROBUSTNESS_REPORT_VERSION
    assert report.description == "One explicit single-size robustness report."
    assert report.entries[0] is entry
    assert report.entries[0].ensemble is entry.ensemble
    assert report.entries[0].metric is entry.metric
    assert report.entries[0].uncertainty is entry.uncertainty
    assert not report.is_cross_size
    assert report.system_sizes is None
    assert report.criterion_key == "even_seed_success"
    assert report.robustness_fractions == pytest.approx((0.7,))
    assert report.successful_counts == (7,)
    assert report.total_counts == (10,)
    assert report.execution_failure_counts == (0,)
    assert report.total_execution_failure_count == 0
    assert report.confidence_intervals == (
        entry.uncertainty.confidence_interval,
    )
    assert "single finite-size" in " ".join(report.warnings)
    first_disorder = report.entries[0].ensemble.members[0].disorder
    assert first_disorder is not None
    assert first_disorder.provenance.seed == entry.ensemble.request.seeds[0]


def test_execution_failures_are_cross_checked_and_reported() -> None:
    request = DisorderEnsembleRequest(seeds=(1, 2, 3))

    def realize(seed: int):
        if seed == 2:
            raise RuntimeError("fixture failure")
        return apply_uniform_parameter_perturbation(
            {"mass": 0.5},
            widths={"mass": 0.2},
            seed=seed,
        )

    ensemble = execute_disorder_ensemble(request, realization_factory=realize)
    metric = compute_robustness_fraction(
        ensemble,
        criterion=RobustnessSuccessCriterion(
            key="completed_realization",
            description="Every completed fixture realization is successful.",
            predicate=lambda member: True,
        ),
    )
    entry = RobustnessReportEntry(
        ensemble=ensemble,
        metric=metric,
        uncertainty=estimate_robustness_uncertainty(metric),
    )
    report = create_robustness_report(
        "failure_fixture",
        description="Report retaining an operational failure.",
        entries=(entry,),
    )

    assert report.robustness_fractions == pytest.approx((2.0 / 3.0,))
    assert report.execution_failure_counts == (1,)
    assert report.total_execution_failure_count == 1
    assert "1 ensemble execution failure(s)" in " ".join(report.warnings)


def test_report_entry_rejects_request_and_failure_partition_mismatches() -> None:
    entry = _entry(2, total_count=3)
    other_metric = RobustnessFractionMetric(
        criterion_key=entry.metric.criterion_key,
        criterion_description=entry.metric.criterion_description,
        request=DisorderEnsembleRequest(seeds=(20, 22, 21)),
        successes=(True, True, False),
    )
    with pytest.raises(ValueError, match="request must match"):
        RobustnessReportEntry(
            ensemble=entry.ensemble,
            metric=other_metric,
            uncertainty=estimate_robustness_uncertainty(other_metric),
        )

    request = DisorderEnsembleRequest(seeds=(1, 2, 3))
    ensemble = execute_disorder_ensemble(
        request,
        realization_factory=lambda seed: (
            (_ for _ in ()).throw(RuntimeError("fixture failure"))
            if seed == 2
            else apply_uniform_parameter_perturbation(
                {"mass": 0.5}, widths={"mass": 0.2}, seed=seed
            )
        ),
    )
    mismatched_metric = RobustnessFractionMetric(
        criterion_key="manual_failure_fixture",
        criterion_description="Manual mismatch fixture.",
        request=request,
        successes=(True, False, True),
    )
    with pytest.raises(ValueError, match="execution failures must match"):
        RobustnessReportEntry(
            ensemble=ensemble,
            metric=mismatched_metric,
            uncertainty=estimate_robustness_uncertainty(mismatched_metric),
        )


def test_report_entry_rejects_uncertainty_for_another_metric() -> None:
    entry = _entry(7)
    other = _entry(8)
    with pytest.raises(ValueError, match="reference the exact entry metric"):
        RobustnessReportEntry(
            ensemble=entry.ensemble,
            metric=entry.metric,
            uncertainty=other.uncertainty,
        )

    equal_but_distinct_metric = RobustnessFractionMetric(
        criterion_key=entry.metric.criterion_key,
        criterion_description=entry.metric.criterion_description,
        request=entry.metric.request,
        successes=entry.metric.successes,
        execution_failure_indices=entry.metric.execution_failure_indices,
    )
    assert equal_but_distinct_metric == entry.metric
    assert equal_but_distinct_metric is not entry.metric
    with pytest.raises(ValueError, match="exact entry metric"):
        RobustnessReportEntry(
            ensemble=entry.ensemble,
            metric=entry.metric,
            uncertainty=estimate_robustness_uncertainty(
                equal_but_distinct_metric
            ),
        )


def test_cross_size_report_requires_complete_ordered_provenance() -> None:
    entries, scaling, family = _cross_size_components()

    report = create_robustness_report(
        "cross_size_fixture",
        description="Cross-size robustness report fixture.",
        entries=entries,
        scaling=scaling,
        geometry_family=family,
    )

    assert report.is_cross_size
    assert report.scaling is scaling
    assert report.geometry_family is family
    assert report.system_sizes == (2.0, 3.0, 4.0)
    assert report.robustness_fractions == pytest.approx((0.7, 0.8, 0.9))
    assert report.successful_counts == (7, 8, 9)
    assert report.total_counts == (10, 10, 10)
    assert report.execution_failure_counts == (0, 0, 0)
    assert "do not prove a thermodynamic" in " ".join(report.warnings)


def test_cross_size_report_requires_scaling_and_family_together() -> None:
    entries, scaling, family = _cross_size_components()
    with pytest.raises(ValueError, match="require both scaling and geometry_family"):
        create_robustness_report(
            "missing_family_fixture",
            description="Invalid cross-size report.",
            entries=entries,
            scaling=scaling,
        )
    with pytest.raises(ValueError, match="require both scaling and geometry_family"):
        create_robustness_report(
            "missing_scaling_fixture",
            description="Invalid cross-size report.",
            entries=entries,
            geometry_family=family,
        )


def test_multiple_entries_are_not_allowed_without_cross_size_context() -> None:
    with pytest.raises(ValueError, match="exactly one entry"):
        create_robustness_report(
            "ambiguous_multi_entry_fixture",
            description="Invalid unscoped multi-entry report.",
            entries=(_entry(7), _entry(8)),
        )


def test_cross_size_entries_must_match_scaling_order() -> None:
    entries, scaling, family = _cross_size_components()
    with pytest.raises(ValueError, match="exact order"):
        create_robustness_report(
            "wrong_order_fixture",
            description="Invalid report entry order.",
            entries=tuple(reversed(entries)),
            scaling=scaling,
            geometry_family=family,
        )


def test_cross_size_report_rejects_family_for_another_scaling_fit() -> None:
    entries, scaling, family = _cross_size_components()
    _, other_scaling, other_family = _cross_size_components(sizes=(2, 4, 8))
    assert other_scaling != scaling

    with pytest.raises(ValueError, match="reference the report scaling"):
        create_robustness_report(
            "wrong_family_fixture",
            description="Family and scaling mismatch fixture.",
            entries=entries,
            scaling=scaling,
            geometry_family=other_family,
        )

    equal_but_distinct_scaling = FiniteSizeScalingResult(
        spec=scaling.spec,
        points=scaling.points,
        method=scaling.method,
    )
    assert equal_but_distinct_scaling == scaling
    assert equal_but_distinct_scaling is not scaling
    with pytest.raises(ValueError, match="reference the report scaling"):
        create_robustness_report(
            "copied_scaling_fixture",
            description="Value-equal but separately constructed scaling fixture.",
            entries=entries,
            scaling=equal_but_distinct_scaling,
            geometry_family=family,
        )


def test_cross_size_report_rejects_reused_entries_and_ensembles() -> None:
    entry = _entry(7)
    _, _, reference_family = _cross_size_components()
    scaling = fit_finite_size_scaling(
        reference_family.scaling.spec,
        points=tuple(
            FiniteSizeRobustnessPoint(size, entry.uncertainty)
            for size in (2, 3, 4)
        ),
    )
    family = create_cross_size_geometry_family(
        reference_family.spec,
        scaling=scaling,
        geometries=tuple(member.geometry for member in reference_family.members),
    )
    with pytest.raises(ValueError, match="distinct entry objects"):
        create_robustness_report(
            "reused_entry_fixture",
            description="Invalid reused entry fixture.",
            entries=(entry, entry, entry),
            scaling=scaling,
            geometry_family=family,
        )

    duplicate_wrappers = tuple(
        RobustnessReportEntry(
            ensemble=entry.ensemble,
            metric=entry.metric,
            uncertainty=entry.uncertainty,
        )
        for _ in range(3)
    )
    with pytest.raises(ValueError, match="distinct ensemble objects"):
        create_robustness_report(
            "reused_ensemble_fixture",
            description="Invalid reused ensemble fixture.",
            entries=duplicate_wrappers,
            scaling=scaling,
            geometry_family=family,
        )


def test_cross_size_report_rejects_mixed_disorder_protocols() -> None:
    entries, _, reference_family = _cross_size_components()
    transform = FunctionDisorderTransform(
        key="alternate_parameter_disorder",
        target=DisorderTarget.MODEL_PARAMETERS,
        function=lambda source, parameters, rng: source,
    )
    request = entries[2].ensemble.request
    ensemble = execute_disorder_ensemble(
        request,
        realization_factory=lambda seed: realize_disorder(
            {"mass": 0.5},
            transform=transform,
            request=DisorderRequest(seed=seed, parameters={"strength": 0.2}),
        ),
    )
    metric = compute_robustness_fraction(
        ensemble,
        criterion=RobustnessSuccessCriterion(
            key="even_seed_success",
            description="The fixed report fixture accepts even ensemble seeds.",
            predicate=lambda member: member.seed % 2 == 0,
        ),
    )
    alternate_entry = RobustnessReportEntry(
        ensemble=ensemble,
        metric=metric,
        uncertainty=estimate_robustness_uncertainty(metric),
    )
    mixed_entries = (*entries[:2], alternate_entry)
    scaling = fit_finite_size_scaling(
        reference_family.scaling.spec,
        points=tuple(
            FiniteSizeRobustnessPoint(size, entry.uncertainty)
            for size, entry in zip((2, 3, 4), mixed_entries, strict=True)
        ),
    )
    family = create_cross_size_geometry_family(
        reference_family.spec,
        scaling=scaling,
        geometries=tuple(member.geometry for member in reference_family.members),
    )

    with pytest.raises(ValueError, match="share one disorder transform"):
        create_robustness_report(
            "mixed_disorder_fixture",
            description="Invalid mixed disorder protocol fixture.",
            entries=mixed_entries,
            scaling=scaling,
            geometry_family=family,
        )


def test_out_of_range_scaling_intercept_is_reported_without_clipping() -> None:
    entries, scaling, family = _cross_size_components(
        (8, 9, 10),
        sizes=(2, 4, 8),
    )
    assert scaling.infinite_size_intercept > 1.0
    report = create_robustness_report(
        "out_of_range_fixture",
        description="Report with an inadmissible extrapolated intercept.",
        entries=entries,
        scaling=scaling,
        geometry_family=family,
    )

    assert report.scaling is not None
    assert report.scaling.infinite_size_intercept > 1.0
    assert "outside the physical unit interval" in " ".join(report.warnings)


@pytest.mark.parametrize("report_key", ["", "Report", "report-key"])
def test_report_requires_a_stable_technical_key(report_key: str) -> None:
    with pytest.raises(ValueError, match="report_key"):
        create_robustness_report(
            report_key,
            description="Valid description.",
            entries=(_entry(7),),
        )


def test_report_is_immutable_and_validates_entry_types() -> None:
    report = create_robustness_report(
        "immutable_fixture",
        description="Immutable report fixture.",
        entries=(_entry(7),),
    )
    with pytest.raises(FrozenInstanceError):
        report.entries = ()  # type: ignore[misc]
    equivalent_fields = create_robustness_report(
        "immutable_fixture",
        description="Immutable report fixture.",
        entries=report.entries,
    )
    assert equivalent_fields != report
    with pytest.raises(TypeError, match="RobustnessReportEntry"):
        create_robustness_report(
            "invalid_entry_fixture",
            description="Invalid entry type fixture.",
            entries=(None,),  # type: ignore[arg-type]
        )
